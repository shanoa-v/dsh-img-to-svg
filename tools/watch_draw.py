"""watch_draw.py -- SVG 复刻全过程可视化插件（莱茵生命风格）

## 用途

用户想"看绘制过程"时调用。自动打开浏览器，实时显示：
  - 左中：工作区，逐阶段/逐层呈现绘制进度
  - 右侧：参数面板 + 阶段说明 + 进度日志
  - 底部：时间轴（可拖动 / 左右箭头逐帧），计时器

## 适用档位

任何档位（1看过程 / 2最高质量 / 3局部重绘 / 4上采样）都适用。
本插件只负责"展示"，不改变管线结果。

## 架构（关键设计）

**矛盾**：`vec.py`（权威实现）不输出任何中间产物；
`live_vec_server.py` 能逐层展示，但它是**自己重新实现的管线**，
结果与 vec.py 不完全一致（少了 BASECOL/SEAL 等修复）。

**解法：双轨**
  1. 用真正的 `vec.py` 跑出**权威最终 SVG**（与正式交付完全一致）
  2. 并行用轻量流程生成**可视化中间态**（平滑图 / 量化图 / 逐层 SVG）
     仅用于"看"，不参与交付
  3. 时间轴帧 = [原图] + [各阶段中间态] + [最终 SVG]

这样"看到的最终结果"就是真结果，中间过程如实反映管线的实际阶段。

## 帧序列（与管线阶段一一对应）

    00 SOURCE      原图
    01 RESIZE      缩放至工作分辨率
    02 SMOOTH      2x3 中值（平滑）
    03 EDGE        保边混合 w=EDGE_KEEP
    04 QUANTIZE    Oklab KMeans 量化（N 色）
    05 MERGE       簇合并 / 碎块清理（示意：色数统计）
    06 TRACE-L1.. 逐层描摹（每层一帧）
    99 FINAL       vec.py 权威产出（渲染校验图）

## 风格

参考 rhine-lock（明日方舟莱茵生命）：
  黑底 #0a0b0e、琥珀橙 #f5a623 强调、等宽字体、细线框、
  编号标签（SEC.01 / 02）、扫描线纹理、技术数据面板

## 调用

    python watch_draw.py "<源图>" [--port 8899] [--colors 56] [--edge 0.16]
                                [--k 1] [--no-open]

防火墙/端口占用会自动 +1 重试。
"""
import os
import sys
import json
import time
import math
import shutil
import argparse
import threading
import webbrowser
import subprocess
import io
import http.server
import socketserver
from urllib.parse import urlparse, parse_qs

import numpy as np
from PIL import Image, ImageFilter, ImageDraw

# --- 脚本目录解析：自身路径优先，DSH_HOME 作后备 ---
# 本文件位于 <skill>/tools/watch_draw.py，因此 <skill>/scripts 可直接推导，
# 无需任何环境变量；若自身布局不成立（例如被单独拷走），再回落到 DSH_HOME。
# 这样"有 DSH 能跑，没有 DSH 也能跑"。
_THIS = os.path.dirname(os.path.abspath(__file__))
_SKILL = os.path.dirname(_THIS)
_SELF_SCRIPTS = os.path.join(_SKILL, "scripts")
if os.path.exists(os.path.join(_SELF_SCRIPTS, "vec.py")):
    SK = _SELF_SCRIPTS
else:
    SK = os.path.join(os.environ.get("DSH_HOME", ""),
                      "skills", "dsh-img-to-svg", "scripts")
PY = sys.executable

# ---------------------------------------------------------------- 全局状态

# ---- 页面关闭/心跳释放（v1.13 新增）----
# LAST_BEAT   最后一次收到 /beat 的 monotonic 时间（0=从未）
# EVER_BEAT   是否曾经收到过心跳（决定 idle 超时是否生效）
# EXIT_REQ    页面卸载信标：置 True 后主循环立即退出
_RL = threading.Lock()
LAST_BEAT = [0.0]
EVER_BEAT = [False]
EXIT_REQ = [False]
VEC_PROC = [None]     # 当前正在跑的 vec.py 子进程（供看门狗终止）
KEEPALIVE = [False]   # --keepalive 时忽略 /bye（用户要求常驻）
# 最近一次任务的实际交付物 (路径, 描述)。由 run_pipeline / run_tier2 /
# run_tier3 各自回填，供 main() 打印 —— FINAL_SAVED 是它们的局部变量，
# main() 看不到，直接引用会 NameError。
FINAL_INFO = [None, ""]
# 本次运行是否**真的把目标文件写成功**。
# 不能用 os.path.exists(out) 判断：目录里可能残留上一次的旧文件，
# 本次写盘失败（如 PermissionError）时那个旧文件仍在，
# 只看存在性会把失败报成成功（实测用独占句柄锁住输出文件时就是这样）。
OUT_WRITTEN = [False]
# 多标签页 / 刷新支持：
#   SESSIONS   活跃 session id -> 最后心跳时间
#   BYE_AT     收到 /bye 的时刻（用于刷新宽限）
#   REFRESH_GRACE  收到 /bye 后再等多久，若期间有新心跳（=刷新）则放弃退出
SESSIONS = {}
BYE_AT = [0.0]
REFRESH_GRACE = 3.0


STATE = {
    "phase": "init",
    "phase_text": "等待开始",
    "done": False,
    "error": None,
    "frames": [],          # [{id, kind, title, desc, meta, png_b64|svg}]
    "logs": [],            # [{t, level, text}]
    "params": {},
    "elapsed": 0.0,
    "started_at": time.time(),
}
LOCK = threading.Lock()


def log(text, level="info"):
    with LOCK:
        STATE["logs"].append({
            "t": round(time.time() - STATE["started_at"], 2),
            "level": level, "text": text,
        })
    print(f"[{time.time() - STATE['started_at']:6.1f}s] {text}", flush=True)


def set_phase(key, text):
    with LOCK:
        STATE["phase"] = key
        STATE["phase_text"] = text
    log(f"PHASE {text}", "phase")


def add_frame(kind, title, desc, meta=None, png=None, svg=None):
    """加一帧。png 为 PIL Image，会编码成 base64 data URL"""
    import base64
    import io as _io
    fr = {"id": len(STATE["frames"]), "kind": kind, "title": title,
          "desc": desc, "meta": meta or {},
          "t": round(time.time() - STATE["started_at"], 2)}
    if png is not None:
        # 缩略到"显示友好"尺寸：
        # 长边 <= 1280 且总像素 <= 1.2MP。
        # 竖图若只按长边缩，1600x2225 仍会在 16:9 工作区里溢出，
        # 所以同时限制总像素。
        im = png
        MAX_SIDE = 1280
        MAX_PIX = 1_200_000
        w0, h0 = im.size
        sc = 1.0
        if max(w0, h0) > MAX_SIDE:
            sc = min(sc, MAX_SIDE / max(w0, h0))
        if w0 * h0 * sc * sc > MAX_PIX:
            sc = min(sc, (MAX_PIX / (w0 * h0)) ** 0.5)
        if sc < 1.0:
            im = im.resize((max(1, int(w0 * sc)), max(1, int(h0 * sc))),
                           Image.LANCZOS)
        buf = _io.BytesIO()
        im.convert("RGB").save(buf, format="PNG", optimize=True)
        # 不入 frames（否则每次轮询都重传 MB 级 base64），
        # 改为登记到图片表，前端用 /frame/<id>.png 按需取 + 强缓存
        with LOCK:
            STATE.setdefault("imgs", {})[fr["id"]] = (
                "image/png", buf.getvalue())
        fr["img"] = fr["id"]
    if svg is not None:
        # SVG 帧同样栅格化成 PNG 作为显示图，两者都登记到图片表
        if png is None:
            ras = _rasterize_svg(svg, max_side=1280)
            if ras is not None:
                buf2 = _io.BytesIO()
                ras.convert("RGB").save(buf2, format="PNG", optimize=True)
                with LOCK:
                    STATE.setdefault("imgs", {})[fr["id"]] = (
                        "image/png", buf2.getvalue())
                fr["img"] = fr["id"]
    with LOCK:
        STATE["frames"].append(fr)
    return fr["id"]


_RASTER_ERR = []


def _rasterize_svg(svg_text, max_side=1280):
    """把 SVG 字符串栅格化成 PIL Image（长边 <= max_side）

    目的：SVG 帧直接塞给浏览器体积过大（实测末层约 10MB，
    56 帧累积超 200MB）。栅格化后每帧仅数百 KB，且与 PNG 帧
    走同一渲染路径，object-fit:contain 必然生效，不会出界。
    失败则返回 None（调用方退回 SVG 原样）。
    """
    try:
        import cairosvg
    except Exception:
        return None
    try:
        m = __import__("re").search(
            r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg_text[:400])
        if m:
            vw, vh = float(m.group(1)), float(m.group(2))
            sc = min(1.0, max_side / max(vw, vh, 1))
            ow = max(1, int(vw * sc))
        else:
            ow = max_side
        png = cairosvg.svg2png(bytestring=svg_text.encode("utf-8"),
                               output_width=ow)
        import io as _io2
        return Image.open(_io2.BytesIO(png)).convert("RGB")
    except Exception as _e:
        _RASTER_ERR.append(str(_e))
        return None


def svg_wrap(inner, w, h, base_hex):
    """包一层 SVG。

    关键：`preserveAspectRatio="xMidYMid meet"` 让浏览器**等比缩放**到容器内，
    不因 width/height="100%" 而按容器高度撑开导致横向溢出。
    外层再配合 CSS 的 max-width/max-height 与 overflow:hidden，
    无论工作分辨率多大（本插件可能到 3000x4206 甚至更大）都不会出界。
    """
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {w} {h}" '
        f'preserveAspectRatio="xMidYMid meet" '
        f'width="100%" height="100%" '
        f'style="max-width:100%;max-height:100%;display:block">'
        f'<rect width="{w}" height="{h}" fill="{base_hex}"/>'
        f'{inner}</svg>'
    )


# ---------------------------------------------------------------- 管线复现

def hex2(rgb):
    r, g, b = [int(max(0, min(255, round(v)))) for v in rgb]
    return f"#{r:02X}{g:02X}{b:02X}"


def _rdp(pts, eps):
    """纯 numpy Ramer-Douglas-Peucker"""
    if len(pts) < 3:
        return pts
    keep = np.zeros(len(pts), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        p0, p1 = pts[a], pts[b]
        seg = p1 - p0
        L = float(np.hypot(*seg))
        if L < 1e-9:
            d = np.hypot(*(pts[a + 1:b] - p0).T)
        else:
            n = np.array([-seg[1], seg[0]]) / L
            d = np.abs((pts[a + 1:b] - p0) @ n)
        k = int(np.argmax(d))
        if d[k] > eps:
            keep[a + 1 + k] = True
            stack.append((a, a + 1 + k))
            stack.append((a + 1 + k, b))
    return pts[keep]


def _trace_contour(border, start):
    """从 start 出发沿边界走一圈（8 邻域，Moore 跟踪简化版）"""
    H, W = border.shape
    # 8 邻域顺序（顺时针）
    NB = [(-1, 0), (-1, 1), (0, 1), (1, 1),
          (1, 0), (1, -1), (0, -1), (-1, -1)]
    ys, xs = np.nonzero(border)
    if len(ys) == 0:
        return None
    sx, sy = int(xs[0]), int(ys[0]) if start is None else start
    y, x = sy, sx
    poly = [(x, y)]
    prev = -1
    for _ in range(4 * (len(ys) + 4)):
        moved = False
        for k in range(8):
            idx = (prev + k + 1) % 8 if prev >= 0 else k
            dy, dx = NB[idx]
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and border[ny, nx]:
                if nx == poly[0][0] and ny == poly[0][1]:
                    return np.array(poly, dtype=np.float64)
                y, x = ny, nx
                poly.append((x, y))
                prev = idx
                moved = True
                break
        if not moved:
            break
    return np.array(poly, dtype=np.float64) if len(poly) >= 4 else None


def get_polys(mask, minarea, eps):
    """用 skimage.measure.find_contours 取轮廓（与 vec.py 同款）

    角度排序会破坏凹形状（实测出尖刺），这里改用 marching squares。
    """
    from skimage import measure
    from scipy.ndimage import gaussian_filter

    npx = int(mask.sum())
    if npx < 2:
        return []
    mb = gaussian_filter(mask.astype(np.float32), sigma=0.4)
    polys = []
    for c in measure.find_contours(mb, level=0.5):
        if len(c) < 4:
            continue
        if np.allclose(c[0], c[-1]):
            c = c[:-1]
        if len(c) < 3:
            continue
        sp = _rdp(c, eps)
        if len(sp) < 3:
            continue
        # (row, col) -> (x, y)
        sp = sp[:, ::-1]
        x, y = sp[:, 0], sp[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        if area < minarea:
            continue
        polys.append(sp)
    return polys


def poly_d(poly):
    if len(poly) == 0:
        return ""
    pts = "L".join(f"{float(x):.1f} {float(y):.1f}" for x, y in poly)
    return f"M{pts}Z"


def _render_svg(svg, png):
    """把 SVG 栅格化为 PNG（逐层帧与分块对比都要用）。

    **必须走 _run_child 登记**：单次 render 看着只要几十到几百毫秒，
    但它紧跟在 vec.py 之后调用。若 watch_draw 恰在此刻被看门狗杀掉，
    这个 render_check 子进程就成了孤儿，会继续写出 .png ——
    实测 `_t2_base.png` 在 watch 退出后 2 秒才落盘。
    """
    _run_child([PY, os.path.join(SK, "render_check.py"), svg, png],
               None, SK)
    return os.path.exists(png)


def _load_rgb(path, size):
    im = Image.open(path).convert("RGB")
    if size and im.size != size:
        im = im.resize(size, Image.LANCZOS)
    return im


def _err_heat(ref, cur, scale=4.0):
    """误差热图（纯热图，保留给需要单独看误差的场景）"""
    e = np.abs(ref - cur).max(axis=2)
    h = np.clip(e * scale, 0, 255).astype(np.uint8)
    rgb = np.stack([h, (h * 0.35).astype(np.uint8),
                    np.full_like(h, 20)], axis=-1).astype(np.uint8)
    return Image.fromarray(rgb), e


def _heat_overlay(base_img, ref, cur, scale=4.0, alpha=0.62):
    """在 base_img 上叠加误差热力图（半透明），避免整图换脸

    ## 为什么要叠加而不是换成纯热图

    纯热图与前一帧几乎 100% 像素不同（实测均值变化 199.46），
    时间轴上播放时是"闪一下全屏换色"。
    叠加式只让误差区域泛红，底图保持可见，帧间变化大幅下降，
    同时"哪里差"的信息表达得更直观（能看出是图的哪个部位）。
    """
    e = np.abs(ref - cur).max(axis=2)
    h = np.clip(e * scale, 0, 255).astype(np.float32) / 255.0
    base = np.asarray(base_img.convert("RGB")).astype(np.float32)
    H, W = e.shape
    # 热图颜色：低误差=透明，高误差=橙红
    heat_rgb = np.stack([
        np.full((H, W), 245, np.float32),          # R
        (200 - 120 * h),                            # G 高误差偏红
        np.full((H, W), 40, np.float32),           # B
    ], axis=-1)
    # 叠加强度随误差强度上升
    a = np.clip(h * alpha, 0, alpha)[..., None]
    out = base * (1 - a) + heat_rgb * a
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)), e


def _draw_boxes(png, boxes, color=(245, 166, 35), label=True):
    """在图上框出块/区域。

    必须容忍**反向坐标**与**越界坐标**：regions 由用户手输，
    x1<x0 或 y1<y0 时 PIL 的 rectangle 会抛
    ValueError: x1 must be greater than or equal to x0 ——
    而这个调用点（run_tier3 的 837 行）**早于**下面的逐块循环，
    所以循环里那层越界过滤根本没机会生效（实测会崩且留下 _t3_base 残骸）。
    在通用绘图函数内部做规范化，所有调用方都受益。
    """
    im = png.convert("RGB").copy()
    dr = ImageDraw.Draw(im)
    W, H = im.size
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        # 端点写反就交换
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        # 夹进图像范围，且保证 x1>=x0 / y1>=y0
        x0 = max(0, min(W - 1, x0))
        y0 = max(0, min(H - 1, y0))
        x1 = max(x0, min(W - 1, x1))
        y1 = max(y0, min(H - 1, y1))
        dr.rectangle([x0, y0, x1, y1], outline=color, width=2)
        if label:
            dr.text((x0 + 3, max(0, y0 - 11)), "#%d" % (i + 1), fill=color)
    return im


def _run_child(cmd, env, cwd):
    """跑一个子进程，**登记到 VEC_PROC**，让看门狗在退出时能杀掉它。

    为什么不用 subprocess.run：run 无法中途终止。看门狗 os._exit 时，
    正在跑的 vec.py / max_quality / quality_boost 会变成孤儿，继续跑几十秒
    并写出没人认领的 `_t2_base.svg` / `_t3_base.svg`（实测 1.2~4MB）。

    返回值提供 .returncode / .stdout / .stderr 三个属性，与 run 的结果兼容。
    """
    class _Res:
        pass

    pr = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
        env=env, cwd=cwd)
    with _RL:
        VEC_PROC[0] = pr
    try:
        out, _ = pr.communicate()
    finally:
        with _RL:
            VEC_PROC[0] = None
    r = _Res()
    r.returncode = pr.returncode
    r.stdout = out or ""
    r.stderr = ""          # stderr 已并入 stdout
    return r


def _cleanup_mid(outdir, prefixes, subdirs, keep=()):
    """清理中间产物。清理失败绝不抛异常（不能因此让任务失败）。

    prefixes: 文件名前缀列表（如 ["_t2_base"]）
    subdirs:  子目录名列表（如 ["_t2_mq"]）
    keep:     要保留的文件。顶层文件写文件名；子目录内的写
              "子目录名/文件名"。**子目录里有保留项时只删其它文件，
              不删目录本身**（例如 tier2 的交付物 max_final.png
              就在 _t2_mq/ 里）。

    返回删除的文件数。
    """
    removed = 0
    keep = set(keep or ())
    try:
        if not outdir or not os.path.isdir(outdir):
            return 0
        for fn in os.listdir(outdir):
            fp = os.path.join(outdir, fn)
            if os.path.isdir(fp):
                if fn not in subdirs:
                    continue
                sub_keep = {k.split("/", 1)[1] for k in keep
                            if k.startswith(fn + "/")}
                # 兼容调用方用 os.sep 写 keep（Windows 下是反斜杠）
                if not sub_keep:
                    sub_keep = {k.split(os.sep, 1)[1] for k in keep
                                if k.startswith(fn + os.sep)}
                try:
                    for f2 in list(os.listdir(fp)):
                        if f2 in sub_keep:
                            continue
                        try:
                            os.remove(os.path.join(fp, f2))
                            removed += 1
                        except OSError:
                            pass
                    # 只有「目录已空」时才删目录。若还留着保留项
                    # （如 max_final.png），目录必须保留 —— 那段交付物
                    # 就在这里，删了用户就拿不到成品。
                    try:
                        if not os.listdir(fp):
                            os.rmdir(fp)
                    except OSError:
                        pass
                except OSError:
                    pass
                continue
            if fn in keep:
                continue
            if any(fn.startswith(p) for p in prefixes):
                try:
                    os.remove(fp)
                    removed += 1
                except OSError:
                    pass
    except Exception:
        pass
    return removed


def run_tier2(src, svg_out, colors, edge, w, blocks_n, block, up, nblk):
    FINAL_SAVED = None
    """高质量档：超采样 W + 误差驱动分块重绘（帧序列含逐块）"""
    t0 = time.time()
    im = Image.open(src)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im).convert("RGB")
    else:
        im = im.convert("RGB")
    W0, H0 = im.size
    with LOCK:
        STATE["params"].update({
            "src": os.path.basename(src), "src_size": f"{W0}x{H0}",
            "mp": round(W0 * H0 / 1e6, 2), "colors": colors,
            "edge": edge, "档位": "2 高质量",
            "超采样 W": w, "重绘块数": blocks_n,
            "块尺寸": block, "块内色数": nblk,
        })
    add_frame("source", "SOURCE / 00", "原始输入图像",
              {"尺寸": f"{W0}x{H0}"}, png=im)
    log(f"[高质量档] {os.path.basename(src)} {W0}x{H0}  W={w}")

    # --- 基础：vec.py 以 VEC_W=w 超采样 ---
    set_phase("resize", f"SUPERSAMPLE / 01 (W={w})")
    log(f"超采样矢量化 W={w} …（这一步是高质量档的核心）")
    base_svg = os.path.join(os.path.dirname(svg_out), "_t2_base.svg")
    base_png = base_svg.replace(".svg", ".png")
    env = {k: v for k, v in os.environ.items() if not k.startswith("VEC_")}
    env.update({"VEC_N": str(colors), "VEC_EDGE": str(edge),
                "VEC_W": str(w), "VEC_SEED": "42",
                "VEC_TITLE": "t2base"})
    r = _run_child([PY, os.path.join(SK, "vec.py"), src, base_svg],
                   env, SK)
    if r.returncode != 0:
        err = (r.stderr.strip().splitlines() or ["?"])[-1]
        log(f"基础单矢量化失败：{err}", "error")
        with LOCK:
            STATE["error"] = err
        return False
    _render_svg(base_svg, base_png)
    base_im = _load_rgb(base_png, (W0, H0))
    ref = np.asarray(im).astype(np.float32)
    base = np.asarray(base_im).astype(np.float32)
    mae0 = float(np.abs(ref - base).mean())
    # 工作分辨率（用于说明超采样倍数）
    # 工作分辨率 = vec.py 内部的 WORK_W（受 min(W, W0*2) 限制），
    # 不是 SVG 的 viewBox（那是原图尺寸）
    work_w = min(w, W0 * 2)
    work_h = round(work_w * H0 / W0)
    ww = f"{work_w}x{work_h}"
    add_frame("resize", f"SUPERSAMPLE / 01 (W={w})",
              "以 VEC_W 提高工作分辨率后矢量化。\n"
              "高质量档的第一半收益来自这里。",
              {"超采样 W": w, "工作分辨率": ww,
               "MAE": round(mae0, 3)}, png=base_im)
    log(f"超采样完成  MAE {mae0:.3f}")

    # --- 误差热图 + 选块 ---
    set_phase("quantize", "GAIN-MAP / 02")
    heat, e = _heat_overlay(base_im, ref, base)
    BL = block
    step = max(1, BL // 2)
    blist = []
    for by in range(0, max(1, H0 - BL), step):
        for bx in range(0, max(1, W0 - BL), step):
            mm = float(np.abs(ref[by:by + BL, bx:bx + BL]
                              - base[by:by + BL, bx:bx + BL]).mean())
            blist.append((mm, bx, by))
    blist.sort(reverse=True)
    pick = blist[:blocks_n]
    boxes = [(bx, by, min(bx + BL, W0), min(by + BL, H0))
             for _, bx, by in pick]
    add_frame("gainmap", "GAIN-MAP / 02",
              f"全图误差热图（越亮=误差越大），框出误差最高的 "
              f"{len(pick)} 个块。\n高质量档会逐块把这些位置重新矢量化。",
              {"候选块数": len(blist), "选中块数": len(pick),
               "块尺寸": f"{BL}x{BL}",
               "最高块MAE": round(pick[0][0], 2) if pick else 0},
              png=_draw_boxes(heat, boxes))
    log(f"选块完成：{len(pick)} 块，最高块 MAE {pick[0][0]:.2f}" if pick
        else "选块完成：0 块")

    # --- 逐块重绘（每块一帧）---
    set_phase("trace", "BLOCK-REDRAW / 03")
    canvas = base.copy()
    done = 0
    tmpdir = os.path.dirname(svg_out)
    for i, ((m0, bx, by), bb) in enumerate(zip(pick, boxes)):
        x0, y0 = max(0, bx - 16), max(0, by - 16)
        x1, y1 = min(W0, bx + BL + 16), min(H0, by + BL + 16)
        crop = im.crop((x0, y0, x1, y1))
        cw, ch = crop.size
        if cw < 8 or ch < 8:
            continue
        tp = os.path.join(tmpdir, "_t2b%02d.png" % i)
        crop.resize((cw * up, ch * up), Image.LANCZOS).save(tp)
        sv = os.path.join(tmpdir, "_t2b%02d.svg" % i)
        env2 = {k: v for k, v in os.environ.items()
                if not k.startswith("VEC_")}
        env2.update({"VEC_N": str(nblk), "VEC_MERGE_STEP": "0",
                     "VEC_SEED": "42", "VEC_TITLE": "blk%d" % i})
        r2 = _run_child([PY, os.path.join(SK, "vec.py"), tp, sv],
                        env2, SK)
        if r2.returncode != 0:
            continue
        pg = sv.replace(".svg", ".png")
        if not _render_svg(sv, pg):
            continue
        loc = np.asarray(_load_rgb(pg, (cw, ch))).astype(np.float32)
        bh, bw = min(BL, H0 - by), min(BL, W0 - bx)
        canvas[by:by + bh, bx:bx + bw] = loc[
            by - y0:by - y0 + bh, bx - x0:bx - x0 + bw]
        done += 1
        cur = float(np.abs(ref - canvas).mean())
        # 帧：显示"已贴回的块"用亮框，当前块用另一种标记
        vis = _draw_boxes(Image.fromarray(
            np.clip(canvas, 0, 255).astype(np.uint8)), boxes[:done],
            color=(74, 222, 128), label=False)
        vis = _draw_boxes(vis, [bb], color=(245, 166, 35))
        add_frame("block", f"BLOCK-REDRAW / {done}/{len(pick)}",
                  f"第 {done} 块重绘完成并贴回。\n"
                  f"该块内 MAE {m0:.2f} → "
                  f"{float(np.abs(ref[by:by + bh, bx:bx + bw] - canvas[by:by + bh, bx:bx + bw]).mean()):.2f}",
                  {"块号": f"{done}/{len(pick)}",
                   "位置": f"({bx},{by})",
                   "块内MAE前": round(m0, 2),
                   "全图MAE": round(cur, 3),
                   "相对基础": f"{100 * (cur - mae0) / mae0:+.2f}%"},
                  png=vis)
        with LOCK:
            STATE["layer_done"] = done
            STATE["layer_total"] = len(pick)
        for pth in (tp, sv, pg):
            try:
                if os.path.exists(pth):
                    os.remove(pth)
            except Exception:
                pass
        if done % 5 == 0:
            log(f"  块 {done}/{len(pick)}  全图 MAE {cur:.3f}")

    # --- 贴回完成 ---
    mae1 = float(np.abs(ref - canvas).mean())
    add_frame("patched", "PATCHED / 04",
              "所有选中块贴回后的合成结果（尚未出 SVG）。",
              {"完成块数": done, "MAE": round(mae1, 3),
               "相对基础": f"{100 * (mae1 - mae0) / mae0:+.2f}%"},
              png=_draw_boxes(Image.fromarray(
                  np.clip(canvas, 0, 255).astype(np.uint8)),
                  boxes[:done], color=(74, 222, 128), label=False))
    log(f"贴回完成 MAE {mae0:.3f} -> {mae1:.3f}")

    # --- FINAL：调用真正的 max_quality.py ---
    set_phase("final", "FINAL / 99")
    log("调用权威 max_quality.py 出具最终结果…")
    mqdir = os.path.join(os.path.dirname(svg_out), "_t2_mq")
    os.makedirs(mqdir, exist_ok=True)
    # encoding 必须显式 utf-8 且给子进程注入 PYTHONIOENCODING：
    # 子进程（max_quality.py）在中文 Windows 上默认按 cp936 输出，
    # 父进程按 UTF-8 解码就会乱码，乱码里还藏着真实错误信息。
    _env_mq = {k2: v2 for k2, v2 in os.environ.items()
               if not k2.startswith("VEC_")}
    _env_mq["PYTHONIOENCODING"] = "utf-8"
    # 用 Popen 而非 run：run 无法在运行中途终止，看门狗退出时
    # max_quality 会变成孤儿继续跑完并写盘。登记到 VEC_PROC 让看门狗能杀。
    _mq = subprocess.Popen(
        [PY, os.path.join(SK, "..", "tools", "max_quality.py"),
         src, mqdir, "--w", str(w),
         "--blocks", str(blocks_n), "--block", str(block),
         "--up", str(up), "--n", str(nblk)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
        env=_env_mq, cwd=os.path.dirname(SK))
    with _RL:
        VEC_PROC[0] = _mq
    try:
        _mq_out, _ = _mq.communicate()
    finally:
        with _RL:
            VEC_PROC[0] = None

    class _R3:
        pass

    r3 = _R3()
    r3.returncode = _mq.returncode
    r3.stdout = _mq_out or ""
    r3.stderr = ""      # stderr 已并入 stdout
    fin_png = os.path.join(mqdir, "max_final.png")
    if os.path.exists(fin_png):
        fi = _load_rgb(fin_png, (W0, H0))
        fa = np.asarray(fi).astype(np.float32)
        fm = float(np.abs(ref - fa).mean())
        add_frame("final", "FINAL / 99",
                  "权威 max_quality.py 产出（与正式高质量档交付一致）。",
                  {"MAE": round(fm, 3), "超采样 W": w,
                   "块数": done}, png=fi)
        with LOCK:
            STATE["params"].update({"mae": round(fm, 3)})
        log(f"最终 MAE {fm:.3f}")
        FINAL_SAVED = (fin_png, "max_quality 合成图")
        FINAL_INFO[0], FINAL_INFO[1] = FINAL_SAVED
        OUT_WRITTEN[0] = True
    else:
        FINAL_SAVED = None
        # 诊断信息非常重要：只报「未生成」会让用户和后续调试无从下手。
        _err2 = (r3.stderr.strip().splitlines() or [""])[-1] if r3.stderr else ""
        _tail2 = (r3.stdout.strip().splitlines() or [""])[-1] if r3.stdout else ""
        log(f"max_quality.py 未生成 max_final.png"
            f"（returncode={r3.returncode}）", "error")
        if _err2:
            log(f"  stderr: {_err2}", "error")
        if _tail2:
            log(f"  stdout 末行: {_tail2}", "error")
        with LOCK:
            STATE["error"] = _err2 or "max_quality 未产出"

    # 清理 tier2 中间产物。放在 if/else 之外：max_quality 失败时同样要清，
    # 否则会留下 4MB 级的 _t2_base.svg。_t2_mq/ 里的交付物按名保留。
    _n2 = _cleanup_mid(
        os.path.dirname(os.path.abspath(svg_out)),
        ["_t2_base"],
        ["_t2_mq"],
        keep=["_t2_mq/max_final.png",
              "_t2_mq/max_result.json"])
    if _n2:
        log(f"  已清理 {_n2} 个中间产物")

    with LOCK:
        STATE["elapsed"] = time.time() - t0
        STATE["done"] = True
        STATE["phase"] = "done"
        STATE["phase_text"] = "COMPLETE"
    log(f"[高质量档] 全部完成，总耗时 {time.time() - t0:.1f}s")
    # 最后一条必须是保存路径
    if FINAL_SAVED and os.path.exists(FINAL_SAVED[0]):
        log(f"SAVED  {os.path.abspath(FINAL_SAVED[0])}  "
            f"({os.path.getsize(FINAL_SAVED[0])/1048576:.1f} MB)", "path")
    return True


def run_tier3(src, svg_out, colors, edge, blocks_n, block, up, nblk,
              regions):
    FINAL_SAVED = None
    """档位3：局部重绘（独立帧序列，非全图流程）"""
    t0 = time.time()
    im = Image.open(src)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im).convert("RGB")
    else:
        im = im.convert("RGB")
    W0, H0 = im.size
    ref = np.asarray(im).astype(np.float32)
    tmpdir = os.path.dirname(svg_out)

    mode = "语义区域" if regions else f"自动选块 top{blocks_n}"
    with LOCK:
        STATE["params"].update({
            "src": os.path.basename(src), "src_size": f"{W0}x{H0}",
            "mp": round(W0 * H0 / 1e6, 2), "档位": "3 局部重绘",
            "模式": mode, "块尺寸": block, "块内色数": nblk,
            "块内放大": f"{up}x",
        })
    log(f"[档位3] {os.path.basename(src)} {W0}x{H0}  {mode}")

    # 01 原图
    add_frame("source", "SOURCE / 00", "原始输入图像",
              {"尺寸": f"{W0}x{H0}"}, png=im)

    # 02 基础矢量化
    set_phase("base", "BASE / 01")
    log("基础矢量化（vec.py 默认参数）…")
    base_svg = os.path.join(tmpdir, "_t3_base.svg")
    base_png = base_svg.replace(".svg", ".png")
    env = {k: v for k, v in os.environ.items() if not k.startswith("VEC_")}
    env.update({"VEC_N": str(colors), "VEC_EDGE": str(edge),
                "VEC_SEED": "42", "VEC_TITLE": "t3base"})
    r = _run_child([PY, os.path.join(SK, "vec.py"), src, base_svg],
                   env, SK)
    if r.returncode != 0:
        err = (r.stderr.strip().splitlines() or ["?"])[-1]
        log(f"基础矢量化失败：{err}", "error")
        with LOCK:
            STATE["error"] = err
        return False
    _render_svg(base_svg, base_png)
    base_im = _load_rgb(base_png, (W0, H0))
    base = np.asarray(base_im).astype(np.float32)
    mae0 = float(np.abs(ref - base).mean())
    add_frame("base", "BASE / 01",
              "基础矢量化。档位3 的出发点是这张基础图——"
              "它已经定形，接下来只在选中的局部重画。\n"
              "**注意：未被选中的区域不会改变**。",
              {"基础 MAE": round(mae0, 3)}, png=base_im)
    log(f"基础 MAE {mae0:.3f}")

    # 03 误差叠加图（半透明，不换脸）
    set_phase("gainmap", "GAIN-MAP / 02")
    heat, e = _heat_overlay(base_im, ref, base)
    log("误差叠加图已生成（半透明，底图为矢量结果）")

    # 04 选定目标（与热图合并成一帧）
    boxes = []
    if regions:
        for seg in regions.split(";"):
            seg = seg.strip()
            if not seg:
                continue
            p = [int(v) for v in seg.split(",")]
            if len(p) == 4:
                boxes.append((p[0], p[1], p[2], p[3]))
    else:
        BL = block
        step = max(1, BL // 2)
        blist = []
        for by in range(0, max(1, H0 - BL), step):
            for bx in range(0, max(1, W0 - BL), step):
                mm = float(np.abs(ref[by:by + BL, bx:bx + BL]
                                  - base[by:by + BL, bx:bx + BL]).mean())
                blist.append((mm, bx, by))
        blist.sort(reverse=True)
        boxes = [(bx, by, min(bx + BL, W0), min(by + BL, H0))
                 for _, bx, by in blist[:blocks_n]]

    # 热图 + 选框合并为一帧（避免两次整图切换）
    heat_boxes = _draw_boxes(heat, boxes)
    add_frame("gainmap", "GAIN-MAP+TARGETS / 02",
              f"误差叠加图 + 选定 {len(boxes)} 个重绘目标"
              f"（{'手工语义区域' if regions else '自动按误差'}）。\n"
              "红热区域=误差大，橙框=将要重绘的目标；"
              "每块会单独裁剪、放大后重新矢量化再贴回。",
              {"全图MAE": round(mae0, 3),
               "误差p99": round(float(np.percentile(e, 99)), 1),
               "目标数": len(boxes), "模式": mode},
              png=heat_boxes)
    with LOCK:
        STATE["layer_total"] = len(boxes)
        STATE["layer_done"] = 0
    log(f"选定 {len(boxes)} 个目标")

    # 05+ 逐块重绘（每块一帧）
    set_phase("block", "REDRAW / 03")
    canvas = base.copy()
    done = 0
    for i, bb in enumerate(boxes):
        x0, y0, x1, y1 = bb
        # 越界保护：regions 由用户手输，可能给出超出图像范围的坐标。
        # 若不做这层过滤，px1 会被 min() 压到比 px0 还小，
        # im.crop((px0, py0, px1, py1)) 直接抛
        # ValueError: Coordinate 'right' is less than 'left'，
        # 而异常走 work() 的 except，**不经过 run_tier3 末尾的
        # _cleanup_mid**，于是留下 _t3_base.png/svg 残骸。
        if x0 >= W0 or y0 >= H0 or x1 <= 0 or y1 <= 0:
            log(f"  跳过越界区域 {bb}（图像 {W0}x{H0}）", "warn")
            continue
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(W0 - 1, x1)
        y1 = min(H0 - 1, y1)
        if x1 < x0 or y1 < y0:
            log(f"  跳过无效区域 {bb}", "warn")
            continue
        px0, py0 = max(0, x0 - 16), max(0, y0 - 16)
        px1, py1 = min(W0, x1 + 16), min(H0, y1 + 16)
        # 双保险：即使上面算完，也不允许 px1 < px0 传给 crop
        px1 = max(px0 + 1, px1)
        py1 = max(py0 + 1, py1)
        crop = im.crop((px0, py0, px1, py1))
        cw, ch = crop.size
        if cw < 8 or ch < 8:
            continue
        core_before = float(np.abs(ref[y0:y1 + 1, x0:x1 + 1]
                                   - canvas[y0:y1 + 1, x0:x1 + 1]).mean())
        tp = os.path.join(tmpdir, "_t3b%02d.png" % i)
        crop.resize((cw * up, ch * up), Image.LANCZOS).save(tp)
        sv = os.path.join(tmpdir, "_t3b%02d.svg" % i)
        env2 = {k: v for k, v in os.environ.items()
                if not k.startswith("VEC_")}
        env2.update({"VEC_N": str(nblk), "VEC_MERGE_STEP": "0",
                     "VEC_SEED": "42", "VEC_TITLE": "t3blk%d" % i})
        r2 = _run_child([PY, os.path.join(SK, "vec.py"), tp, sv],
                        env2, SK)
        if r2.returncode != 0:
            continue
        pg = sv.replace(".svg", ".png")
        if not _render_svg(sv, pg):
            continue
        loc = np.asarray(_load_rgb(pg, (cw, ch))).astype(np.float32)
        canvas[y0:y1 + 1, x0:x1 + 1] = loc[py0 - py0 + (y0 - py0):
                                           y0 - py0 + (y1 - y0 + 1),
                                           (x0 - px0):(x0 - px0) + (x1 - x0 + 1)]
        done += 1
        core_after = float(np.abs(ref[y0:y1 + 1, x0:x1 + 1]
                                  - canvas[y0:y1 + 1, x0:x1 + 1]).mean())
        vis = _draw_boxes(Image.fromarray(
            np.clip(canvas, 0, 255).astype(np.uint8)), boxes[:done],
            color=(74, 222, 128), label=False)
        vis = _draw_boxes(vis, [bb], color=(245, 166, 35))
        add_frame("block", f"REDRAW / {done}/{len(boxes)}",
                  f"第 {done} 个目标重绘完成。\n"
                  f"该区域核心 MAE {core_before:.2f} → {core_after:.2f}"
                  + ("（改善 %.1f%%）" % (100 * (core_after - core_before) / core_before)
                     if core_before > 0 else ""),
                  {"目标": f"{done}/{len(boxes)}",
                   "区域": f"{x0},{y0},{x1},{y1}",
                   "核心MAE前": round(core_before, 2),
                   "核心MAE后": round(core_after, 2),
                   "全图MAE": round(float(np.abs(ref - canvas).mean()), 3)},
                  png=vis)
        with LOCK:
            STATE["layer_done"] = done
        for pth in (tp, sv, pg):
            try:
                if os.path.exists(pth):
                    os.remove(pth)
            except Exception:
                pass
        log(f"  目标 {done}/{len(boxes)}  核心 MAE "
            f"{core_before:.2f}→{core_after:.2f}")

    # 06 贴回完成
    mae1 = float(np.abs(ref - canvas).mean())
    add_frame("patched", "PATCHED / 04",
              "全部目标贴回后的合成图。\n"
              "**未选中区域完全没变** —— 这是档位3 的本质特征。",
              {"完成": done, "全图MAE": round(mae1, 3),
               "相对基础": f"{100 * (mae1 - mae0) / mae0:+.2f}%"},
              png=_draw_boxes(Image.fromarray(
                  np.clip(canvas, 0, 255).astype(np.uint8)),
                  boxes[:done], color=(74, 222, 128), label=False))
    log(f"贴回完成 MAE {mae0:.3f} -> {mae1:.3f}")

    # 99 FINAL：真正的 quality_boost.py
    set_phase("final", "FINAL / 99")
    log("调用权威 quality_boost.py 出具最终 SVG…")
    qb_out = svg_out
    qcmd = [PY, os.path.join(SK, "..", "tools", "quality_boost.py"),
            src, qb_out]
    if regions:
        qcmd += ["--regions", regions]
    else:
        qcmd += ["--auto", str(blocks_n)]
    qcmd += ["--block", str(block), "--up", str(up), "--n", str(nblk)]
    # 同样用 Popen + 登记：quality_boost 要跑几分钟，看门狗退出时必须能杀。
    _qb = subprocess.Popen(
        qcmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
        cwd=os.path.dirname(SK))
    with _RL:
        VEC_PROC[0] = _qb
    try:
        _qb_out, _ = _qb.communicate()
    finally:
        with _RL:
            VEC_PROC[0] = None

    class _R4:
        pass

    r4 = _R4()
    r4.returncode = _qb.returncode
    r4.stdout = _qb_out or ""
    r4.stderr = ""
    # 真实约定：quality_boost.py <out.svg> 会写出 <out>_render.png
    # 与 <out>_result.json；它**不生成可编辑 SVG**
    stem = os.path.splitext(qb_out)[0]
    fin_png = stem + "_render.png"
    res_json = stem + "_result.json"
    qb_meta = {}
    if os.path.exists(res_json):
        try:
            qb_meta = json.load(open(res_json, encoding="utf-8"))
        except Exception:
            qb_meta = {}

    if os.path.exists(fin_png):
        fi = _load_rgb(fin_png, (W0, H0))
        fa = np.asarray(fi).astype(np.float32)
        fm = float(np.abs(ref - fa).mean())
        meta = {"MAE": round(fm, 3), "目标数": done}
        if qb_meta:
            meta["官方记录 MAE"] = round(qb_meta.get("mae_after", fm), 3)
            meta["基础 MAE"] = round(qb_meta.get("mae_base", mae0), 3)
            meta["成本倍数"] = f"{qb_meta.get('ratio', 0):.1f}x"
        add_frame("final", "FINAL / 99",
                  "权威 quality_boost.py 的像素域最终结果。\n"
                  "**注意：该工具只产出像素域合成图与指标，不产出可编辑 SVG**"
                  "（脚本自带说明：要可编辑 SVG 需用 proto_semantic_overlay.py）。\n"
                  "此处 FINAL 用于确认「局部重绘后的真实效果」，"
                  "与上面 PATCHED 帧应当接近。",
                  meta, png=fi)
        with LOCK:
            STATE["params"].update({
                "mae": round(fm, 3),
                "产出说明": "档位3 只给像素域结果，无可编辑SVG",
            })
        log(f"最终（像素域）MAE {fm:.3f}")
        FINAL_SAVED = (fin_png, "quality_boost 像素域结果")
        FINAL_INFO[0], FINAL_INFO[1] = FINAL_SAVED
        OUT_WRITTEN[0] = True
        if qb_meta.get("mae_after"):
            log(f"  官方记录 MAE {qb_meta['mae_after']:.3f}，"
                f"基础 {qb_meta.get('mae_base', mae0):.3f}")
    else:
        err = (r4.stderr.strip().splitlines() or [""])[-1] if r4.stderr else ""
        err = err or ("未找到 %s" % os.path.basename(fin_png))
        log(f"quality_boost.py 未产出：{err}", "error")
        with LOCK:
            STATE["error"] = err

    # 清理 tier3 中间产物（_t3_base / _qbase / _lr_b 等）。
    # 放在 if/else 之外：无论 quality_boost 成功还是失败都要清，
    # 否则失败时会留一堆 1.5MB 级中间图。交付物（_render.png 与
    # _result.json）按文件名保留。
    _n3 = _cleanup_mid(
        os.path.dirname(os.path.abspath(svg_out)),
        ["_t3_base", "_qbase", "_lr_b", "_lr_r", "_lhb"],
        [],
        keep=[os.path.basename(fin_png), os.path.basename(res_json)])
    if _n3:
        log(f"  已清理 {_n3} 个中间产物")

    with LOCK:
        STATE["elapsed"] = time.time() - t0
        STATE["done"] = True
        STATE["phase"] = "done"
        STATE["phase_text"] = "COMPLETE"
    log(f"[档位3] 全部完成，总耗时 {time.time() - t0:.1f}s")
    if FINAL_SAVED and os.path.exists(FINAL_SAVED[0]):
        log(f"SAVED  {os.path.abspath(FINAL_SAVED[0])}  "
            f"({os.path.getsize(FINAL_SAVED[0])/1048576:.1f} MB)", "path")
    return True


def _k_advice(W0, H0, k):
    """给出 k 的尺度合理性提示（档位4 由调用方/模型定 k 时用）。

    ## 为什么不预报精确耗时

    实测数据表明耗时**不只取决于像素数**：
    2560x1440 的图 k=1 用 582.7s，而 k=2 只用 277s ——
    因为大图在 k=1 时层数/路径数极多，反量化开销反而更大。
    所以这里只报**算术确定值**（分辨率、像素数、倍数）+ 一个明确标注为粗估的档位。

    ## 两条硬事实（已实测）

    1. vec.py 的 `AUTOW` 默认**关闭**（`VEC_AUTOW` 缺省为 "0"），
       所以 k 放大后的尺寸不会被悄悄压回 1920 —— k 的增益是真实的。
       （负对照：显式 `VEC_AUTOW=1` 时 2400 宽会被压到 1920。）
    2. 工作像素超过约 1 亿时，单张图会跑到十几分钟量级。
    """
    if k <= 1:
        return
    _k_advice_print(W0, H0, k)
    log(f"    注：AUTOW 默认关闭（VEC_AUTOW 缺省 \"0\"），"
        f"放大后的尺寸不会被压回 1920，k 增益真实生效", "info")


def _k_tier(mp):
    """按工作像素数给出规模档位描述（纯函数，供提示与 --plan 复用）。"""
    if mp < 8:
        return "轻", "通常 1-3 分钟", ""
    if mp < 25:
        return "中", "通常 3-8 分钟", ""
    if mp < 60:
        return "重", "5-15 分钟量级", "已进入长任务区间，注意 idle-exit 与心跳宽限"
    if mp < 120:
        return "很重", "10-30 分钟量级", "长时间占用；agent 调用要确认不会触发空闲退出"
    return "极重", "可能 30 分钟以上", "超出单张常规上限，强烈建议降低 k"


def _k_advice_print(W0, H0, k):
    """打印单档 k 的尺度提示（供运行中与 --plan 复用）。"""
    wp, hp = W0 * k, H0 * k
    mp = wp * hp / 1e6
    base_mp = W0 * H0 / 1e6
    ratio = mp / max(base_mp, 1e-9)
    name, dur, warn = _k_tier(mp)
    log(f"  k={k} 尺度提示：工作分辨率 {wp}x{hp}（{mp:.1f} MP，"
        f"是 k=1 的 {ratio:.1f} 倍）", "warn" if warn else "info")
    log(f"    规模档位：{name}（{dur}）"
        + (f"（{warn}）" if warn else ""), "warn" if warn else "info")
    log(f"    参考：实测 1066x600 在 k=4（10.2 MP）约 96 秒；"
        f"1280x720 在 k=4（14.8 MP）约 199 秒；"
        f"2560x1440 在 k=2（14.8 MP）约 277 秒", "info")


def print_k_plan(src):
    """--plan：只读图 + 打一张完整的 k 档位表，**不启动服务、不跑管线**。

    用途：档位4 由调用方（模型/用户）决定 k。这时应当**先看表再定 k**，
    而不是等跑起来才看到提示（那时 k 已经定了，来不及改）。
    哪怕定错也要花几秒，比跑十分钟才发现要好。
    """
    im = Image.open(src)
    W0, H0 = im.size
    print(f"\n  源图：{os.path.basename(src)}  {W0}x{H0}"
          f"（{W0 * H0 / 1e6:.2f} MP）", flush=True)
    print(f"  档位4 由你（调用方）决定 k —— 下面是各档的**预演**，"
          f"看完再定，不用试错。\n", flush=True)

    hdr = (f"  {'k':>2}  {'工作分辨率':>15}  {'工作像素':>9}  "
           f"{'相对k=1':>8}  {'规模':<4}  说明")
    print(hdr, flush=True)
    print("  " + "-" * (len(hdr) - 2), flush=True)

    for k in range(1, 9):
        wp, hp = W0 * k, H0 * k
        mp = wp * hp / 1e6
        name, dur, warn = _k_tier(mp)
        ratio = f"{k * k}x"
        print(f"  {k:>2}  {wp:>7}x{hp:<7}  {mp:>7.1f}MP  "
              f"{ratio:>8}  {name:<4}  {dur}"
              + (f"；{warn}" if warn else ""), flush=True)

    print(f"\n  提示：k=1 是快速档（最快）；k>1 才进入上采样，增益真实"
          f"（AUTOW 默认关闭，不会被压回 1920）。", flush=True)
    print(f"  建议：先按「规模=轻/中」选一个能接受的档，跑完看 MAE/grad_keep 再决定要不要加 k。",
          flush=True)
    print(f"\n  定了之后这样跑：\n"
          f"    tools/watch_draw.py \"{src}\" --tier 4 --k <你选的数>\n", flush=True)
    return 0


def run_pipeline(src, svg_out, colors, edge, target_k):
    FINAL_SAVED = None
    """复现可视化中间态 + 调 vec.py 出权威结果"""
    t0 = time.time()
    im = Image.open(src)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im).convert("RGB")
    else:
        im = im.convert("RGB")
    W0, H0 = im.size

    with LOCK:
        STATE["params"].update({
            "src": os.path.basename(src), "src_size": f"{W0}x{H0}",
            "mp": round(W0 * H0 / 1e6, 2), "colors": colors,
            "edge": edge, "k": target_k,
        })

    add_frame("source", "SOURCE / 00", "原始输入图像",
              {"尺寸": f"{W0}x{H0}", "MP": round(W0 * H0 / 1e6, 2)}, png=im)
    log(f"读取 {os.path.basename(src)}  {W0}x{H0}")

    # ---- 决定工作分辨率（与档位 4 一致）----
    set_phase("resize", "RESIZE / 01")
    _k_advice(W0, H0, target_k)      # 档位4 由调用方定 k，这里给尺度提示
    if target_k > 1:
        WW, WH = W0 * target_k, H0 * target_k
    else:
        WW, WH = W0, H0
    ws = im.resize((WW, WH), Image.LANCZOS)
    add_frame("resize", "RESIZE / 01",
              f"放大至工作分辨率（k={target_k}）",
              {"工作尺寸": f"{WW}x{WH}", "k": target_k}, png=ws)
    log(f"工作分辨率 {WW}x{WH}（k={target_k}）")

    # ---- 平滑 ----
    set_phase("smooth", "SMOOTH / 02")
    med = ws.filter(ImageFilter.MedianFilter(3)).filter(
        ImageFilter.MedianFilter(3))
    sm = np.asarray(med).astype(np.float32)
    orig = np.asarray(ws).astype(np.float32)
    add_frame("smooth", "SMOOTH / 02",
              "两遍 3x3 中值滤波：压掉纹理噪声，代价是 1~2px 细线会变淡",
              {"算法": "MedianFilter(3) x2"}, png=med)
    log("平滑完成（2 x MedianFilter3）")

    # ---- 保边混合 ----
    set_phase("edge", "EDGE-BLEND / 03")
    gray = orig.mean(axis=2)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    grad = np.hypot(gx, gy)
    gmax = float(np.percentile(grad, 99)) + 1e-6
    wk = np.clip(np.clip(grad / gmax, 0, 1) * edge * 3, 0, 1)[..., None]
    edge_sm = sm * (1 - wk) + orig * wk
    wmax = float(wk.max())
    add_frame("edge", "EDGE-BLEND / 03",
              "按梯度把原图混回平滑图，保住边缘。"
              "注意 wk 上限 = EDGE_KEEP*3，实测经常贴顶饱和",
              {"EDGE_KEEP": edge, "wk_max": round(wmax, 3),
               "wk_mean": round(float(wk.mean()), 4)},
              png=Image.fromarray(np.clip(edge_sm, 0, 255).astype(np.uint8)))
    log(f"保边混合完成  wk_max={wmax:.3f}  wk_mean={float(wk.mean()):.4f}")

    # ---- 量化 ----
    set_phase("quantize", "QUANTIZE / 04")
    flat = edge_sm.reshape(-1, 3)
    # Oklab 近似：用 sRGB->Oklab 线性化矩阵（与 vec.py 同思路）
    def to_oklab(rgb01):
        r, g, b = rgb01[..., 0], rgb01[..., 1], rgb01[..., 2]
        l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
        m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
        s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
        l_ = np.cbrt(np.maximum(l, 0))
        m_ = np.cbrt(np.maximum(m, 0))
        s_ = np.cbrt(np.maximum(s, 0))
        return np.stack([
            0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
            1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
            0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
        ], axis=-1)

    def from_oklab(lab):
        L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
        l_ = L + 0.3963377774 * a + 0.2158037573 * b
        m_ = L - 0.1055613458 * a - 0.0638541728 * b
        s_ = L - 0.0894841775 * a - 1.2914855480 * b
        l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
        return np.stack([
            +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
            -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
            -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
        ], axis=-1)

    from sklearn.cluster import MiniBatchKMeans
    lab = to_oklab(flat / 255.0)
    km = MiniBatchKMeans(n_clusters=colors, random_state=42,
                         batch_size=16384, n_init=1, max_iter=60)
    idx = km.fit_predict(lab)
    cen = from_oklab(km.cluster_centers_) * 255.0
    cen = np.clip(cen, 0, 255)
    qimg = cen[idx].reshape(WH, WW, 3).astype(np.uint8)
    idx2 = idx.reshape(WH, WW)
    add_frame("quantize", "QUANTIZE / 04",
              f"Oklab 空间 KMeans 聚类到 {colors} 色。"
              f"这是误差的主要来源之一",
              {"色数 N": colors, "聚类算法": "MiniBatchKMeans"},
              png=Image.fromarray(qimg))
    log(f"量化完成 {colors} 色")

    # ---- 逐层描摹 ----
    order = list(np.argsort(-np.bincount(idx, minlength=colors)))
    counts = np.bincount(idx, minlength=colors)
    # 底色：取四边 6% 带宽的中位色（与 BASECOL 一致）
    bh = max(1, int(WH * 0.06))
    bw = max(1, int(WW * 0.06))
    edges_px = np.concatenate([
        orig[:bh].reshape(-1, 3), orig[-bh:].reshape(-1, 3),
        orig[:, :bw].reshape(-1, 3), orig[:, -bw:].reshape(-1, 3)])
    base_rgb = np.median(edges_px, axis=0)
    # 贴到最近簇中心
    d = ((cen - base_rgb) ** 2).sum(axis=1)
    base_hex = hex2(cen[int(np.argmin(d))])
    log(f"底色 BASECOL = {base_hex}（边框带中位色贴最近簇）")

    set_phase("trace", "TRACE / 05")
    minarea = max(4.0, WW * WH / 400000.0)
    eps = 1.2
    # 先算出总层数（供顶栏 LAYER x/y 与标题使用）
    real_total = int(sum(1 for ci in order if int(counts[ci]) >= minarea))
    with LOCK:
        STATE["params"]["总层数"] = real_total
        STATE["layer_total"] = real_total
        STATE["layer_done"] = 0
    log(f"预计描摹 {real_total} 层（每层一帧）")
    inner = []
    drawn = 0
    t_trace = time.time()
    for ci in order:
        npx = int(counts[ci])
        if npx < minarea:
            continue
        mask = (idx2 == ci)
        polys = get_polys(mask, minarea, eps)
        if not polys:
            continue
        c_hex = hex2(cen[ci])
        d_str = " ".join(poly_d(p) for p in polys)
        inner.append(f'<path fill="{c_hex}" d="{d_str}"/>')
        drawn += 1
        # 每层一帧（不抽稀，用户要求逐层展示）
        add_frame("trace", f"TRACE / 层 {drawn}/{real_total}",
                  f"描摹第 {drawn} 层（颜色 {c_hex}，"
                  f"占比 {100 * npx / idx2.size:.2f}%）",
                  {"层号": drawn, "总层数": real_total,
                   "颜色": c_hex, "覆盖像素": npx,
                   "占总面积": f"{100 * npx / idx2.size:.2f}%"},
                  svg=svg_wrap("".join(inner), WW, WH, base_hex))
        with LOCK:
            STATE["layer_done"] = drawn
        if drawn % 10 == 0 or drawn == real_total:
            log(f"  描摹 {drawn}/{real_total} 层…")
    log(f"描摹完成 {drawn} 层，用时 {time.time() - t_trace:.1f}s")

    # ---- 权威结果：调用真正的 vec.py ----
    set_phase("final", "FINAL / 99")
    log("调用权威管线 vec.py 生成最终 SVG…")
    env = {k: v for k, v in os.environ.items() if not k.startswith("VEC_")}
    env["VEC_N"] = str(colors)
    env["VEC_EDGE"] = str(edge)
    env["VEC_TITLE"] = os.path.splitext(os.path.basename(src))[0]
    env["VEC_SEED"] = "42"
    # ---- k 真实生效：把放大后的图作为 vec.py 的输入 ----
    # vec.py 内部有 WORK_W = min(VEC_W, W0 * 2)，其中 W0 是**它读到的源图宽度**。
    # 若直接传原始小图，W0 = 小图宽，min() 会把 k 夹到最多 2 倍。
    # 因此 k>1 时改传已放大的 ws（临时文件），使 W0 即放大后的宽度。
    vec_src = src
    tmp_big = None
    if target_k > 1:
        tmp_big = os.path.join(
            os.path.dirname(os.path.abspath(svg_out)),
            "." + os.path.splitext(os.path.basename(svg_out))[0]
            + f"_k{target_k}_{os.getpid()}_src.png")
        try:
            ws.save(tmp_big, "PNG")
            vec_src = tmp_big
            log(f"  k={target_k} 生效：以 {WW}x{WH} 放大图作为矢量化输入")
            with LOCK:
                STATE["params"]["vec_src_size"] = f"{WW}x{WH}"
        except Exception as e:
            log(f"  放大图存盘失败，回退原图：{e}", "error")
            vec_src = src
            tmp_big = None
    # 用 Popen 而非 run：run 无法中途终止，看门狗退出时 vec.py 会变孤儿
    # 记录旧产物状态：目录里可能残留上一次的 out。vec.py 若写盘失败
    # （PermissionError 等），旧文件仍在 —— 只看 exists 会把失败报成成功。
    _old_stamp = None
    try:
        if os.path.exists(svg_out):
            _st = os.stat(svg_out)
            _old_stamp = (_st.st_mtime_ns, _st.st_size)
    except OSError:
        _old_stamp = None

    _vp = subprocess.Popen(
        [PY, os.path.join(SK, "vec.py"), vec_src, svg_out],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
        env=env, cwd=SK)
    with _RL:
        VEC_PROC[0] = _vp
    try:
        _out, _ = _vp.communicate()
    finally:
        with _RL:
            VEC_PROC[0] = None

    class _R:
        pass

    r = _R()
    r.returncode = _vp.returncode
    r.stdout = _out or ""
    r.stderr = ""    # stderr 已并入 stdout
    if tmp_big and os.path.exists(tmp_big):
        try:
            os.remove(tmp_big)
        except OSError:
            pass
    if r.returncode != 0:
        # _run_child 把 stderr 并入了 stdout（r.stderr 恒为空），
        # 所以必须从 stdout 里找真正的错误行，否则永远只能报 "unknown"。
        _lines = [ln.strip() for ln in (r.stdout or "").splitlines()
                  if ln.strip()]
        _pick = ""
        for ln in _lines:
            low = ln.lower()
            if ("error" in low or "exception" in low
                    or "traceback" in low or "cannot" in low
                    or "not found" in low or "errno" in low):
                _pick = ln
        if not _pick and _lines:
            _pick = _lines[-1]
        err = _pick or f"返回码 {r.returncode}（无输出）"
        log(f"vec.py 失败：{err}", "error")
        if _lines:
            # 多打两行上下文，便于定位
            log("  vec.py 输出尾部：" + " | ".join(_lines[-3:]), "error")
        with LOCK:
            STATE["error"] = err
        return False

    # 渲染最终 SVG 并测误差
    png_out = svg_out.replace(".svg", "_render.png")
    subprocess.run([PY, os.path.join(SK, "render_check.py"),
                    svg_out, png_out], capture_output=True, cwd=SK)
    if os.path.exists(png_out):
        v = Image.open(png_out).convert("RGB")
        if v.size != (W0, H0):
            v = v.resize((W0, H0), Image.LANCZOS)
        ref = np.asarray(im).astype(np.float32)
        ren = np.asarray(v).astype(np.float32)
        e = np.abs(ref - ren)
        em = e.max(axis=2)
        mae = float(e.mean())
        p99 = float(np.percentile(em, 99))
        gt60 = 100 * float((em > 60).mean())
        # png 只作"预览占位"：SVG 加载完成的那一刻会被立刻替换掉
        _fid = add_frame(
            "final", "FINAL / 99",
            "权威管线最终产出（含 BASECOL + SEAL 补缝）。"
            "这一帧就是交付的 SVG。",
            {"MAE": round(mae, 3), "p99": round(p99, 1),
             "gt60": f"{gt60:.2f}%",
             "SVG 体积": f"{os.path.getsize(svg_out) / 1048576:.1f} MB"},
            png=v)
        with LOCK:
            for _f in STATE["frames"]:
                if _f["id"] == _fid:
                    _f["final_svg"] = True
                    break
        with LOCK:
            STATE["final_svg_path"] = svg_out
        with LOCK:
            STATE["params"].update({
                "mae": round(mae, 3), "p99": round(p99, 1),
                "gt60": round(gt60, 2),
                "svg_mb": round(
                    os.path.getsize(svg_out) / 1048576, 1)})
        log(f"最终结果 MAE {mae:.3f}  p99 {p99:.0f}  gt60 {gt60:.2f}%")
    else:
        add_frame("final", "FINAL / 99",
                  "权威管线最终产出", {}, svg=open(
                      svg_out, encoding="utf-8").read())

    # 保存路径：先记录，最后一条再输出（用户要能在最末一眼看到）
    # 判定「本次真的写成功」：文件存在 **且** mtime/size 与起跑前不同。
    # 只看 exists 会把「目录里残留的旧文件」当成本次产物。
    _ok_out = False
    try:
        if os.path.exists(svg_out):
            _st2 = os.stat(svg_out)
            _now = (_st2.st_mtime_ns, _st2.st_size)
            _ok_out = (_old_stamp is None) or (_now != _old_stamp)
    except OSError:
        _ok_out = False
    if _ok_out:
        FINAL_SAVED = (svg_out, "矢量 SVG")
        FINAL_INFO[0], FINAL_INFO[1] = FINAL_SAVED
        OUT_WRITTEN[0] = True
    else:
        FINAL_SAVED = None
        log("vec.py 报成功但目标文件未更新（可能写盘失败或被占用），"
            "不视为本次产物", "error")

    with LOCK:
        STATE["elapsed"] = time.time() - t0
        STATE["done"] = True
        STATE["phase"] = "done"
        STATE["phase_text"] = "COMPLETE"
    log(f"全部完成，总耗时 {time.time() - t0:.1f}s")
    if FINAL_SAVED and os.path.exists(FINAL_SAVED[0]):
        log(f"SAVED  {os.path.abspath(FINAL_SAVED[0])}  "
            f"({os.path.getsize(FINAL_SAVED[0])/1048576:.1f} MB)", "path")
    return True


# ---------------------------------------------------------------- 网页

HTML = r"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>VECTORIZATION / RHINE LAB</title>
<style>
/* ============================================================
   Rhine Lab palette — 取自 rhine-lock frontend/src/app.css 真实 token
   三态由 --denied (0..1) 推进：
     0.0  READY      浅纸底 #e7e5e1 + 柠檬绿 #b9e432
     0.5  VERIFYING  浅纸底，绿略降
     1.0  DENIED     黑底 #0b0b0c + 橙 #ff7b00
   ============================================================ */
:root{
  --denied: 0;
  --orange:#ff7b00; --orange-deep:#e35d00;
  --lime:#b9e432;
  --paper: color-mix(in srgb, #e7e5e1, #0b0b0c calc(var(--denied) * 100%));
  --ink:   color-mix(in srgb, #111214, #f2f0eb calc(var(--denied) * 100%));
  --ink-inverse: color-mix(in srgb, #f2f0eb, #111214 calc(var(--denied) * 100%));
  --mute:  color-mix(in srgb, #8b8882, #6f6d68 calc(var(--denied) * 100%));
  --faint: color-mix(in srgb, #c9c6c0, #2b2b2d calc(var(--denied) * 100%));
  --card:  color-mix(in srgb, #f6f5f2, #151618 calc(var(--denied) * 100%));
  --tab:   color-mix(in srgb, #e2e0dc, #1d1e20 calc(var(--denied) * 100%));
  --accent: color-mix(in srgb, var(--lime), var(--orange) calc(var(--denied) * 100%));
  --accent-deep: color-mix(in srgb, #8fae20, var(--orange-deep) calc(var(--denied) * 100%));
  --mono:"JetBrains Mono","Cascadia Mono",Consolas,"SF Mono",Menlo,monospace;
  --display:"Montserrat","MiSans","Noto Sans CJK SC","PingFang SC",sans-serif;
}
/* ---------- 三套独立配色（覆盖上面的混色 token） ----------
   对比度实测（WCAG）：accent 用于【文字】时必须 >=4.5，
   所以把浅色 accent 压深为 accent-deep；浅调只用于装饰线/色块。
   ---------------------------------------------------------- */
/* 1 READY: 浅纸底 + 黄绿（rhine-lock 原味，但文字用深黄绿） */
html.theme-ready{
  --paper:#e7e5e1; --ink:#111214; --mute:#57544f; --faint:#cfccc6;
  --card:#f6f5f2; --tab:#e2e0dc;
  --accent:#7a9c12; --accent-deep:#4d6a00;   /* 文字用 deep 4.5+ */
}
/* 2 VERIFYING: 冷灰底 + 天青蓝（与 READY 明显不同） */
html.theme-verify{
  --paper:#dfe3e8; --ink:#101821; --mute:#4e5a66; --faint:#b8c2cc;
  --card:#eef1f4; --tab:#cfd6dd;
  --accent:#1771bf; --accent-deep:#0b4f88;
}
/* 3 DENIED: 黑底 + 琥珀橙（WARNING lockdown） */
html.theme-denied{
  --paper:#0b0b0c; --ink:#f2f0eb; --mute:#8d8a84; --faint:#2b2b2d;
  --card:#151618; --tab:#1d1e20;
  --accent:#ff7b00; --accent-deep:#ffa040;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%}
body{
  background:var(--paper); color:var(--ink);
  font:13.5px/1.65 var(--mono);
  display:flex; flex-direction:column; overflow:hidden;
  user-select:none; -webkit-font-smoothing:antialiased;
  transition:background .28s ease,color .28s ease;
}
/* 顶部细条（莱茵式） */
body::before{
  content:"";position:fixed;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,var(--accent),transparent 72%);
  z-index:100;transition:background .28s;
}
/* ---------- 顶栏 ---------- */
header{
  height:56px;flex:0 0 56px;display:flex;align-items:center;gap:16px;
  padding:0 16px;border-bottom:1px solid var(--faint);
  background:var(--card);transition:background .28s,border-color .28s;
}
.brand{display:flex;align-items:center;gap:10px;flex:0 0 auto}
.mark{
  width:28px;height:28px;border:1.5px solid var(--ink);
  display:grid;place-items:center;color:var(--ink);
  font:800 15px/1 var(--display);letter-spacing:-2px;
  transition:border-color .28s,color .28s;
}
.brand h1{
  font:800 12px/1.1 var(--display);letter-spacing:2.6px;
  text-transform:uppercase;color:var(--ink);white-space:nowrap;
}
.brand small{
  display:block;font:600 10.5px/1.45 var(--mono);letter-spacing:1.2px;
  color:var(--mute);text-transform:uppercase;margin-top:2px;
}
.spacer{flex:1}
/* 读数块 */
.hstat{display:flex;gap:2px}
.hstat .u{
  min-width:78px;padding:6px 10px;background:var(--tab);
  border-left:2px solid var(--accent);
  transition:background .28s,border-color .28s;
}
.hstat b{
  display:block;font:600 15px/1 var(--mono);color:var(--ink);
  font-variant-numeric:tabular-nums;letter-spacing:.2px;
}
.hstat span{
  display:block;font:600 10px/1.5 var(--mono);letter-spacing:1.4px;
  color:var(--mute);text-transform:uppercase;margin-top:3px;
}
/* 配色切换 */
.themes{display:flex;gap:4px;margin-left:6px}
.th{
  width:26px;height:26px;border:1px solid var(--faint);cursor:pointer;
  background:transparent;position:relative;transition:.18s;
  display:grid;place-items:center;
}
.th:hover{border-color:var(--ink);transform:translateY(-1px)}
.th.on{border-color:var(--ink);box-shadow:0 0 0 1px var(--ink) inset}
.th i{width:12px;height:12px;display:block;border-radius:1px}
.th.t1 i{background:#b9e432}
.th.t2 i{background:linear-gradient(135deg,#b9e432,#e7e5e1)}
.th.t3 i{background:#ff7b00}
/* 状态灯 */
.dot{
  width:9px;height:9px;border-radius:50%;background:var(--accent);
  box-shadow:0 0 0 3px color-mix(in srgb,var(--accent) 22%,transparent);
  flex:0 0 9px;transition:background .28s;
  animation:pulse 1.6s ease-in-out infinite;
}
@keyframes pulse{50%{opacity:.35}}
.dot.done{animation:none}
/* ---------- 主体 ---------- */
main{flex:1;display:flex;min-height:0}
.work{flex:1;display:flex;flex-direction:column;min-width:0;
  border-right:1px solid var(--faint)}
.stagewrap{
  flex:1 1 auto;position:relative;
  min-height:0;box-sizing:border-box;
  padding:16px;overflow:hidden;
  background:
    linear-gradient(var(--faint) 1px,transparent 1px) 0 0/100% 28px,
    linear-gradient(90deg,var(--faint) 1px,transparent 1px) 0 0/28px 100%,
    var(--paper);
  background-blend-mode:normal;transition:background .28s;
}
.stagewrap::before,.stagewrap::after{
  content:"";position:absolute;width:14px;height:14px;
  border:1px solid var(--mute);
}
.stagewrap::before{top:14px;left:14px;border-right:0;border-bottom:0}
.stagewrap::after{bottom:14px;right:14px;border-left:0;border-top:0}
/* #stage 用 absolute + inset 撑满 .stagewrap 的内边距盒。
   宽高都由容器直接决定（不依赖百分比高度推导），
   这样 <img> 的 max-height 才有意义，任何帧都不会出界。 */
#stage{
  position:absolute;inset:16px;
  display:flex;align-items:center;justify-content:center;
  overflow:hidden;
}
/* 统一路径：所有帧都是 <img>（描摹帧已在服务端栅格化成 PNG）。
   用 max-* + auto 按固有比例装进 #stage 这个确定盒子。 */
#stage img{
  max-width:100%;max-height:100%;
  width:auto;height:auto;
  object-fit:contain;object-position:center;
  display:block;
}
/* SVG 是替换元素：width/height=100% 会按容器撑开，
   靠 preserveAspectRatio(meet) + 这里的 max 约束双保险 */
#stage svg{
  max-width:100%;max-height:100%;
  width:auto;height:auto;display:block;
}
/* 工作区本身也不允许子元素撑破 */
.stagewrap{overflow:hidden;min-width:0;min-height:0}
#stage img,#stage svg{
  max-width:100%;max-height:100%;display:block;
  box-shadow:0 2px 22px rgba(0,0,0,.16),0 0 0 1px var(--faint);
}
/* ---------- 工作区缩放层（仅最后一帧可交互） ----------
   #zoom 承载 translate + scale；#stage 的 overflow:hidden
   会把超出部分裁掉，因此放大的图永远不会盖住时间轴/右栏。 */
#zoom{
  position:relative;
  display:flex;align-items:center;justify-content:center;
  width:100%;height:100%;
  transform-origin:center center;
  will-change:transform;
}
/* finalLayer 常驻底层放已加载好的 SVG；frameLayer 在上显示当前帧 */
#finalLayer,#frameLayer{
  position:absolute;inset:0;
  display:flex;align-items:center;justify-content:center;
}
#finalLayer{
  z-index:1;
  opacity:0;                 /* 隐藏但保持渲染，避免切 display 触发重排 */
  visibility:visible;
  pointer-events:none;
  will-change:opacity;       /* 留在合成层，保证 opacity:0 时仍完成光栅化 */
  transition:opacity .16s ease;
}
#finalLayer.show{opacity:1}
#frameLayer{z-index:2}
#finalLayer img,#frameLayer img{
  max-width:100%;max-height:100%;width:auto;height:auto;
  display:block;user-select:none;-webkit-user-drag:none;
  box-shadow:0 2px 22px rgba(0,0,0,.16),0 0 0 1px var(--faint);
}
#zoom.grab{cursor:grab}
#zoom.grabbing{cursor:grabbing}
#zoom img{max-width:100%;max-height:100%;width:auto;height:auto;
  display:block;user-select:none;-webkit-user-drag:none;
  box-shadow:0 2px 22px rgba(0,0,0,.16),0 0 0 1px var(--faint)}
/* 缩放控制条：绝对定位在 #stage 右下角，属于工作区内部 */
/* 缩放条：常驻于时间轴顶部那一行（与 ◀ ▶ ▶ / 帧号同排）；不可用时灰显 */
/* 自动播放速度滑块：位于帧标题与缩放条之间 */
#speedCtl{
  display:flex;align-items:center;gap:5px;flex:0 0 auto;
  margin:0 2px;padding:0 8px;
  border-left:1px solid var(--faint);
  font:600 11px/1 var(--mono);color:var(--mute);
}
#speedCtl .hint{
  color:var(--mute);font-weight:500;font-size:9.5px;
  letter-spacing:.3px;white-space:nowrap;
}
/* 倍速按钮（仿 B 站：点击展开列表，当前项高亮） */
#speedBtn{
  background:transparent;border:1px solid var(--faint);
  color:var(--ink);cursor:pointer;
  font:700 11px/1 var(--mono);
  padding:4px 8px;min-width:44px;text-align:center;
  transition:.16s;font-variant-numeric:tabular-nums;
}
#speedBtn:hover{border-color:var(--ink);background:var(--tab)}
#speedCtl{position:relative}
/* 弹出列表：向上展开，避免被底部裁掉 */
#speedMenu{
  display:none;position:absolute;right:0;bottom:calc(100% + 6px);
  min-width:92px;padding:4px 0;z-index:50;
  background:var(--card);border:1px solid var(--faint);
  box-shadow:0 -4px 18px rgba(0,0,0,.18);
  /* 列表在按钮上方：用 column-reverse 让 2.0x 排在顶部、0.5x 排在底部 */
  flex-direction:column-reverse;
}
#speedMenu.open{display:flex}
#speedMenu .sp{
  display:block;width:100%;flex:0 0 auto;text-align:center;
  background:transparent;border:0;cursor:pointer;
  font:600 11.5px/1 var(--mono);color:var(--mute);
  padding:7px 10px;transition:.14s;
  font-variant-numeric:tabular-nums;
}
#speedMenu .sp:hover{background:var(--tab);color:var(--ink)}
/* 当前倍速高亮（B 站用主题色） */
#speedMenu .sp.on{color:var(--accent-deep);font-weight:800}
#speedMenu .sp.on::after{content:" \2713";font-size:10px}
#zoombar{
  display:flex;align-items:center;gap:3px;
  margin:0 2px;padding:0 8px;
  border-left:1px solid var(--faint);
  border-right:1px solid var(--faint);
  font:600 11px/1 var(--mono);color:var(--ink);
  font-variant-numeric:tabular-nums;
  transition:opacity .2s,border-color .2s;
  flex:0 0 auto;
}
#zoombar b{font-weight:700;min-width:38px;text-align:right;
  font-size:11px;color:var(--accent-deep)}
#zoombar button{
  background:transparent;border:1px solid var(--faint);
  color:var(--mute);cursor:pointer;font:600 11px/1 var(--mono);
  padding:4px 7px;transition:.16s;line-height:1;
}
#zoombar button:hover:not(:disabled){border-color:var(--ink);color:var(--ink);
  background:var(--tab)}
#zoombar .hint{color:var(--mute);font-weight:500;font-size:9.5px;
  letter-spacing:.3px;margin-right:2px;white-space:nowrap}
/* 不可用：整条灰显，字体淡出，按钮禁用 */
#zoombar.disabled{opacity:.45}
#zoombar.disabled b{color:var(--mute)}
#zoombar.disabled .hint{color:var(--faint)}
#zoombar.disabled button{
  color:var(--faint);border-color:var(--faint);cursor:not-allowed;
}
.empty{color:var(--mute);font-size:12px;letter-spacing:1.2px}
/* ---------- 时间轴 ---------- */
.timeline{flex:0 0 auto;padding:12px 16px 14px;
  border-top:1px solid var(--faint);background:var(--card);
  transition:background .28s,border-color .28s}
.tl-top{display:flex;align-items:center;gap:10px;margin-bottom:9px}
.tl-btn{
  width:26px;height:26px;flex:0 0 26px;display:grid;place-items:center;
  background:transparent;border:1px solid var(--faint);color:var(--mute);
  cursor:pointer;font-size:12px;transition:.16s;
}
.tl-btn:hover:not(:disabled){border-color:var(--ink);color:var(--ink)}
.tl-btn:disabled{opacity:.32;cursor:not-allowed}
.tl-title{
  flex:1;font:700 12px/1.4 var(--mono);letter-spacing:1px;
  color:var(--ink);white-space:nowrap;overflow:hidden;
  text-overflow:ellipsis;text-transform:uppercase;
}
.tl-title em{color:var(--accent-deep);font-style:normal}
.tl-idx{font:600 11.5px/1 var(--mono);color:var(--mute);
  font-variant-numeric:tabular-nums;white-space:nowrap}
input[type=range]{
  -webkit-appearance:none;width:100%;height:4px;
  background:var(--faint);outline:none;cursor:pointer;border-radius:2px;
}
input[type=range]::-webkit-slider-thumb{
  -webkit-appearance:none;width:12px;height:12px;border-radius:50%;
  background:var(--accent);border:2px solid var(--paper);
  box-shadow:0 0 0 1px var(--accent);transition:.16s;
}
input[type=range]::-moz-range-thumb{
  width:12px;height:12px;border-radius:50%;background:var(--accent);
  border:2px solid var(--paper);
}
.ticks{
  display:flex;justify-content:space-between;margin-top:6px;
  font:600 10px/1 var(--mono);color:var(--mute);letter-spacing:.5px;
}
/* ---------- 右侧 ---------- */
aside{flex:0 0 348px;display:flex;flex-direction:column;
  background:var(--card);min-height:0;transition:background .28s}
.sec{border-bottom:1px solid var(--faint)}
.sec-h{
  display:flex;align-items:center;gap:8px;padding:8px 13px;
  font:700 10.5px/1 var(--mono);letter-spacing:1.9px;
  color:var(--ink);background:var(--tab);
  border-left:3px solid var(--accent);text-transform:uppercase;
  transition:background .28s,border-color .28s;
}
.sec-h i{
  font-style:normal;font:600 10px/1 var(--mono);
  color:var(--mute);letter-spacing:1px;
}
.sec-b{padding:10px 13px 13px;font-size:12px}
.kv{display:flex;justify-content:space-between;gap:10px;
  padding:3.5px 0;border-bottom:1px dotted var(--faint)}
.kv:last-child{border-bottom:0}
.kv span{color:var(--mute);white-space:nowrap;font-size:11.5px}
.kv b{color:var(--ink);font-weight:600;text-align:right;
  font-variant-numeric:tabular-nums;word-break:break-all;font-size:12px}
.kv b.hot{color:var(--accent-deep)}
.logbox{flex:1;overflow:auto;padding:9px 13px 14px;font-size:11.5px;
  line-height:1.7;min-height:0}
.logbox::-webkit-scrollbar{width:5px}
.logbox::-webkit-scrollbar-thumb{
  background:color-mix(in srgb,var(--mute) 45%,transparent);border-radius:3px}
.lg{display:flex;gap:8px;padding:.5px 0;color:var(--mute)}
.lg .t{color:var(--mute);flex:0 0 46px;
  font-variant-numeric:tabular-nums}
.lg.phase .m{color:var(--ink);font-weight:600}
.lg.error .m{color:var(--orange-deep)}
.lg.ok .m{color:var(--accent-deep);font-weight:600}
.lg.path .m{color:var(--accent-deep);word-break:break-all;font-weight:600}
.desc{color:var(--mute);font-size:12px;line-height:1.85;
  padding:9px 13px 12px;border-bottom:1px solid var(--faint)}
.desc b{color:var(--ink);font-weight:600}
</style></head><body>

<header>
  <div class="brand">
    <div class="mark">&#8734;</div>
    <div><h1>Vectorization Monitor</h1>
      <small>Rhine Lab / SVG Reconstruction</small></div>
  </div>
  <div class="spacer"></div>
  <div class="hstat">
    <div class="u"><b id="hPhase">--</b><span>Phase</span></div>
    <div class="u"><b id="hLayer">0/0</b><span>Layer</span></div>
    <div class="u"><b id="hFrame">0/0</b><span>Frame</span></div>
    <div class="u"><b id="hTime">0.0s</b><span>Elapsed</span></div>
  </div>
  <div class="themes">
    <button class="th t1 on" id="th1" title="READY 浅纸+柠檬绿"></button>
    <button class="th" id="th2" title="VERIFYING 浅纸+绿"></button>
    <button class="th" id="th3" title="DENIED 黑底+橙"></button>
  </div>
  <div class="dot" id="hDot"></div>
</header>

<main>
  <div class="work">
    <div class="stagewrap"><div id="stage">
      <div id="zoom">
        <div id="finalLayer"></div>
        <div id="frameLayer">
          <span class="empty">AWAITING PIPELINE…</span>
        </div>
      </div>
    </div></div>
    <div class="timeline">
      <div class="tl-top">
        <button class="tl-btn" id="bPrev" title="上一帧 (←)">&#9664;</button>
        <button class="tl-btn" id="bPlay" title="播放/暂停 (空格)">&#9654;</button>
        <button class="tl-btn" id="bNext" title="下一帧 (→)">&#9654;</button>
        <div class="tl-title" id="tlTitle">--</div>
        <div id="speedCtl">
          <span class="hint">倍速</span>
          <button id="speedBtn" title="播放倍速">1.0x</button>
          <div id="speedMenu">
            <button class="sp" data-v="0.5">0.5x</button>
            <button class="sp" data-v="0.75">0.75x</button>
            <button class="sp" data-v="1">1.0x</button>
            <button class="sp" data-v="1.25">1.25x</button>
            <button class="sp" data-v="1.5">1.5x</button>
            <button class="sp" data-v="2">2.0x</button>
          </div>
        </div>
        <div id="zoombar" class="disabled">
          <span class="hint">拖动 / 滚轮 / 双击复位</span>
          <button id="zOut" disabled>−</button>
          <b id="zVal">1.00x</b>
          <button id="zIn" disabled>+</button>
          <button id="zReset" disabled>复位</button>
        </div>
        <div class="tl-idx" id="tlIdx">0 / 0</div>
        <button class="tl-btn" id="bLast" title="跳到最新 (End)">&#8674;</button>
      </div>
      <input type="range" id="slider" min="0" max="0" value="0" step="1">
      <div class="ticks" id="ticks"></div>
    </div>
  </div>

  <aside>
    <div class="sec">
      <div class="sec-h"><i>Sec.01</i> Parameters</div>
      <div class="sec-b" id="params"></div>
      <div id="finalBadge" style="display:none;padding:6px 13px 10px;
        font:600 11px/1.5 var(--mono);border-top:1px solid var(--faint)">
      </div>
    </div>
    <div class="sec">
      <div class="sec-h"><i>Sec.02</i> Stage</div>
      <div class="desc" id="desc">--</div>
    </div>
    <div class="sec-h" style="border-bottom:1px solid var(--faint)">
      <i>Sec.03</i> Progress Log</div>
    <div class="logbox" id="logs"></div>
  </aside>
</main>

<script>
let FRAMES=[], LOGS=[], PARAMS={}, cur=-1, playing=false, playTimer=null;
let userSeek=false;
const $=id=>document.getElementById(id);

/* ---------- 配色切换 ---------- */
/* 三个主题不再是"同一套配色加滤镜"，而是各自独立的 palette。
   rhine-lock 的三态靠 --denied 线性混色，视觉差异不够；
   这里给每个主题单独覆盖关键 token，保证一眼可辨。 */
const THEMES={
  1:{cls:'theme-ready',  name:'READY'},
  2:{cls:'theme-verify', name:'VERIFYING'},
  3:{cls:'theme-denied', name:'DENIED'}
};
function setTheme(n){
  const t=THEMES[n]||THEMES[1];
  const el=document.documentElement;
  ['theme-ready','theme-verify','theme-denied'].forEach(c=>{
    el.classList.remove(c);
  });
  el.classList.add(t.cls);
  ['th1','th2','th3'].forEach((id,k)=>{
    $(id).classList.toggle('on', k+1===n);
  });
  try{localStorage.setItem('rhineTheme',String(n));}catch(e){}
}
[1,2,3].forEach(n=>{$('th'+n).onclick=()=>setTheme(n);});
try{
  const saved=parseInt(localStorage.getItem('rhineTheme')||'1',10);
  if(THEMES[saved]) setTheme(saved);
}catch(e){}

/* ---------- 连续计时器 ----------
   问题：轮询 ~320ms 一次，服务端 elapsed 只在有事件时更新，
        直接显示会让数字一跳一跳。
   解法：拿到服务器时间时记录 (serverElapsed, perfNow)；
        每帧按 performance.now() 推算，渲染时保证单调递增。 */
let T0=null;           // {server:秒, perf:毫秒}
let lastShown=-1;
function calibrate(serverElapsed){
  T0={server:serverElapsed, perf:performance.now()};
  if(serverElapsed>lastShown) lastShown=serverElapsed;
}
function tickElapsed(){
  if(T0){
    let v=T0.server+(performance.now()-T0.perf)/1000;
    if(v<lastShown) v=lastShown;      // 永不回退
    else lastShown=v;
    const frozen = STATE_DONE ? true : false;
    if(!frozen) $('hTime').textContent=v.toFixed(1)+'s';
  }
  requestAnimationFrame(tickElapsed);
}
let STATE_DONE=false;

function esc(s){
  return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;',
    '>':'&gt;'}[c]));
}

function renderParams(){
  const keys=Object.keys(PARAMS);
  if(!keys.length){
    $('params').innerHTML='<span style="color:var(--mute)">'+
      '等待管线启动…</span>';return;}
  const HOT=new Set(['colors','edge','k','mae','p99','gt60','总层数','超采样 W']);
  $('params').innerHTML=keys.map(k=>{
    let v=PARAMS[k];
    if(v===null||v===undefined)v='--';
    const hot=HOT.has(k)?' hot':'';
    return `<div class="kv"><span>${esc(k)}</span>`+
      `<b class="${hot.trim()}">${esc(v)}</b></div>`;
  }).join('');
}

function renderLogs(){
  const el=$('logs');
  const atBottom=el.scrollHeight-el.scrollTop-el.clientHeight<44;
  el.innerHTML=LOGS.map(l=>
    `<div class="lg ${l.level}"><span class="t">${l.t.toFixed(1)}</span>`+
    `<span class="m">${esc(l.text)}</span></div>`).join('');
  if(atBottom) el.scrollTop=el.scrollHeight;
}

/* ---------- 无闪烁换图 ----------
   绝不在旧图还在解码新图时清空容器：
   新图先在后台上解码，onload 后再一次性替换子元素。 */
const _imgCache=new Map();   // url -> 已解码的 HTMLImageElement

function setStageImg(url, cb){
  const z=$('frameLayer')||$('zoom');
  if(!url){
    z.innerHTML='<span class="empty">--</span>';
    if(cb) cb();
    return;
  }
  const paint=(srcEl)=>{
    // 一次性替换所有子节点，旧图在新图就绪后才移除
    const clone=srcEl.cloneNode();
    clone.setAttribute('draggable','false');
    z.replaceChildren(clone);
    if(cb) cb();
  };
  const cached=_imgCache.get(url);
  if(cached&&cached.complete&&cached.naturalWidth){
    paint(cached);
    return;
  }
  const im=new Image();
  im.onload=()=>{
    _imgCache.set(url, im);
    paint(im);                 // 此刻已解码完成，替换不会闪白
  };
  im.onerror=()=>{
    // 加载失败：保留旧图不动，不制造白屏
    if(cb) cb();
  };
  im.src=url;
  // 注意：此处旧图仍在 DOM 中，等待 onload 期间画面不空
}

function show(i){
  if(!FRAMES.length)return;
  i=Math.max(0,Math.min(FRAMES.length-1,i));
  // 到达最后一帧即恢复"跟随最新"：
  // 无论用户是拖滑条、点右箭头还是按 End 到这里，之后新帧都应自动显示。
  if(i===FRAMES.length-1) userSeek=false;
  cur=i;
  const f=FRAMES[i];
  const z=$('zoom');

  /* FINAL 帧：用真矢量 SVG 显示（按需加载，缓存一次）
     其余帧：用服务端栅格化好的 PNG（轻量）
     两者都走 <img>，object-fit:contain 保证不出界 */
  /* ---- final 常驻层的显隐 ----
     final 或 final-1 时取消隐藏；其余帧隐藏（用不到就不占画面） */
  const fl=$('finalLayer');
  const lastIdx = FRAMES.length-1;
  const isFinal = !!f.final_svg || (i===lastIdx && !!_finalSvgUrl);
  const isPrev  = (i===lastIdx-1) && !!_finalSvgUrl;
  const needSvg = isFinal || isPrev;

  if(needSvg && _finalSvgUrl){
    if(fl && !fl.querySelector('img')){
      const im=document.createElement('img');
      im.src=_finalSvgUrl;
      im.setAttribute('draggable','false');
      fl.replaceChildren(im);
    }
    if(fl) fl.classList.add('show');   // 只改透明度，不触发布局
  }else{
    if(fl) fl.classList.remove('show');
  }

  /* ---- frameLayer 放当前帧 ----
     isFinal：清空，露出底层 SVG（零停顿）
     其他（含 final-1）：显示该帧本身，自然覆盖在 final 之上 */
  if(isFinal){
    const fz=$('frameLayer');
    if(fz) fz.replaceChildren();
    if(!_finalSvgUrl){
      // SVG 未就绪：先给 PNG 预览，就绪后会自动重绘
      const pu=f.img?`/frame/${f.img}.png`:null;
      if(pu) setStageImg(pu);
      loadFinalSvg(()=>{ if(cur===i) show(i); });
    }
  }else if(f.img){
    setStageImg(`/frame/${f.img}.png`);
  }else if(f.svg){
    setStageImg('data:image/svg+xml;charset=utf-8,'
                +encodeURIComponent(f.svg));
  }else{
    setStageImg(null);
  }

  /* 任何帧都允许拖动与缩放（不再限制最后一帧） */
  // 有图即可交互：f.img(外置图) / f.png(内联图) / f.svg(兜底) 任一存在
  _zoomInteractive = !!(f.img || f.final_svg || f.png || f.svg);
  if(!_zoomInteractive) resetZoom();
  else { applyZoom(); }
  const zb=$('zoombar');
  if(zb){
    zb.classList.toggle('disabled', !_zoomInteractive);
    const btns=['zOut','zIn','zReset'];
    btns.forEach(id=>{ const b=$(id); if(b) b.disabled=!_zoomInteractive; });
  }
  z.classList.toggle('grab', _zoomInteractive);
  $('tlTitle').innerHTML=`<em>${esc(f.title)}</em>`;
  $('tlIdx').textContent=`${i+1} / ${FRAMES.length}`;
  $('slider').value=i;
  const meta=f.meta||{};
  let html=esc(f.desc||'--');
  const mk=Object.keys(meta);
  if(mk.length){
    html+='<div style="margin-top:8px">'+mk.map(k=>
      `<div class="kv"><span>${esc(k)}</span>`+
      `<b>${esc(meta[k])}</b></div>`).join('')+'</div>';
  }
  $('desc').innerHTML=html;
  $('hFrame').textContent=`${i+1}/${FRAMES.length}`;
  // LAYER 跟随当前帧
  if(f.kind==='trace'&&meta['层号']!==undefined){
    $('hLayer').textContent=`${meta['层号']}/${meta['总层数']||'?'}`;
  }
  $('bPrev').disabled=i<=0;
  $('bNext').disabled=i>=FRAMES.length-1;
}

function ticks(){
  if(!FRAMES.length)return;
  const n=FRAMES.length;
  const step=Math.max(1,Math.floor(n/7));
  let out=[];
  for(let i=0;i<n;i+=step){
    const f=FRAMES[i];
    let lb;
    if(f.kind==='trace'&&f.meta&&f.meta['层号']!==undefined)
      lb='L'+f.meta['层号'];
    else if(f.kind==='block') lb='BLK';
    else lb=f.title.split(' /')[0].slice(0,7);
    out.push(`<span>${lb}</span>`);
  }
  $('ticks').innerHTML=out.join('');
}

let _lastSig='';
let _lastSeenId=0;     // 已拉取到的最大帧 id（增量游标）
let _doneStopped=false;

async function poll(){
  try{
    const r=await fetch('/state?since='+_lastSeenId);
    const s=await r.json();

    /* 增量合并：只把新帧追加进本地数组 */
    const inc=s.frames||[];
    if(inc.length){
      for(const f of inc){
        if(!FRAMES.some(x=>x.id===f.id)) FRAMES.push(f);
        if(f.id>_lastSeenId) _lastSeenId=f.id;
      }
      FRAMES.sort((a,b)=>a.id-b.id);
    }

    /* 只有"内容真的变了"才重建 DOM，避免每 300ms 白跑一次 */
    const newLogs=s.logs||[];
    const sig=[FRAMES.length, s.phase_short, newLogs.length,
               s.layer_done, s.layer_total, s.done].join('|');
    const changed = sig!==_lastSig;
    LOGS=newLogs; PARAMS=s.params||{};
    STATE_DONE=!!s.done;

    $('slider').max=Math.max(0,FRAMES.length-1);
    $('hPhase').textContent=(s.phase_short||s.phase||'--').toUpperCase();
    $('hDot').className='dot'+(s.done?' done':'');
    calibrate(s.elapsed||0);
    if(s.done) $('hTime').textContent=(s.elapsed||0).toFixed(1)+'s';

    if(changed){
      if(s.layer_total>0 && cur<0)
        $('hLayer').textContent=`${s.layer_done||0}/${s.layer_total}`;
      renderParams(); renderLogs(); ticks();
      if(!userSeek&&FRAMES.length) show(FRAMES.length-1);
      _lastSig=sig;
      /* FINAL 帧一出现就后台预加载 SVG（不等用户切过去），
         等用户播到 final 时早已就绪 -> 无停顿 */
      if(FRAMES.some(x=>x.final_svg)) loadFinalSvg(null);
    }

    /* 任务完成后再同步一次就停止轮询，省掉空转 */
    if(s.done){
      if(_doneStopped) return;
      _doneStopped=true;
    }
  }catch(e){}
  setTimeout(poll, STATE_DONE?2000:300);
}

$('slider').addEventListener('input',e=>{
  userSeek=true; stopPlay(); resetZoom(); show(+e.target.value);
});
$('bPrev').onclick=()=>{userSeek=true;stopPlay();resetZoom();show(cur-1);};
$('bNext').onclick=()=>{userSeek=true;stopPlay();resetZoom();show(cur+1);};
$('bLast').onclick=()=>{userSeek=false;stopPlay();resetZoom();
  if(FRAMES.length)show(FRAMES.length-1);};
function stopPlay(){
  if(playTimer){clearInterval(playTimer);playTimer=null;}
  $('bPlay').innerHTML='&#9654;'; playing=false;
}
$('bPlay').onclick=()=>{
  if(playing){stopPlay();return;}
  userSeek=true; playing=true; $('bPlay').innerHTML='&#10073;&#10073;';
  playTimer=setInterval(()=>{
    if(cur>=FRAMES.length-1){stopPlay();return;}
    resetZoom(); show(cur+1);
  }, intervalMs());
};
/* 倍速菜单绑定（仿 B 站） */
function initSpeed(){
  const btn=$('speedBtn'), menu=$('speedMenu');
  if(!btn||!menu) return;

  btn.onclick=(e)=>{
    e.stopPropagation();
    menu.classList.toggle('open');
  };
  menu.querySelectorAll('.sp').forEach(el=>{
    el.onclick=(e)=>{
      e.stopPropagation();
      const v=parseFloat(el.dataset.v);
      if(SPEED_PRESETS.some(x=>Math.abs(x-v)<1e-6)) _playSpeed=v;
      menu.classList.remove('open');
      applySpeed();
    };
  });
  // 点其它地方收起
  document.addEventListener('click',()=>menu.classList.remove('open'));

  try{
    const saved=parseFloat(localStorage.getItem('playSpeed')||'1');
    if(SPEED_PRESETS.some(x=>Math.abs(x-saved)<1e-6)) _playSpeed=saved;
  }catch(e){}
  applySpeed();
}

document.addEventListener('keydown',e=>{
  if(e.key==='ArrowLeft'){userSeek=true;stopPlay();resetZoom();show(cur-1);e.preventDefault();}
  if(e.key==='ArrowRight'){userSeek=true;stopPlay();resetZoom();show(cur+1);e.preventDefault();}
  if(e.key===' '){$('bPlay').click();e.preventDefault();}
  if(e.key==='End'){userSeek=false;stopPlay();show(FRAMES.length-1);}
  if(e.key==='1'){setTheme(1);}
  if(e.key==='2'){setTheme(2);}
  if(e.key==='3'){setTheme(3);}
});
  /* ==================== 工作区缩放 / 拖动 ====================
   仅最后一帧（最新帧）可交互。
   所有位移与缩放都作用于 #zoom 的 transform，
   而 #stage 是 overflow:hidden 的确定尺寸盒子，
   因此放大后的内容会被裁在工作区内，绝不覆盖时间轴或右侧面板。
   ============================================================ */
let _zoomInteractive=false;
/* 自动播放速度：1.0 = 默认 420ms/帧 */
let _playSpeed=1;
const BASE_INTERVAL=420;

function intervalMs(){ return Math.round(BASE_INTERVAL/_playSpeed); }

const SPEED_PRESETS=[0.5,0.75,1,1.25,1.5,2];

function speedLabel(v){
  // B 站显示习惯：0.5x / 0.75x / 1.0x / 1.25x / 1.5x / 2.0x
  return (v===1?-1:v).toFixed(v===1?1:(v===2?1:2)).replace(/\.?0+$/,'')
         +'x';
}
function fmtSpeed(v){
  if(v===0.5) return '0.5x';
  if(v===0.75) return '0.75x';
  if(v===1) return '1.0x';
  if(v===1.25) return '1.25x';
  if(v===1.5) return '1.5x';
  if(v===2) return '2.0x';
  return v+'x';
}

function applySpeed(){
  const b=$('speedBtn');
  if(b) b.textContent=fmtSpeed(_playSpeed);
  const menu=$('speedMenu');
  if(menu){
    menu.querySelectorAll('.sp').forEach(el=>{
      const v=parseFloat(el.dataset.v);
      el.classList.toggle('on', Math.abs(v-_playSpeed)<1e-6);
    });
  }
  // 正在播放时立刻按新速度重建定时器
  if(playing){
    if(playTimer){clearInterval(playTimer);playTimer=null;}
    playTimer=setInterval(()=>{
      if(cur>=FRAMES.length-1){stopPlay();return;}
      resetZoom(); show(cur+1);
    }, intervalMs());
  }
  try{localStorage.setItem('playSpeed',String(_playSpeed));}catch(e){}
}
let _finalSvgUrl=null;      // 已成功解码的 SVG dataURL（有值即表示可显示）
let _finalSvgLoading=false;
let _finalSvgErr=null;

/* 按需加载最终 SVG（24MB 级），并且**等到浏览器真正解码成功**才置位。
   流程：fetch 文本 -> 转 dataURL -> new Image() 预加载
         -> onload 触发（此刻浏览器已把 SVG 渲染成图像）
         -> 写入 _finalSvgUrl 并立刻回调重绘（隐藏 PNG、显示 SVG）
   用 onload 而不是 fetch 成功就置位，是因为"拿到文本"不等于"能渲染"。 */
function loadFinalSvg(cb){
  if(_finalSvgUrl||_finalSvgLoading) return;
  _finalSvgLoading=true;
  _finalSvgErr=null;
  updateFinalBadge('loading');
  fetch('/final.svg').then(r=>{
    if(!r.ok) throw new Error('HTTP '+r.status);
    return r.text();
  }).then(txt=>{
    const url='data:image/svg+xml;charset=utf-8,'+encodeURIComponent(txt);
    const probe=new Image();          // 预加载，等解码完成
    probe.onload=()=>{
      _finalSvgUrl=url;               // 此刻才真正可显示
      _finalSvgLoading=false;
      /* 预热：立刻插入 #finalLayer（此时它是 opacity:0、用户不可见），
         浏览器会在后台完成这一次巨大 SVG 的光栅化。
         之后切到 final-1 / final 只需要把 opacity 改成 1，不再有重排/重绘。 */
      try{
        const fl=$('finalLayer');
        if(fl && !fl.querySelector('img')){
          const im=document.createElement('img');
          im.src=url;
          im.setAttribute('draggable','false');
          fl.replaceChildren(im);
        }
      }catch(e){}
      updateFinalBadge('svg');
      if(cb) cb();                    // 立刻重绘（无 cb 时仅静默就绪）
    };
    probe.onerror=()=>{
      _finalSvgLoading=false;
      _finalSvgErr='SVG 解码失败';
      updateFinalBadge('error');
      if(cb) cb();
    };
    probe.src=url;
  }).catch(e=>{
    _finalSvgLoading=false;
    _finalSvgErr=String(e);
    updateFinalBadge('error');
    console.warn('final.svg 加载失败:',e);
    if(cb) cb();
  });
}

/* 在参数面板标注当前渲染源，避免"看到的到底是位图还是矢量"说不清 */
function updateFinalBadge(state){
  const el=$('finalBadge');
  if(!el) return;
  const map={
    loading:['加载矢量 SVG…','var(--mute)'],
    svg:    ['矢量 SVG（已切换）','var(--accent-deep)'],
    error:  ['矢量加载失败，当前为位图预览','var(--orange-deep)'],
  };
  const it=map[state];
  if(!it){ el.style.display='none'; return; }
  el.style.display='block';
  el.innerHTML=`<span style="color:${it[1]};font-weight:700">${it[0]}</span>`;
  if(_finalSvgErr) el.innerHTML+=`<span style="color:var(--mute)"> (${esc(_finalSvgErr)})</span>`;
}
let _zScale=1, _zX=0, _zY=0;      // 当前缩放与平移
let _drag=null;                    // 拖拽状态

function applyZoom(){
  const z=$('zoom');
  z.style.transform =
    `translate(${_zX}px,${_zY}px) scale(${_zScale})`;
  $('zVal').textContent=_zScale.toFixed(2)+'x';
}
function resetZoom(){
  _zScale=1;_zX=0;_zY=0;_drag=null;
  const z=$('zoom');
  if(z){ z.style.transform=''; }
  const v=$('zVal'); if(v) v.textContent='1.00x';
}
/* 以 #stage 中心为基准限制平移，避免把图拖到完全看不见 */
function clampPan(){
  const st=$('stage'), z=$('zoom');
  if(!st||!z) return;
  const img=z.querySelector('img');
  if(!img) return;
  const W=st.clientWidth, H=st.clientHeight;
  const iw=img.clientWidth*_zScale, ih=img.clientHeight*_zScale;
  const mx=Math.max(0,(iw-W)/2)+ (iw<W ? (W-iw)/2 : 0);
  const my=Math.max(0,(ih-H)/2)+ (ih<H ? (H-ih)/2 : 0);
  _zX=Math.max(-mx,Math.min(mx,_zX));
  _zY=Math.max(-my,Math.min(my,_zY));
}
/* 以光标为锚点缩放 */
function zoomAt(cx,cy,factor){
  const st=$('stage');
  if(!st) return;
  const ns=Math.max(1,Math.min(8,_zScale*factor));
  if(Math.abs(ns-_zScale)<1e-6) return;
  const r=st.getBoundingClientRect();
  const px=cx-(r.left+r.width/2), py=cy-(r.top+r.height/2);
  const k=ns/_zScale;
  _zX=px-(px-_zX)*k;
  _zY=py-(py-_zY)*k;
  _zScale=ns;
  clampPan(); applyZoom();
}

(function bindZoom(){
  const st=$('stage');
  if(!st) return;

  /* 滚轮：以光标为锚点缩放 */
  st.addEventListener('wheel',e=>{
    if(!_zoomInteractive) return;
    e.preventDefault();
    zoomAt(e.clientX,e.clientY, e.deltaY<0 ? 1.16 : 1/1.16);
  },{passive:false});

  /* 拖动平移 */
  st.addEventListener('pointerdown',e=>{
    if(!_zoomInteractive) return;
    if(e.button!==0) return;
    _drag={x:e.clientX,y:e.clientY,ox:_zX,oy:_zY};
    $('zoom').classList.add('grabbing');
    st.setPointerCapture(e.pointerId);
  });
  st.addEventListener('pointermove',e=>{
    if(!_drag) return;
    _zX=_drag.ox+(e.clientX-_drag.x);
    _zY=_drag.oy+(e.clientY-_drag.y);
    clampPan(); applyZoom();
  });
  const endDrag=e=>{
    if(!_drag) return;
    _drag=null;
    $('zoom').classList.remove('grabbing');
  };
  st.addEventListener('pointerup',endDrag);
  st.addEventListener('pointercancel',endDrag);

  /* 双击复位 */
  st.addEventListener('dblclick',e=>{
    if(!_zoomInteractive) return;
    e.preventDefault();
    resetZoom();
  });

  /* 缩放条按钮 */
  const zb=$('zoombar');
  if(zb){
    zb.addEventListener('pointerdown',e=>e.stopPropagation());
    $('zIn').onclick=()=>{
      if(!_zoomInteractive) return;
      const r=st.getBoundingClientRect();
      zoomAt(r.left+r.width/2,r.top+r.height/2,1.25);
    };
    $('zOut').onclick=()=>{
      if(!_zoomInteractive) return;
      const r=st.getBoundingClientRect();
      zoomAt(r.left+r.width/2,r.top+r.height/2,1/1.25);
    };
    $('zReset').onclick=()=>resetZoom();
  }
})();

/* 窗口尺寸变化后重新约束，避免图被推出可视区 */
window.addEventListener('resize',()=>{
  if(_zoomInteractive){ clampPan(); applyZoom(); }
});

requestAnimationFrame(tickElapsed);

initSpeed();   // 放在所有声明之后调用，避免 TDZ

requestAnimationFrame(tickElapsed);
poll();
/* ---- 心跳与关闭释放（v1.13）---- */
/* 每个标签页一个唯一 sid：多标签时 B 关闭不会连坐 A（服务端按 sid 分别记账） */
var SID = (function(){
  try{
    var v = sessionStorage.getItem('vc_sid');
    if(!v){
      v = Math.random().toString(36).slice(2) + Date.now().toString(36);
      sessionStorage.setItem('vc_sid', v);
    }
    return v;
  }catch(e){
    return Math.random().toString(36).slice(2) + Date.now().toString(36);
  }
})();
/* 每 2 秒给服务端一个心跳；服务端据此判断页面是否还在 */
(function(){
  function beat(){
    try{
      fetch('/beat?sid='+SID, {cache:'no-store'}).catch(function(){});
    }catch(e){}
  }
  beat();
  setInterval(beat, 2000);

  /* 页面卸载时发信标。sendBeacon 在卸载后仍会发出，
     普通 fetch 会被浏览器取消，所以这里优先用 sendBeacon。 */
  function bye(){
    try{
      if(navigator.sendBeacon){
        navigator.sendBeacon('/bye?sid='+SID);
      }else{
        fetch('/bye?sid='+SID, {method:'POST', keepalive:true}).catch(function(){});
      }
    }catch(e){}
  }
  window.addEventListener('pagehide', bye);
  window.addEventListener('beforeunload', bye);
})();

</script></body></html>
"""


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        u = urlparse(self.path)
        qs = parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if u.path in ("/final.svg", "/final_svg"):
            # 按需返回最终 SVG 原文（24MB 级，不走 frames 轮询）
            with LOCK:
                fp = STATE.get("final_svg_path")
            if not fp or not os.path.exists(fp):
                self.send_response(404)
                self.end_headers()
                return
            with open(fp, "rb") as fh:
                body = fh.read()
            self.send_response(200)
            self.send_header("Content-Type",
                             "image/svg+xml; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if u.path.startswith("/frame/"):
            # 单帧图片：内容生成后不再变化 -> immutable 强缓存
            name = u.path[len("/frame/"):]
            fid = name.split(".")[0]
            try:
                fid_int = int(fid)
            except ValueError:
                self.send_response(400)
                self.end_headers()
                return
            with LOCK:
                item = STATE.get("imgs", {}).get(fid_int)
            if not item:
                self.send_response(404)
                self.end_headers()
                return
            ctype, blob = item
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(blob)))
            self.send_header("Cache-Control",
                             "public, max-age=31536000, immutable")
            self.end_headers()
            self.wfile.write(blob)
            return
        if u.path == "/beat":
            # 页面心跳（session 级，共用 _record_beat）
            _record_beat((qs.get("sid") or [""])[0])
            body = json.dumps({'ok': True}).encode()
            self.send_response(200)
            self.send_header('Content-Type',
                             'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if u.path == "/bye":
            # 页面卸载信标（session 级，共用 _record_bye）
            _record_bye((qs.get("sid") or [""])[0])
            body = json.dumps({'ok': True}).encode()
            self.send_response(200)
            self.send_header('Content-Type',
                             'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if u.path == "/state":
            qs = parse_qs(u.query)
            since = 0
            if "since" in qs:
                try:
                    since = int(qs["since"][0])
                except ValueError:
                    since = 0
            with LOCK:
                all_frames = STATE["frames"]
                # 增量：只回 since 之后的新帧（帧不可变，旧帧无需重传）
                frames_out = [f for f in all_frames
                              if int(f.get("id", 0)) > since] if since else all_frames
                payload = {
                    "phase": STATE["phase"],
                    "phase_short": STATE["phase_text"].split("/")[0].strip(),
                    "done": STATE["done"],
                    "error": STATE["error"],
                    "elapsed": round(time.time() - STATE["started_at"], 2),
                    "frames": frames_out,
                    "frame_total": len(all_frames),
                    "logs": STATE["logs"][-200:],
                    "params": STATE["params"],
                    "layer_done": STATE.get("layer_done", 0),
                    "layer_total": STATE.get("layer_total", 0),
                }
            b = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        """sendBeacon 用 POST 发 /bye（也接受 /beat）。

        必须与 do_GET 走同一套 session 逻辑：真实浏览器关闭页面时
        走的就是 sendBeacon(POST)，只修 do_GET 等于没修。
        """
        u2 = urlparse(self.path)
        qs2 = parse_qs(u2.query)
        sid = (qs2.get("sid") or [""])[0]
        if u2.path == "/beat":
            _record_beat(sid)
        elif u2.path == "/bye":
            _record_bye(sid)
        else:
            self.send_response(501)
            self.end_headers()
            return
        body = json.dumps({"ok": True}).encode()
        self.send_response(200)
        self.send_header("Content-Type",
                         "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)



def _record_beat(sid):
    """记一次心跳（session 级），并取消待定的退出。"""
    with _RL:
        now = time.monotonic()
        LAST_BEAT[0] = now
        EVER_BEAT[0] = True
        SESSIONS[sid or "default"] = now
        BYE_AT[0] = 0.0


def _record_bye(sid):
    """记一次页面卸载（session 级）。只有最后一个 session 关闭才置 BYE_AT，
    且不直接置 EXIT_REQ——交给看门狗在 REFRESH_GRACE 后判定（容忍刷新）。"""
    with _RL:
        now = time.monotonic()
        SESSIONS.pop(sid or "default", None)
        # 清僵尸 session（超过 90s 无心跳）
        for kk in [k2 for k2, t2 in SESSIONS.items() if now - t2 > 90]:
            SESSIONS.pop(kk, None)
        if not SESSIONS:
            BYE_AT[0] = now


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True




def serve(port, tries=20):
    for i in range(tries):
        try:
            srv = Server(("127.0.0.1", port + i), Handler)
            return srv, port + i
        except OSError:
            continue
    raise RuntimeError("无可用端口")


def _watchdog(srv, idle, start_grace, out_path, t_start):
    """独立线程：页面关闭 / 心跳超时 / 启动宽限 任一满足即结束进程。

    必须与绘制并行运行 —— 否则 w.join() 会阻塞到绘制结束（30-60s），
    期间任何超时判定都不会被检查（这是 v1.13 初版失败的原因）。

    用 os._exit 而非 return：绘制线程是 daemon，可能正卡在 vec.py
    子进程调用里，优雅退出会被它拖住；本题要求"关闭网页立刻释放"。
    """
    while True:
        time.sleep(0.5)
        now = time.monotonic()
        reason = None

        # 1) 页面卸载信标（最快）。--keepalive 时忽略（用户显式要求常驻）
        with _RL:
            # 只取 BYE_AT；是否真退出由下方「刷新宽限」判定
            # （这样 F5 刷新后的新心跳不会被误杀）
            bye_at = BYE_AT[0]
        if (reason is None and bye_at
                and time.monotonic() - bye_at > REFRESH_GRACE
                and not KEEPALIVE[0]):
            reason = "页面已关闭（刷新宽限后）"

        # 2) 心跳超时（仅在曾收到过心跳时才生效）
        if reason is None and idle > 0:
            with _RL:
                ever = EVER_BEAT[0]
                last = LAST_BEAT[0]
            if ever:
                # 阈值放宽到 3 倍：浏览器会把后台标签的 setInterval
                # 节流到约 1 次/分钟，按 idle(=30s) 判定会误杀仍在看的页面。
                eff = idle * 3 if idle >= 10 else idle
                if now - last > eff:
                    reason = (f"心跳超时 {idle:.0f}s"
                              f"（最后心跳 {now - last:.0f}s 前）")
            else:
                pass    # 启动宽限改到外层独立判定（见下方）

        # 启动宽限：独立判定，不受 --idle-exit 影响
        # （旧版把它嵌在 idle>0 里，导致 --idle-exit 0 时永不退出）
        #
        # 必须同时排除 --keepalive：webui 用 --no-open 启动 watch_draw，
        # 永远不会收到 /beat，EVER_BEAT 恒为 False。若不排除 keepalive，
        # start_grace（默认 60s）就成了绘制的硬上限 —— 而 tier2 需要约
        # 495 秒、tier4/k=4 需要 200+ 秒，全部会被提前杀死且不产出文件。
        if reason is None and start_grace > 0:
            with _RL:
                _ever2 = EVER_BEAT[0]
                _keep2 = KEEPALIVE[0]
            if not _ever2 and not _keep2 and now - t_start > start_grace:
                reason = f"启动后 {start_grace:.0f}s 无人访问"

        if reason:
            try:
                srv.shutdown()
            except Exception:
                pass
            try:
                srv.server_close()
            except Exception:
                pass
            # 清理本次可能残留的放大临时图（os._exit 不做 finally）
            try:
                od = os.path.dirname(os.path.abspath(out_path)) if out_path else ""
                if od and os.path.isdir(od):
                    base2 = os.path.splitext(
                        os.path.basename(out_path))[0] if out_path else ""
                    for fn2 in os.listdir(od):
                        if fn2.startswith("." + base2 + "_k") and \
                                fn2.endswith("_src.png"):
                            try:
                                os.remove(os.path.join(od, fn2))
                            except OSError:
                                pass
            except Exception:
                pass
            # 终止仍在跑的子进程，避免它们变成孤儿继续写盘。
            # VEC_PROC 在 run_pipeline / run_tier2 / run_tier3 三处都会登记，
            # 所以 tier1/2/3/4 全覆盖（之前只登记 vec.py，tier2/3 的子进程漏杀）。
            try:
                with _RL:
                    _vp2 = VEC_PROC[0]
                if _vp2 is not None and _vp2.poll() is None:
                    if os.name == "nt":
                        subprocess.run(
                            ["taskkill", "/T", "/F", "/PID", str(_vp2.pid)],
                            capture_output=True, timeout=15)
                    else:
                        _vp2.terminate()
                    try:
                        _vp2.wait(timeout=3)
                    except Exception:
                        _vp2.kill()
            except Exception:
                pass

            # 清理中间产物。os._exit 会跳过所有 finally，所以必须在这里显式做：
            # 被 start-grace / 心跳超时杀死时任务没跑完，残留最大（如
            # _t2_base.svg 3.7MB）。清理失败不能阻止退出，全部包在 try 里。
            try:
                _od = os.path.dirname(os.path.abspath(out_path)) \
                    if out_path else ""
                _base = os.path.splitext(
                    os.path.basename(out_path))[0] if out_path else ""
                # 放大临时图（.xxx_k?_src.png）
                if _od and _base and os.path.isdir(_od):
                    for _fn in os.listdir(_od):
                        if _fn.startswith("." + _base + "_k") and \
                                _fn.endswith("_src.png"):
                            try:
                                os.remove(os.path.join(_od, _fn))
                            except OSError:
                                pass
                # tier2/tier3 中间产物
                _cleanup_mid(_od,
                             ["_t2_base", "_t3_base", "_qbase", "_t2b", "_t3b",
                              "_lr_b", "_lr_r", "_lhb"],
                             ["_t2_mq"],
                             keep=["_t2_mq/max_final.png",
                                   "_t2_mq/max_result.json"])
            except Exception:
                pass
            print(f"\n  退出：{reason}", flush=True)
            try:
                if out_path and os.path.exists(out_path):
                    print(f"  最终 SVG：{out_path}", flush=True)
            except Exception:
                pass
            sys.stdout.flush()
            os._exit(0)


def main():
    ap = argparse.ArgumentParser(
        description="SVG 复刻全过程可视化（莱茵生命风格）")
    ap.add_argument("src", help="源图路径")
    ap.add_argument("--out", default="", help="最终 SVG 输出路径")
    ap.add_argument("--colors", type=int, default=56, help="色数 N")
    ap.add_argument("--edge", type=float, default=0.16, help="EDGE_KEEP")
    ap.add_argument("--k", type=int, default=1,
                    help="上采样倍数。档位4 的用途就是让**调用方（大模型）**"
                         "看过图后自己定这个值；档位1 固定 k=1。"
                         "非 agent 用户可在 webui 控制台的「放大 k」下拉框里选")
    ap.add_argument("--tier", type=int, default=1,
                    help="档位：1快速(k=1) 2高质量 3局部重绘 4上采样(由模型定 k)")
    ap.add_argument("--w", type=int, default=1800,
                    help="高质量档超采样宽度（最优 1800）")
    ap.add_argument("--blocks", type=int, default=20,
                    help="高质量档 / 局部重绘档 重绘块数")
    ap.add_argument("--block", type=int, default=96,
                    help="高质量档 / 局部重绘档 块尺寸")
    ap.add_argument("--regions", default="",
                    help='档位3 语义区域 "x0,y0,x1,y1;..."（省略则自动选块）')
    ap.add_argument("--up", type=int, default=8, help="块内放大倍数")
    ap.add_argument("--nblk", type=int, default=16, help="块内色数")
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--no-open", action="store_true", help="不自动开浏览器")
    ap.add_argument("--idle-exit", type=float, default=30.0,
                    help="无心跳多少秒后自动退出；0=禁用（默认 30）")
    ap.add_argument("--start-grace", type=float, default=60.0,
                    help="启动宽限秒数，期间不因无心跳退出（默认 60）")
    ap.add_argument("--keepalive", action="store_true",
                    help="强制保持进程（等价于 --idle-exit 0）")
    ap.add_argument("--plan", action="store_true",
                    help="**预演模式**：只读图并打印 k=1..8 的完整档位表，"
                         "不启动服务、不跑管线。档位4 要由你定 k 时，"
                         "先跑这个看完再定，不用试错")
    args = ap.parse_args()

    src = os.path.abspath(args.src)
    if not os.path.exists(src):
        print(f"错误：源图不存在 {src}")
        return 1

    # --plan：提前把档位表摆出来，让人/模型先定 k 再跑
    if args.plan:
        try:
            return print_k_plan(src)
        except Exception as e:
            print(f"错误：无法读取源图 {src}：{e}")
            return 1
    out = args.out or os.path.join(
        os.path.dirname(src),
        os.path.splitext(os.path.basename(src))[0] + "_watch.svg")

    srv, port = serve(args.port)
    url = f"http://127.0.0.1:{port}/"
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    log(f"服务已启动 {url}")
    log(f"档位 = {args.tier}（1快速 2高质量 3局部重绘 4上采样）")
    print(f"\n  可视化地址：{url}\n", flush=True)
    if not args.no_open:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    def work():
        try:
            if args.tier == 2:
                run_tier2(src, out, args.colors, args.edge, args.w,
                          args.blocks, args.block, args.up, args.nblk)
            elif args.tier == 3:
                run_tier3(src, out, args.colors, args.edge, args.blocks,
                          args.block, args.up, args.nblk, args.regions)
            else:
                run_pipeline(src, out, args.colors, args.edge, args.k)
        except Exception as e:
            import traceback
            traceback.print_exc()
            with LOCK:
                STATE["error"] = str(e)
                STATE["done"] = True
            log(f"异常：{e}", "error")

    w = threading.Thread(target=work, daemon=True)
    w.start()

    # ---- 看门狗独立线程，与绘制并行（关键：不能等 w.join）----
    _idle = 0.0 if args.keepalive else max(0.0, args.idle_exit)
    with _RL:
        KEEPALIVE[0] = bool(args.keepalive)
    print(f"  页面关闭或空闲 {_idle:.0f}s 后自动退出"
          f"（--idle-exit 0 可禁用）\n", flush=True)
    _t_start = time.monotonic()
    threading.Thread(
        target=_watchdog,
        args=(srv, _idle, args.start_grace, out, _t_start),
        daemon=True, name="watchdog").start()

    # 主线程等绘制结束（看门狗若先触发会直接 os._exit）
    try:
        w.join()
    except KeyboardInterrupt:
        pass


    # 只有真的产出了目标文件才打印交付路径；tier2/tier3 的交付是
    # 像素域结果（PNG），无条件打印「最终 SVG」会误导用户。
    # 判定必须用 OUT_WRITTEN（本次是否真的写成功），不能用 os.path.exists ——
    # 目录里可能残留上一次的旧文件，本次写盘失败时它仍在，
    # 只看存在性会把失败报成成功。
    if OUT_WRITTEN[0] and FINAL_INFO[0] and os.path.exists(FINAL_INFO[0]):
        print(f"\n  最终产物：{FINAL_INFO[0]}  ({FINAL_INFO[1]})", flush=True)
    elif OUT_WRITTEN[0] and os.path.exists(out):
        print(f"\n  最终 SVG：{out}", flush=True)
    else:
        print(f"\n  未产出目标文件：{out}", flush=True)
        print("  失败原因见上方日志中的 error / vec.py 输出", flush=True)
    print(f"  可视化页面保持可用：{url}", flush=True)
    print(f"  关闭页面或空闲 {_idle:.0f}s 后自动退出\n", flush=True)

    # 绘制已完成，但页面仍需保持可查看 —— 交给看门狗负责退出：
    #   页面关闭（/bye） / 心跳超时 / 启动宽限
    # 主线程在此阻塞等待，不要直接 return（否则画完就关，用户来不及看）。
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n  退出：Ctrl+C", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
