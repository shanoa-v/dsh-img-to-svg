"""
live_vec_server.py — 实时观看 SVG 矢量化过程的浏览器预览服务（支持任意数量图片）

用法:
  # 单张
  python live_vec_server.py photo.png

  # 多张（任意数量，依次自动渲染）
  python live_vec_server.py a.png b.png c.png d.png

  # 整个目录（递归收集所有图片）
  python live_vec_server.py ./images --outdir ./out

  # 通配符（Windows 下建议加引号，或让 shell 展开）
  python live_vec_server.py "shots/*.png"

工作方式:
  1. 启动本地 HTTP 服务，并自动打开浏览器标签页
  2. 后台线程按队列**依次**处理每张图，每完成一层就把"到当前层为止的 SVG"写入内存
  3. 页面每 120ms 轮询 /state；带上版本号，版本未变时服务端不回传 SVG
     （关键优化：避免每 120ms 重传数 MB 的 SVG 字符串）
  4. 页面显示：图片队列 | 原图 | 矢量重建（实时逐层浮现）| 状态与日志
  5. 点队列里任意一张可切换查看；已完成的可回看其 SVG

参数:
  --port 8765        端口（被占用时自动 +1 重试，最多 20 次）
  --outdir DIR       所有输出 SVG 放到该目录（默认为各自源图同目录）
  --no-browser       不自动打开浏览器
  --layer-delay N    层间暂停秒数；默认自适应：单图 0.35，多图 0.12
  --colors/--edge/--eps/--minarea/--width   同 vec.py（也读 VEC_* 环境变量）
"""
import os, sys, io, json, time, glob, base64, threading, webbrowser, argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter, label as ndlabel, generic_filter
from skimage import measure, color as skcolor
from sklearn.cluster import MiniBatchKMeans

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff",
           ".gif", ".avif", ".heic", ".jfif"}


# ============ 参数 ============
ap = argparse.ArgumentParser()
ap.add_argument("srcs", nargs="+", help="一张或多张源图，也可以是目录")
ap.add_argument("--outdir", default=None, help="输出目录（默认与源图同目录）")
ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--no-browser", action="store_true")
ap.add_argument("--colors", type=int, default=int(os.environ.get("VEC_N", "56")))
ap.add_argument("--edge", type=float, default=float(os.environ.get("VEC_EDGE", "0.16")))
ap.add_argument("--eps", type=float, default=float(os.environ.get("VEC_EPS", "0.30")))
ap.add_argument("--minarea", type=float, default=float(os.environ.get("VEC_MINAREA", "5")))
ap.add_argument("--width", type=int, default=int(os.environ.get("VEC_W", "0")))
ap.add_argument("--layer-delay", type=float, default=None,
                help="每层之间暂停秒数；不填则自适应（单图 0.35，多图 0.12）")
args = ap.parse_args()


# ============ 收集源图（支持目录 / 通配符）============
def collect_sources(paths):
    out, seen = [], set()

    def add(p):
        rp = os.path.abspath(p)
        if rp not in seen:
            seen.add(rp)
            out.append(rp)

    for p in paths:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for f in sorted(files):
                    if os.path.splitext(f)[1].lower() in IMG_EXT:
                        add(os.path.join(root, f))
        elif os.path.isfile(p):
            add(p)
        else:
            for m in sorted(glob.glob(p, recursive=True)):
                if os.path.isfile(m) and os.path.splitext(m)[1].lower() in IMG_EXT:
                    add(m)
    return out


SOURCES = collect_sources(args.srcs)
if not SOURCES:
    print("错误：没有找到任何图片。传入图片路径、目录或通配符均可。")
    sys.exit(1)

# 生成输出路径（重名自动加序号）
OUTDIR = os.path.abspath(args.outdir) if args.outdir else None
if OUTDIR:
    os.makedirs(OUTDIR, exist_ok=True)

JOBS = []
used = set()
for i, s in enumerate(SOURCES):
    d = OUTDIR or os.path.dirname(s)
    stem = os.path.splitext(os.path.basename(s))[0]
    out = os.path.join(d, f"{stem}_vector.svg")
    k = 2
    while out.lower() in used:
        out = os.path.join(d, f"{stem}_vector_{k}.svg")
        k += 1
    used.add(out.lower())
    JOBS.append({
        "name": os.path.basename(s), "src": s, "out": out,
        "status": "pending", "layer": 0, "total_layers": 0,
        "similarity": None, "elapsed": 0.0, "error": None, "size": [0, 0],
    })

MULTI = len(JOBS) > 1
LAYER_DELAY = args.layer_delay if args.layer_delay is not None else (0.12 if MULTI else 0.35)
PORT = args.port


# ============ 全局状态（后台线程写，HTTP 线程读）============
STATE = {
    "phase": "init",
    "index": 0,
    "total_jobs": len(JOBS),
    "jobs": JOBS,
    "svg": "",
    "log": [],
    "src_size": [0, 0],
    "palette": "",
    "elapsed": 0.0,
    "similarity": None,
    "output": JOBS[0]["out"] if JOBS else "",
    "error": None,
    "version": 0,
}
LOCK = threading.Lock()


def log(msg):
    with LOCK:
        STATE["log"].append(f"[{time.strftime('%H:%M:%S')}] {msg}")
    bump()


def bump():
    with LOCK:
        STATE["version"] += 1


def set_job(i, **kw):
    with LOCK:
        STATE["jobs"][i].update(kw)


def set_state(**kw):
    with LOCK:
        STATE.update(kw)


# ============ 矢量化核心 ============
def rdp(pts, eps):
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    if n < 3:
        return pts
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 <= i0 + 1:
            continue
        p0, p1 = pts[i0], pts[i1]
        seg = p1 - p0
        L = np.hypot(seg[0], seg[1])
        if L < 1e-9:
            dd = np.hypot(pts[i0 + 1:i1, 0] - p0[0], pts[i0 + 1:i1, 1] - p0[1])
        else:
            nx, ny = -seg[1] / L, seg[0] / L
            dd = np.abs((pts[i0 + 1:i1, 0] - p0[0]) * nx
                        + (pts[i0 + 1:i1, 1] - p0[1]) * ny)
        k = int(np.argmax(dd))
        if dd[k] > eps:
            m = i0 + 1 + k
            keep[m] = True
            stack.append((i0, m))
            stack.append((m, i1))
    return pts[keep]


def poly_to_d(poly):
    if len(poly) < 3:
        return ""
    n = len(poly)
    if n >= 4:
        o = [f"M{(poly[0][0] + poly[-1][0]) / 2:.1f} {(poly[0][1] + poly[-1][1]) / 2:.1f}"]
        for i in range(n):
            p = poly[i]
            q = poly[(i + 1) % n]
            o.append(f"Q{p[0]:.1f} {p[1]:.1f} {(p[0] + q[0]) / 2:.1f} {(p[1] + q[1]) / 2:.1f}")
        o.append("Z")
        return "".join(o)
    o = [f"M{poly[0][0]:.1f} {poly[0][1]:.1f}"]
    for p in poly[1:]:
        o.append(f"L{p[0]:.1f} {p[1]:.1f}")
    o.append("Z")
    return "".join(o)


def build_svg(size, base_hex, layers):
    W, H = size
    P = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'width="100%" preserveAspectRatio="xMidYMid meet">',
        f'<rect width="{W}" height="{H}" fill="{base_hex}"/>',
        '<g stroke="none">',
    ]
    for L in layers:
        P.append(f'<path fill="{L["hexc"]}" d="{L["d"]}"/>')
    P.append("</g></svg>")
    return "\n".join(P)


def vectorize_one(i):
    """处理第 i 张图（完整流程），返回是否成功"""
    job = JOBS[i]
    SRC, OUT = job["src"], job["out"]
    t0 = time.time()
    set_state(phase="loading", svg="", layer=0, total_layers=0,
              palette="", similarity=None, error=None,
              output=OUT, index=i, elapsed=0.0)
    set_job(i, status="running", layer=0, total_layers=0, similarity=None, error=None)
    log(f"[{i + 1}/{len(JOBS)}] 读取 {os.path.basename(SRC)}")
    bump()

    im = Image.open(SRC)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im).convert("RGB")
    else:
        im = im.convert("RGB")
    W0, H0 = im.size
    WORK_W = args.width or W0
    WORK_W = min(WORK_W, W0 * 2)
    WORK_H = round(WORK_W * H0 / W0)
    set_state(src_size=[W0, H0])
    set_job(i, size=[W0, H0])
    log(f"    尺寸 {W0}x{H0}，工作分辨率 {WORK_W}x{WORK_H}")

    # --- 保边平滑 ---
    set_state(phase="smooth")
    ws = im.resize((WORK_W, WORK_H), Image.LANCZOS)
    med = ws.filter(ImageFilter.MedianFilter(3)).filter(ImageFilter.MedianFilter(3))
    sm = np.asarray(med).astype(np.float32)
    orig = np.asarray(ws).astype(np.float32)
    gray = orig.mean(axis=2)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    grad = np.hypot(gx, gy)
    gmax = np.percentile(grad, 99) + 1e-6
    wk = np.clip(np.clip(grad / gmax, 0, 1) * args.edge * 3, 0, 1)[..., None]
    edge_sm = sm * (1 - wk) + orig * wk
    log(f"    保边平滑完成 (edge={args.edge})")
    bump()

    # --- 量化 ---
    set_state(phase="quantize")
    log(f"    LAB 空间 KMeans 量化中（{args.colors} 色）…")
    bump()
    lab = skcolor.rgb2lab(edge_sm / 255.0).reshape(-1, 3)
    km = MiniBatchKMeans(n_clusters=args.colors, random_state=42,
                         batch_size=16384, n_init=4, max_iter=150)
    idx = km.fit_predict(lab).reshape(WORK_H, WORK_W).astype(np.int32)
    cen = (skcolor.lab2rgb(km.cluster_centers_.reshape(1, -1, 3))[0] * 255).clip(0, 255)
    cnt = np.bincount(idx.ravel(), minlength=args.colors)
    log(f"    量化完成 {args.colors} 色")

    # --- 碎斑清理 ---
    small = np.zeros_like(idx, dtype=bool)
    for ci in range(args.colors):
        lb, n = ndlabel(idx == ci)
        if n == 0:
            continue
        sizes = np.bincount(lb.ravel())
        small |= np.isin(lb, np.nonzero((sizes < 24) & (np.arange(len(sizes)) > 0))[0])
    if small.any():
        filled = generic_filter(idx.astype(np.float32),
                                lambda v: np.bincount(v.astype(np.int32)).argmax(), size=3)
        idx = np.where(small, filled.astype(np.int32), idx)
    log(f"    碎斑清理 {int(small.sum())} px")

    # --- 逐层描摹（用户实时观看的部分）---
    set_state(phase="trace")
    order = list(np.argsort(-cnt))
    base = cen[np.argmax(cnt)].astype(int)
    base_hex = f"#{base[0]:02X}{base[1]:02X}{base[2]:02X}"
    real_total = sum(1 for ci in order if int((idx == ci).sum()) >= args.minarea)
    set_state(total_layers=real_total)
    set_job(i, total_layers=real_total)
    log(f"    开始描摹，共 {real_total} 层")
    bump()

    layers = []
    drawn = 0
    for ci in order:
        mask = (idx == ci)
        npx = int(mask.sum())
        if npx < args.minarea:
            continue
        frac = npx / idx.size
        eps = args.eps * (1.0 if frac < 0.02 else (1.3 if frac < 0.10 else 1.8))
        mb = gaussian_filter(mask.astype(np.float32), sigma=0.4)
        polys = []
        for c in measure.find_contours(mb, level=0.5):
            if len(c) < 4:
                continue
            if np.allclose(c[0], c[-1]):
                c = c[:-1]
            if len(c) < 3:
                continue
            pl = rdp(c, eps)
            if len(pl) < 3:
                continue
            x = pl[:, 1]
            y = pl[:, 0]
            if 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) < args.minarea:
                continue
            polys.append(pl[:, ::-1])  # (row,col) -> (x,y)
        if not polys:
            continue
        c = cen[ci].astype(int)
        hexc = f"#{c[0]:02X}{c[1]:02X}{c[2]:02X}"
        layers.append(dict(hexc=hexc, d=" ".join(poly_to_d(p) for p in polys)))
        drawn += 1
        set_state(svg=build_svg((W0, H0), base_hex, layers),
                  layer=drawn, palette=hexc, elapsed=time.time() - t0)
        set_job(i, layer=drawn)
        bump()
        if LAYER_DELAY:
            time.sleep(LAYER_DELAY)

    # --- 写最终 SVG（带标准头与缩放变换）---
    sx, sy = W0 / WORK_W, H0 / WORK_H
    P = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W0} {H0}" '
        f'width="{W0}" height="{H0}" shape-rendering="geometricPrecision">',
        f'<title>{os.path.basename(SRC)}</title>',
        '<desc>Layered vector reconstruction</desc>',
        f'<rect width="{W0}" height="{H0}" fill="{base_hex}"/>',
        f'<g transform="scale({sx:.6f} {sy:.6f})" stroke="none">',
    ]
    for L in layers:
        P.append(f'<path fill="{L["hexc"]}" d="{L["d"]}"/>')
    P.append("</g></svg>")
    final = "\n".join(P)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(final)
    set_state(svg=final, elapsed=time.time() - t0)
    log(f"    完成！{len(layers)} 层，{len(final) / 1024:.1f} KB")

    # --- 自动评估相似度 ---
    sim = None
    try:
        import cairosvg
        png = cairosvg.svg2png(url=OUT, output_width=W0, output_height=H0)
        svg_img = Image.open(io.BytesIO(png)).convert("RGB")
        CW = 960
        CH = max(1, round(CW * H0 / W0))
        a = np.asarray(im.resize((CW, CH), Image.LANCZOS)).astype(np.float32)
        b = np.asarray(svg_img.resize((CW, CH), Image.LANCZOS)).astype(np.float32)
        mae = float(np.abs(a - b).mean())
        sim = (1 - mae / 255) * 100
        set_state(similarity={"mae": mae, "similarity": sim})
        set_job(i, similarity=sim)
        log(f"    相似度 {sim:.2f}% (MAE {mae:.2f})")
    except Exception as e:
        log(f"    相似度评估跳过：{e}")

    set_job(i, status="done", elapsed=time.time() - t0)
    set_state(phase="done", elapsed=time.time() - t0)
    bump()
    return True


def worker():
    """按队列依次处理所有图片"""
    ok = 0
    for i in range(len(JOBS)):
        try:
            vectorize_one(i)
            ok += 1
        except Exception as e:
            import traceback
            traceback.print_exc()
            set_job(i, status="error", error=str(e))
            set_state(phase="error", error=str(e))
            log(f"    [{i + 1}] 失败：{e}")
    set_state(phase="done" if ok else "error")
    log(f"全部结束：成功 {ok} / {len(JOBS)}")
    bump()


# ============ 图片服务（懒加载 + 缓存）============
_SRC_CACHE, _THUMB_CACHE = {}, {}


def _jpeg(i, maxdim):
    cache = _SRC_CACHE if maxdim > 400 else _THUMB_CACHE
    if i in cache:
        return cache[i]
    im = Image.open(JOBS[i]["src"]).convert("RGB")
    W, H = im.size
    s = min(1.0, maxdim / max(W, H))
    if s < 1.0:
        im = im.resize((max(1, int(W * s)), max(1, int(H * s))), Image.LANCZOS)
    b = io.BytesIO()
    im.save(b, "JPEG", quality=82 if maxdim > 400 else 72)
    data = b.getvalue()
    cache[i] = data
    return data


PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>SVG 矢量化实时预览</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,"Microsoft YaHei",sans-serif;
     background:#0f1117;color:#e6e8ef;height:100vh;display:flex;flex-direction:column;overflow:hidden}
header{background:#161923;border-bottom:1px solid #262a38;padding:10px 16px;
       display:flex;align-items:center;gap:14px;flex:0 0 auto}
h1{font-size:14px;font-weight:500;white-space:nowrap}
.badge{font-size:11px;padding:3px 9px;border-radius:10px;background:#2a3050;color:#8fa4ff;white-space:nowrap}
.badge.done{background:#1d3a2a;color:#5ddc9a}
.badge.error{background:#3a1d1d;color:#ff8080}
.prog{flex:1;max-width:300px;height:6px;background:#242838;border-radius:3px;overflow:hidden}
.prog i{display:block;height:100%;background:linear-gradient(90deg,#5b7cfa,#8f6cff);width:0;transition:width .25s}
main{flex:1;display:flex;min-height:0}
#queue{flex:0 0 200px;overflow-y:auto;background:#131620;border-right:1px solid #262a38}
.qhead{padding:8px 12px;font-size:11px;color:#7d86a3;background:#141721;
       border-bottom:1px solid #21242f;position:sticky;top:0}
.qitem{display:grid;grid-template-columns:20px 1fr;gap:2px 6px;align-items:center;
       padding:8px 12px;cursor:pointer;border-left:3px solid transparent;font-size:12px}
.qitem:hover{background:#1a1e2b}
.qitem.act{background:#1c2233;border-left-color:#5b7cfa}
.qitem.view{background:#232a42;border-left-color:#8f6cff}
.qn{color:#5f687f;font-size:10.5px;font-variant-numeric:tabular-nums}
.qt{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#cdd4e8}
.qb{grid-column:2;font-size:10px;padding:1px 6px;border-radius:8px;justify-self:start;
    background:#2a3050;color:#8fa4ff}
.qb.done{background:#1d3a2a;color:#5ddc9a}
.qb.error{background:#3a1d1d;color:#ff8080}
.qb.running{background:#3a3050;color:#c9a8ff}
.qm{grid-column:2;font-size:10px;color:#5ddc9a;font-variant-numeric:tabular-nums}
.pane{flex:1;display:flex;flex-direction:column;min-width:0;border-right:1px solid #262a38}
.pane:last-child{border-right:none;flex:0 0 320px}
.plabel{padding:7px 14px;font-size:11px;color:#7d86a3;background:#141721;
        border-bottom:1px solid #21242f;letter-spacing:.4px;display:flex;justify-content:space-between;gap:8px}
.plabel span:last-child{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.stage{flex:1;overflow:auto;padding:14px;display:flex;align-items:center;justify-content:center;background:#0b0d13}
.stage img,.stage svg{max-width:100%;max-height:100%;border-radius:4px;display:block}
#log{flex:0 0 190px;overflow-y:auto;padding:10px 14px;font:11.5px/1.75 ui-monospace,Consolas,monospace;
     color:#9aa3bd;background:#0b0d13;border-top:1px solid #21242f;white-space:pre-wrap}
.metrics{padding:10px 14px;background:#141721;border-top:1px solid #21242f;
         font-size:12px;display:grid;grid-template-columns:auto 1fr;gap:5px 12px}
.metrics b{color:#7d86a3;font-weight:400}
.metrics span{color:#cdd4e8;font-variant-numeric:tabular-nums}
.sw{display:inline-block;width:11px;height:11px;border-radius:2px;vertical-align:-1px;margin-right:5px;
    border:1px solid #ffffff22}
.follow{font-size:10px;padding:2px 8px;border-radius:8px;background:#2a3050;color:#8fa4ff;
        cursor:pointer;border:none;margin-left:6px}
</style></head><body>
<header>
  <h1>SVG 矢量化实时预览</h1>
  <span class="badge" id="phase">初始化</span>
  <span class="badge" id="overall">0 / 0</span>
  <div class="prog"><i id="bar"></i></div>
  <span class="badge" id="cnt">0 / 0 层</span>
</header>
<main>
  <div id="queue">
    <div class="qhead" id="qhead">队列</div>
    <div id="qlist"></div>
  </div>
  <div class="pane">
    <div class="plabel"><span>原图</span><span id="srcname"></span></div>
    <div class="stage"><img id="srcimg" alt="原图"></div>
  </div>
  <div class="pane">
    <div class="plabel"><span>矢量重建（实时）</span><span id="outname"></span></div>
    <div class="stage" id="svgstage"></div>
  </div>
  <div class="pane">
    <div class="plabel"><span>状态</span><button class="follow" id="follow">跟随当前</button></div>
    <div class="metrics">
      <b>阶段</b><span id="m-phase">-</span>
      <b>当前图</b><span id="m-cur">-</span>
      <b>总进度</b><span id="m-overall">-</span>
      <b>层进度</b><span id="m-prog">-</span>
      <b>当前层色</b><span id="m-color">-</span>
      <b>耗时</b><span id="m-time">-</span>
      <b>相似度</b><span id="m-sim">-</span>
      <b>输出</b><span id="m-out">-</span>
    </div>
    <div id="log"></div>
  </div>
</main>
<script>
var PH={init:'初始化',loading:'读取图像',smooth:'保边平滑',quantize:'色彩量化',
        trace:'逐层描摹',done:'完成',error:'出错'};
var ST={pending:'待处理',running:'进行中',done:'完成',error:'失败'};
var viewIndex=-1, lastVer=-1, latest=null, viewedSvg=null, viewedIdx=-1;
function el(id){return document.getElementById(id);}

function renderQueue(s){
  var box=el('qlist'); box.innerHTML='';
  el('qhead').textContent='队列 ('+s.jobs.length+')';
  s.jobs.forEach(function(j,i){
    var d=document.createElement('div');
    d.className='qitem'+(i===s.index?' act':'')+(i===viewIndex?' view':'');
    var n=document.createElement('span'); n.className='qn'; n.textContent=(i+1);
    var t=document.createElement('span'); t.className='qt'; t.title=j.name; t.textContent=j.name;
    var b=document.createElement('span'); b.className='qb '+(j.status||'pending');
    b.textContent=ST[j.status]||j.status||'';
    d.appendChild(n); d.appendChild(t); d.appendChild(b);
    if(j.similarity!==null&&j.similarity!==undefined){
      var m=document.createElement('span'); m.className='qm';
      m.textContent=j.similarity.toFixed(2)+'%'; d.appendChild(m);
    }
    d.onclick=function(){ viewIndex=i; viewedSvg=null; viewedIdx=-1; apply(latest); };
    box.appendChild(d);
  });
}

function setStageSvg(txt){
  if(!txt) return;
  el('svgstage').innerHTML=txt;
}

function apply(s){
  if(!s) return;
  latest=s;
  var cur=s.jobs[s.index]||{};
  var done=s.jobs.filter(function(j){return j.status==='done';}).length;
  var term=s.jobs.filter(function(j){return j.status==='done'||j.status==='error';}).length;

  el('phase').textContent=PH[s.phase]||s.phase;
  el('phase').className='badge'+(s.phase==='done'?' done':s.phase==='error'?' error':'');
  el('overall').textContent=done+' / '+s.jobs.length+' 完成';
  var pct=s.total_layers?Math.round(s.layer/s.total_layers*100):0;
  el('bar').style.width=pct+'%';
  el('cnt').textContent=s.layer+' / '+s.total_layers+' 层';

  renderQueue(s);

  el('m-phase').textContent=PH[s.phase]||s.phase;
  el('m-cur').textContent=(s.index+1)+'/'+s.jobs.length+(cur.name?' '+cur.name:'');
  el('m-overall').textContent=done+'/'+s.jobs.filter(function(j){return j.status!=='pending';}).length+' 已完成，'+term+'/'+s.jobs.length+' 结束';
  el('m-prog').textContent=s.total_layers?s.layer+'/'+s.total_layers+' ('+pct+'%)':'-';
  el('m-color').innerHTML=s.palette?'<i class="sw" style="background:'+s.palette+'"></i>'+s.palette:'-';
  el('m-time').textContent=(s.elapsed||0).toFixed(1)+' s';
  el('m-out').textContent=s.output||'-';

  var vi=viewIndex<0?s.index:viewIndex;
  var vj=s.jobs[vi]||{};
  el('srcname').textContent=vj.name||'';
  el('outname').textContent=(vj.out||'').split(/[\\/]/).pop();
  el('m-sim').textContent=(vj.similarity!==null&&vj.similarity!==undefined)
      ?vj.similarity.toFixed(2)+'%':(vj.status==='done'?'-':'计算中…');

  if(vi!==s.index){
    if(viewedIdx!==vi){
      viewedIdx=vi; viewedSvg=null;
      fetch('/svg/'+vi).then(function(r){return r.ok?r.text():'';}).then(function(t){
        if(t){viewedSvg=t; setStageSvg(t);}
      });
    } else if(viewedSvg){ setStageSvg(viewedSvg); }
  } else if(s.svg){
    setStageSvg(s.svg);
  }

  el('srcimg').src='/src/'+vi;
  el('log').textContent=(s.log||[]).join('\n');
  var lg=el('log'); lg.scrollTop=lg.scrollHeight;
}

el('follow').onclick=function(){ viewIndex=-1; viewedSvg=null; viewedIdx=-1; apply(latest); };

function refresh(){
  fetch('/state?v='+lastVer).then(function(r){return r.json();}).then(function(s){
    lastVer=s.version;
    apply(s);
    var term=s.jobs.length>0&&s.jobs.every(function(j){
      return j.status==='done'||j.status==='error';});
    setTimeout(refresh, term?900:120);
  }).catch(function(){ setTimeout(refresh,600); });
}
refresh();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = urlparse(self.path).path
        q = urlparse(self.path).query

        if p in ("/", "/index.html"):
            self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")

        elif p == "/state":
            # 关键优化：客户端带版本号，版本未变则不回传 SVG（避免每 120ms 传数 MB）
            try:
                cv = int(q.split("v=")[1].split("&")[0]) if "v=" in q else -1
            except Exception:
                cv = -1
            with LOCK:
                s = {
                    "phase": STATE["phase"], "index": STATE["index"],
                    "total_jobs": STATE["total_jobs"],
                    "jobs": [dict(j) for j in STATE["jobs"]],
                    "layer": STATE["layer"], "total_layers": STATE["total_layers"],
                    "svg": STATE["svg"] if cv != STATE["version"] else "",
                    "log": STATE["log"][-400:], "src_size": STATE["src_size"],
                    "palette": STATE["palette"], "elapsed": STATE["elapsed"],
                    "similarity": STATE["similarity"], "output": STATE["output"],
                    "error": STATE["error"], "version": STATE["version"],
                }
            self._send(json.dumps(s).encode("utf-8"), "application/json; charset=utf-8")

        elif p.startswith("/src/"):
            try:
                i = int(p.rsplit("/", 1)[1])
                self._send(_jpeg(i, 1400), "image/jpeg")
            except Exception:
                self.send_error(404)

        elif p.startswith("/thumb/"):
            try:
                i = int(p.rsplit("/", 1)[1])
                self._send(_jpeg(i, 220), "image/jpeg")
            except Exception:
                self.send_error(404)

        elif p == "/svg":
            with LOCK:
                svg = STATE["svg"]
            self._send(svg.encode("utf-8"), "image/svg+xml; charset=utf-8")

        elif p.startswith("/svg/"):
            try:
                i = int(p.rsplit("/", 1)[1])
                with open(JOBS[i]["out"], "r", encoding="utf-8") as f:
                    self._send(f.read().encode("utf-8"), "image/svg+xml; charset=utf-8")
            except Exception:
                self.send_error(404)

        else:
            self.send_error(404)


def serve(port):
    """绑定端口，被占用时自动 +1 重试"""
    # 必须关掉 allow_reuse_address：在 Windows 上 SO_REUSEADDR 会让**两个实例
    # 同时绑定同一端口**都成功，于是下面的 "except OSError -> port+k" 永不触发，
    # 「端口被占用自动 +1 重试」形同虚设（实测两实例可同时 LISTEN 9311）。
    # 设为 False 后，被占用会真的抛 OSError(errno 10048)，重试逻辑才生效。
    ThreadingHTTPServer.allow_reuse_address = False
    for k in range(20):
        try:
            return ThreadingHTTPServer(("127.0.0.1", port + k), Handler), port + k
        except OSError:
            continue
    raise SystemExit(f"端口 {port}~{port + 19} 都被占用，请用 --port 指定其他端口")


if __name__ == "__main__":
    srv, real_port = serve(PORT)
    url = f"http://127.0.0.1:{real_port}/"
    # 必须 flush：不加时把输出重定向到日志文件会**一个字都写不出去**
    # （stdout 被块缓冲，进程长跑不 flush，日志恒为 0 字节）。
    print(f"预览服务已启动: {url}", flush=True)
    print(f"共 {len(JOBS)} 张图，层间延迟 {LAYER_DELAY}s", flush=True)
    for i, j in enumerate(JOBS, 1):
        print(f"  [{i}] {j['name']}  ->  {j['out']}", flush=True)
    print("\n按 Ctrl+C 结束\n", flush=True)

    threading.Thread(target=worker, daemon=True).start()

    if not args.no_browser:
        threading.Timer(0.9, lambda: webbrowser.open(url)).start()

    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止")
