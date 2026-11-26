"""webui.py -- dsh-img-to-svg 网页版图形界面（非 agent 入口）

## 定位

替代 gui.py（tkinter 版）。双击 `启动网页界面.bat` 即启动本地服务并打开浏览器，
用户通过网页选图/选目录、设参数、跑任务。分辨率随浏览器自适应，无高 DPI 问题。

## 两种模式

### 单张（显示绘制过程）
服务端启动 `watch_draw.py` 子进程（它自带完整的逐层可视化页面），
网页拿到端口后**跳转**过去。用户看到的就是一直以来的那套界面：
逐层时间轴、倍速菜单、拖动缩放、final 常驻加载。

### 批量（不展示过程）
服务端逐张跑 `vec.py`，网页轮询 `/api/task` 显示进度、体积、耗时，
完成后写出 `_batch_report.json`。

## 选文件的实现：服务端目录浏览（无上传）

浏览器不能读本地绝对路径，所以由服务端列出目录：

    GET /api/ls?path=<目录>      列出子目录与图片文件
    GET /api/drives              列出盘符

网页里点目录进入、点图片选中。**不传输文件内容**，直接读原图，
输出目录也可指定。

## 接口

    GET  /                       主界面
    GET  /api/drives             盘符列表
    GET  /api/ls?path=...        目录内容
    POST /api/run                启动任务（JSON: mode/src/out/...）
    GET  /api/task               当前任务状态与日志
    POST /api/stop               停止任务

## 风格

沿用莱茵实验室三套配色（READY / VERIFYING / DENIED），
与绘制过程页保持统一。等宽字体、编号面板、细线框。
"""
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler
from socketserver import ThreadingTCPServer
from urllib.parse import parse_qs, urlparse

PY = sys.executable
THIS = os.path.dirname(os.path.abspath(__file__))
TOOLS = THIS
SKILL = os.path.dirname(THIS)
SCRIPTS = os.path.join(SKILL, "scripts")
WATCH = os.path.join(TOOLS, "watch_draw.py")
VEC = os.path.join(SCRIPTS, "vec.py")

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

LOCK = threading.RLock()   # 可重入兜底（lint_locks.py 仍做静态检查）
TASK = {
    "running": False,
    "mode": None,
    "phase": "idle",
    "total": 0,
    "done": 0,
    "current": "",
    "logs": [],
    "results": [],
    "started_at": None,
    "ended_at": None,
    "watch_port": None,
    "error": None,
}
PROC = None
PROC_LIST = []    # 所有活跃 watch_draw 子进程（多张/并发）


def _log_raw(msg):
    """写入日志，**不加锁**。只能在已持 LOCK 的位置调用。

    t 存的是**相对任务开始的秒数**（如 0.0 / 13.7 / 34.2），
    而非 Unix epoch 时间戳——前端直接 toFixed(1) 显示成秒。
    """
    try:
        rel = round(time.time() - _task_t0(), 2)
    except Exception:
        rel = round(time.time() - _MODULE_T0, 2)
    TASK["logs"].append({"t": rel, "m": msg})
    # logs_total 必须是**累计条数**：截断到 600 后 len 恒 600，
    # 前端判据 `logs_total !== lastN` 永远为假 -> 页面在 600 条后彻底停更
    # （审计实测：写 750 条时页面看不到新日志，用户以为卡死）。
    TASK["logs_total"] = TASK.get("logs_total", 0) + 1
    if len(TASK["logs"]) > 600:
        del TASK["logs"][:-600]


_MODULE_T0 = time.time()
# outcome: None（未开始）/ "success" / "cancelled" / "failed"
# 平台审计 H-2：停止、图片损坏、真正成功三种结局原本界面完全一样
# （都显示「单张完成」+100%），加这个字段才能区分。


def _task_t0():
    """任务起点（用于日志的相对秒数）。未开始任务时用模块加载时刻。"""
    try:
        st = TASK.get("started_at")
    except Exception:
        st = None
    return st or _MODULE_T0


def log(msg):
    """写入日志的**外部安全入口**（未持锁时用这个）。"""
    with LOCK:
        _log_raw(msg)


# ======================================================================
# 目录浏览
# ======================================================================
def list_drives():
    out = []
    if os.name == "nt":
        for c in "CDEFGHIJKLMNOPQRSTUVWXYZ":
            p = c + ":\\"
            if os.path.exists(p):
                out.append(p)
    else:
        out = ["/"]
    return out


def _html_escape(t):
    """最小 HTML 转义。

    error 会被前端拼进 innerHTML（`+LAST.error+`），其中含用户可控路径
    -> 反射型 XSS（代码审计实测可用）。path 一并转义做纵深防御。
    """
    if not isinstance(t, str):
        return t
    return (t.replace("&", "&amp;").replace("<", "&lt;")
             .replace(">", "&gt;").replace('"', "&quot;")
             .replace("'", "&#39;"))


def _empty(path, parent, err):
    return {"path": _html_escape(path), "parent": parent,
            "dirs": [], "files": [],
            "drives": [], "error": _html_escape(err)}


def list_dir(path):
    """返回 {path, parent, dirs, files, drives, error}

    对不存在/无权限/非法路径一律返回带 error 的空结果，不抛异常，
    保证 /api/ls 永远回 200 且前端能显示可读提示。
    """
    if not path:
        return {"path": "", "parent": None, "dirs": [], "files": [],
                "drives": list_drives(), "error": None}

    path = os.path.abspath(path)

    # 只在"确实是文件"时才取所在目录；不存在时不再猜父目录
    if os.path.isfile(path):
        path = os.path.dirname(path)
    elif not os.path.isdir(path):
        return _empty(path, None,
                      "路径不存在：" + path)

    dirs, files = [], []
    try:
        for name in sorted(os.listdir(path), key=lambda x: x.lower()):
            fp = os.path.join(path, name)
            try:
                if os.path.isdir(fp):
                    dirs.append({"name": name, "path": fp})
                else:
                    ext = os.path.splitext(name)[1].lower()
                    if ext in IMG_EXT:
                        files.append({"name": name, "path": fp,
                                      "size": os.path.getsize(fp)})
            except OSError:
                continue
    except PermissionError:
        return _empty(path, None, "无权限访问：" + path)
    except OSError as e:
        return _empty(path, None, f"无法读取目录：{e}")

    parent = os.path.dirname(path.rstrip("\\/"))
    # 盘符根（C:\ D:\ ...）的 dirname 是 "C:"，那是「当前盘当前目录」的
    # 相对写法，list_dir("C:") 会落回进程工作目录 -> 前端「↑ 上级」在
    # 盘符根无限打转。
    #
    # 但**不能用空串表示「回盘符列表」**：空串是 falsy，会被下面的
    # `if not parent` 立刻覆盖成 None，前端判 `parent!==null` 为假，
    # 「上级」按钮就完全点不动（用户实测报此问题）。
    # 这里用 at_drive_root 哨兵把两种含义彻底分开。
    at_drive_root = bool(re.match(r"^[A-Za-z]:[\\/]?$", path.rstrip("\\/")))
    if at_drive_root:
        # 显式标记：前端据此回到盘符列表
        return {"path": path, "parent": "", "dirs": dirs,
                "files": files, "drives": [], "error": None,
                "at_drive_root": True}
    if not parent or parent == path:
        parent = None
    return {"path": path, "parent": parent, "dirs": dirs,
            "files": files, "drives": [], "error": None}


# ======================================================================
# 任务
# ======================================================================
def _free_port(start=8899, tries=60):
    for i in range(tries):
        for probe in (start + i,):
            with socket.socket() as s:
                try:
                    s.bind(("127.0.0.1", probe))
                    return probe
                except OSError:
                    continue
    return start


def _collect(src):
    """src 可以是：单文件 / 目录 / 文件路径列表"""
    if isinstance(src, list):
        ok = [p for p in src if os.path.isfile(p)]
        return sorted(ok)
    if os.path.isfile(src):
        return [src]
    out = []
    for root, _, fs in os.walk(src):
        for f in sorted(fs):
            if os.path.splitext(f)[1].lower() in IMG_EXT:
                out.append(os.path.join(root, f))
    return out


def _vec_one(src, out_svg, outdir, k, n, edge):
    from PIL import Image
    tmpdir = os.path.join(outdir, "_tmp")
    os.makedirs(tmpdir, exist_ok=True)
    target = src
    if k > 1:
        im = Image.open(src).convert("RGB")
        W, H = im.size
        big = os.path.join(tmpdir, os.path.basename(src) + f"_x{k}.png")
        im.resize((W * k, H * k), Image.LANCZOS).save(big)
        target = big
    env = {kk: vv for kk, vv in os.environ.items()
           if not kk.startswith("VEC_")}
    env.update({"VEC_N": str(n), "VEC_EDGE": str(edge), "VEC_SEED": "42",
                "VEC_W": "0",
                "VEC_TITLE": os.path.splitext(os.path.basename(src))[0]})
    try:
        _env_utf8 = dict(env)
        _env_utf8["PYTHONIOENCODING"] = "utf-8"
        r = subprocess.run([PY, VEC, target, out_svg],
                           capture_output=True, text=True,
                           encoding="utf-8", errors="replace",
                           env=_env_utf8, cwd=SCRIPTS,
                           timeout=7200)
        ok = r.returncode == 0 and os.path.exists(out_svg)
        if not ok:
            tail = (r.stderr or "").strip().splitlines()[-1:] or ["?"]
            log("    vec: " + tail[0][:120])
        return ok
    finally:
        if target != src:
            try:
                os.remove(target)
            except OSError:
                pass


def run_batch(src, outdir, k, n, edge):
    """批量跑目录内所有图片。

    所有出口（正常跑完 / 用户停止 / **任何异常**）都在 finally 统一收敛
    running=False。原实现没有 try/finally，审计实测：注入 OSError 后
    TASK["running"] 恒为 True，之后 /api/run 永远回「已有任务在运行」，
    按钮永久 disabled —— 只能重启进程。
    """
    try:
        _run_batch_impl(src, outdir, k, n, edge)
    except Exception as e:
        log(f"批量任务异常：{type(e).__name__}: {e}")
        with LOCK:
            TASK["error"] = f"{type(e).__name__}: {e}"
    finally:
        with LOCK:
            TASK["running"] = False
            TASK["ended_at"] = time.time()
            TASK["phase"] = "done"
            # 结局判定：有 error -> failed；用户主动停 -> cancelled；
            # 否则看有没有成功产物 -> success/failed
            if TASK.get("error"):
                TASK["outcome"] = "failed"
            elif TASK.get("outcome") == "cancelled":
                pass
            else:
                okn = len([r for r in (TASK.get("results") or [])
                           if r.get("svg")])
                TASK["outcome"] = "success" if okn else "failed"


def _run_batch_impl(src, outdir, k, n, edge):
    """run_batch 的实际实现（状态收敛由外层 finally 负责）。"""
    global PROC
    files = _collect(src)
    with LOCK:
        TASK.update({"running": True, "mode": "batch", "phase": "running",
                     "total": len(files), "done": 0, "current": "",
                     "results": [], "started_at": time.time(),
                     "ended_at": None, "error": None})
    log(f"批量开始：发现 {len(files)} 张图片")
    if not files:
        log("目录内没有可处理图片")
        with LOCK:
            TASK.update({"running": False, "phase": "done",
                         "ended_at": time.time()})
        return
    for i, f in enumerate(files, 1):
        with LOCK:
            if not TASK["running"]:
                # 注意：此处已持有 LOCK，必须用不加锁的 _log_raw，
                # 否则同一线程二次获取非可重入 Lock 会永久死锁。
                _log_raw(f"在第 {i} 张处停止")
                break
            TASK["current"] = os.path.basename(f)
        name = os.path.basename(f)
        out_svg = os.path.join(outdir,
                               os.path.splitext(name)[0] + ".svg")
        log(f"[{i}/{len(files)}] {name}")
        t0 = time.time()
        ok = _vec_one(f, out_svg, outdir, k, n, edge)
        dt = time.time() - t0
        if ok:
            mb = os.path.getsize(out_svg) / 1048576
            log(f"    -> {mb:.1f} MB  {dt:.0f}s")
            with LOCK:
                TASK["results"].append({"file": name, "svg": out_svg,
                                        "mb": round(mb, 1),
                                        "sec": round(dt, 1)})
        else:
            log(f"    -> 失败 {dt:.0f}s")
            with LOCK:
                TASK["results"].append({"file": name, "svg": None,
                                        "err": "vec 失败"})
        with LOCK:
            TASK["done"] = i
    good = [r for r in TASK["results"] if r.get("svg")]
    log(f"批量完成：成功 {len(good)} / {len(files)}")
    rep = os.path.join(outdir, "_batch_report.json")
    try:
        json.dump(TASK["results"], open(rep, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        log(f"清单：{rep}")
    except OSError as e:
        log(f"清单写入失败：{e}")
    with LOCK:
        TASK.update({"running": False, "phase": "done",
                     "ended_at": time.time()})


def run_single(src, outdir, tier, k, n, edge):
    """src 可为单文件、目录或文件数组。数组时逐张跑（每张一个 SVG）。

    所有出口（正常跑完 / 用户停止 / 异常）都在 finally 统一收敛状态。
    """
    files = _collect(src)
    if not files:
        with LOCK:
            TASK.update({"running": False, "phase": "done",
                         "ended_at": time.time(),
                         "error": "没有可处理图片"})
        return
    total = len(files)
    stopped = False
    try:
        with LOCK:
            TASK.update({"running": True, "mode": "single",
                         "phase": "running", "total": total, "done": 0,
                         "results": [], "started_at": time.time(),
                         "ended_at": None, "error": None})
        for idx, one in enumerate(files, 1):
            with LOCK:
                if not TASK["running"]:
                    stopped = True
                    _log_raw(f"在第 {idx} 张处停止")
                    break
                TASK["done"] = idx - 1
                TASK["current"] = os.path.basename(one)
            _single_one(one, outdir, tier, k, n, edge, idx, total)
            with LOCK:
                TASK["done"] = idx
    except Exception as e:
        _log_raw(f"run_single 异常：{e}")
        with LOCK:
            TASK["error"] = str(e)
    finally:
        with LOCK:
            TASK.update({"running": False, "phase": "done",
                         "ended_at": time.time(), "current": ""})
        if stopped:
            log(f"已停止（共 {total} 张）")


def _watch_state(port, timeout=2.0):
    """读取 watch_draw 的 /state，失败返回 None。"""
    import urllib.request
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/state", timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except Exception:
        return None


def _pump_logs(proc, stop_evt):
    """后台线程：读子进程 stdout 并转发到 TASK 日志（不阻塞主流程）。"""
    try:
        for line in proc.stdout:
            if stop_evt.is_set():
                break
            line = line.rstrip()
            if line:
                log("  " + line)
    except Exception:
        pass


def _kill_watch():
    """终止**所有**遗留的可视化子进程（避免端口堆积 / 并发漏杀）。

    必须杀**整棵进程树**：watch_draw 会再派出 vec.py / max_quality.py /
    quality_boost.py 这些孙进程。`Popen.kill()` 在 Windows 上就是
    TerminateProcess，只杀直接子进程、不级联，也不给 watch_draw 自己的
    看门狗任何运行机会 —— 实测会留下 10~20 秒的孤儿窗口，期间还在写盘
    产出无人认领的中间文件（如 `_t3_base.svg` 1.2MB）。
    """
    global PROC
    with LOCK:
        procs = list(PROC_LIST)
        PROC_LIST.clear()
        PROC = None
        TASK["watch_port"] = None
    for pr in procs:
        try:
            if pr.poll() is not None:
                continue
            _kill_tree(pr.pid)
        except Exception:
            try:
                pr.kill()
            except Exception:
                pass


def _kill_tree(pid):
    """杀 pid 及其全部后代。Windows 用 taskkill /T /F，其它平台用 killpg。"""
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(pid)],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=15)
        else:
            import signal
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except Exception:
                os.kill(pid, signal.SIGKILL)
    except Exception:
        pass



def _single_one(src, outdir, tier, k, n, edge, idx, total):
    """启动 watch_draw 子进程，等它跑完（用 /state 的 done 信号，不等 EOF）。

    注意：watch_draw 画完后**不会退出**（它保持可视化页面可用），
    所以绝不能用 `for line in PROC.stdout` 等 EOF —— 那会永久阻塞，
    导致任务状态永远停在 running，前端按钮永久禁用。
    """
    global PROC
    import threading as _th

    base = os.path.splitext(os.path.basename(src))[0]
    out_svg = os.path.join(outdir, base + ".svg")
    # 记录旧产物状态：目录里若已有同名文件，不能用 exists 判本次成功
    _old_stamp = None
    if os.path.exists(out_svg):
        try:
            _st = os.stat(out_svg)
            _old_stamp = (_st.st_mtime_ns, _st.st_size)
        except OSError:
            _old_stamp = None
    port = _free_port(9700, 120)

    # 起新任务前，先清掉上一次遗留的可视化进程
    _kill_watch()

    # --keepalive 必须传：webui 用 --no-open 启动，不会打开浏览器，
    # 因此永远收不到页面心跳。watch_draw 的启动宽限（--start-grace，
    # 默认 60s）会因此把绘制当成"无人访问"而提前杀死 —— tier2 需要约
    # 495 秒、tier4/k=4 需要 200+ 秒，都会被 60s 硬上限砍掉且不产出文件。
    #
    # 注意：光传 --keepalive 还不够。watch_draw 的启动宽限判定曾漏掉
    # `and not KEEPALIVE[0]` 检查，导致 keepalive 形同虚设（v1.16.0 已修）。
    # 进程生命周期由 webui 自己管：
    # 下次任务开始时 _kill_watch() 会清理上一个。
    cmd = [PY, "-u", "-X", "utf8", WATCH, src,
           "--tier", str(tier), "--k", str(k),
           "--colors", str(n), "--edge", str(edge),
           "--out", out_svg, "--port", str(port),
           "--no-open", "--keepalive"]

    with LOCK:
        TASK.update({"mode": "single", "phase": "starting",
                     "watch_port": port, "started_at": time.time(),
                     "error": None})
    log(f"[{idx}/{total}] 单张：{os.path.basename(src)}")
    log(f"  档位={tier}  N={n}  k={k}  EDGE={edge}")
    log(f"  输出={out_svg}")
    log(f"  启动可视化服务 端口 {port}")

    stop_evt = _th.Event()
    try:
        # encoding 必须显式指定 utf-8：子进程中文输出若按 GBK 编码，
        # 父进程按 UTF-8 解码就会乱码（Windows 子系统默认 locale=cp936）。
        _env_utf8 = {k: v for k, v in os.environ.items()
                     if not k.startswith("VEC_")}
        _env_utf8["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            env=_env_utf8, cwd=TOOLS,
            creationflags=getattr(subprocess,
                                  "CREATE_NEW_PROCESS_GROUP", 0))
        global PROC
        with LOCK:
            PROC = proc
            PROC_LIST.append(proc)
        # 日志转发放后台线程，主流程继续往下走
        _th.Thread(target=_pump_logs, args=(proc, stop_evt),
                   daemon=True).start()

        # 轮询 /state 等 done
        t0 = time.time()
        last_phase = ""
        while True:
            st = _watch_state(port)
            if st:
                ph = st.get("phase_short") or st.get("phase") or ""
                if ph and ph != last_phase:
                    last_phase = ph
                    with LOCK:
                        TASK["phase"] = "running"
                if st.get("done"):
                    break
                if st.get("error"):
                    log(f"  子进程报错：{st['error']}")
                    break
            # 子进程意外退出也要收敛
            if proc.poll() is not None:
                break
            # 用户点了停止
            with LOCK:
                if not TASK["running"]:
                    # 已持锁，必须用不加锁版本，否则二次取锁永久死锁
                    _log_raw("  收到停止请求")
                    break
            if time.time() - t0 > 6 * 3600:
                log("  超时（6 小时）")
                break
            time.sleep(0.5)
    except Exception as e:
        log(f"  异常：{e}")
    finally:
        stop_evt.set()

    ok = False
    if os.path.exists(out_svg):
        try:
            _st2 = os.stat(out_svg)
            _now2 = (_st2.st_mtime_ns, _st2.st_size)
            # 本次成功 <=> 文件是新的（mtime/size 与起任务前不同）
            ok = (_old_stamp is None) or (_now2 != _old_stamp)
        except OSError:
            ok = False
    if ok:
        log(f"完成：{out_svg}  {os.path.getsize(out_svg)/1048576:.1f} MB")
        with LOCK:
            TASK["results"].append({
                "file": os.path.basename(src), "svg": out_svg,
                "mb": round(os.path.getsize(out_svg)/1048576, 1)})
    else:
        log("未生成 SVG")
    # 注意：这里**故意不杀**子进程，让用户还能打开可视化页面。
    # 下一次启动时由 _kill_watch() 清理。


# ======================================================================
# 页面
# ======================================================================
HTML = r"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>dsh-img-to-svg</title>
<style>
:root{
  --orange:#ff7b00; --lime:#b9e432;
  --paper:#e7e5e1; --ink:#111214; --mute:#57544f; --faint:#cfccc6;
  --card:#f6f5f2; --tab:#e2e0dc;
  --accent:#7a9c12; --accent-deep:#4d6a00;
  --mono:"JetBrains Mono","Cascadia Mono",Consolas,"SF Mono",monospace;
  --display:"Montserrat","MiSans","Noto Sans CJK SC","PingFang SC",sans-serif;
}
html.theme-verify{
  --paper:#dfe3e8; --ink:#101821; --mute:#4e5a66; --faint:#b8c2cc;
  --card:#eef1f4; --tab:#cfd6dd;
  --accent:#1771bf; --accent-deep:#0b4f88;
}
html.theme-denied{
  --paper:#0b0b0c; --ink:#f2f0eb; --mute:#8d8a84; --faint:#2b2b2d;
  --card:#151618; --tab:#1d1e20;
  --accent:#ff7b00; --accent-deep:#ffa040;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%}
body{
  background:var(--paper);color:var(--ink);
  font:13.5px/1.65 var(--mono);
  display:flex;flex-direction:column;overflow:hidden;
  transition:background .28s,color .28s;
}
body::before{
  content:"";position:fixed;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,var(--accent),transparent 72%);
  z-index:99;
}
header{
  height:56px;flex:0 0 56px;display:flex;align-items:center;gap:14px;
  padding:0 16px;border-bottom:1px solid var(--faint);background:var(--card);
}
.mark{width:28px;height:28px;border:1.5px solid var(--ink);
  display:grid;place-items:center;font:800 15px/1 var(--display)}
h1{font:800 12px/1.15 var(--display);letter-spacing:2.4px;
  text-transform:uppercase}
.brand small{display:block;font:600 10px/1.4 var(--mono);
  letter-spacing:1.2px;color:var(--mute);margin-top:2px}
.spacer{flex:1}
.themes{display:flex;gap:4px}
.th{width:24px;height:24px;border:1px solid var(--faint);
  background:transparent;cursor:pointer;display:grid;place-items:center}
.th i{width:11px;height:11px;display:block;border-radius:1px}
.th.t1 i{background:#b9e432}.th.t2 i{background:#1771bf}
.th.t3 i{background:#ff7b00}
.th.on{border-color:var(--ink);box-shadow:0 0 0 1px var(--ink) inset}
main{flex:1;display:flex;min-height:0}
aside{flex:0 0 330px;display:flex;flex-direction:column;overflow-y:auto;
  border-right:1px solid var(--faint);background:var(--card);min-height:0}
.sec{border-bottom:1px solid var(--faint)}
.sec-h{display:flex;align-items:center;gap:8px;padding:8px 12px;
  font:700 10.5px/1 var(--mono);letter-spacing:1.8px;background:var(--tab);
  border-left:3px solid var(--accent);text-transform:uppercase}
.sec-h i{font-style:normal;font:600 10px/1 var(--mono);color:var(--mute)}
.sec-b{padding:10px 12px 12px}
.row{display:flex;align-items:center;gap:8px;margin-bottom:9px}
.row label{font:600 11.5px/1 var(--mono);color:var(--mute);width:74px;
  flex:0 0 74px}
/* 悬停预览浮层：跟随鼠标、不吃事件、不参与布局 */
/* 预览浮层：**不要底色/边框/内边距**，否则亮色缩略图外会包一圈黑框。
   只留阴影做视觉层次（阴影在图片外缘，不侵占图片区域）。 */
#go,#go *{white-space:nowrap}
#hvprev{position:fixed;z-index:9999;pointer-events:none;display:none;
  line-height:0;border-radius:3px;
  box-shadow:0 6px 22px rgba(0,0,0,.45)}
#hvprev img{display:block;border-radius:3px}

/* ---- 侧栏缩略图预览区（「就绪」下方，蓝框区域）---- */
/* flex:1 自动占满剩余高度；图片 max-* 保证不超出该区域且不变形 */
#areaWrap{flex:1 1 auto;min-height:110px;display:flex;flex-direction:column;
  border-top:1px solid var(--faint);margin-top:10px;
  padding:8px 12px 4px}   /* 左右 12px 与侧栏其余块对齐 */
/* 关闭时：只隐藏**预览体**，绝不隐藏头部——否则开关自己也消失，
   用户再也点不回来（这是 v1.31 修 M4 时引入的回归）。
   收起用 flex:0 0 auto，开关行仍可见可点。 */
#areaWrap.off{flex:0 0 auto}
#areaWrap.off #areaBody{display:none}
#areaHead{display:flex;align-items:center;gap:8px;margin-bottom:6px}
#areaHead .lbl{font-family:var(--mono);font-size:10px;letter-spacing:1.6px;
  color:var(--mute);text-transform:uppercase}
/* 莱茵风格开关：细线方框 + 等宽标签；开启时填充 accent */
.sw{display:inline-flex;align-items:center;gap:6px;cursor:pointer;
  user-select:none;border:1px solid var(--faint);background:var(--card);
  padding:3px 8px;font-family:var(--mono);font-size:10px;letter-spacing:1.2px;
  color:var(--mute);transition:border-color .12s,color .12s}
.sw:hover{border-color:var(--accent);color:var(--ink)}
.sw i{width:9px;height:9px;border:1px solid var(--mute);display:block;
  flex:0 0 9px}
.sw.on{border-color:var(--accent);color:var(--accent-deep);font-weight:700}
.sw.on i{background:var(--accent);border-color:var(--accent-deep)}
#areaBody{flex:1;min-height:0;display:flex;align-items:flex-start;
  justify-content:flex-start;overflow:hidden}
#areaBody img{max-width:100%;max-height:100%;object-fit:contain;
  display:block;border:1px solid var(--faint);background:var(--tab)}
#areaBody .ph{font-family:var(--mono);font-size:10px;color:var(--faint);
  letter-spacing:.6px}
.khint{font-size:11.5px;color:var(--mute);margin-left:8px;line-height:1.5;
  word-break:break-all}
.khint.warn{color:#e0b341}
select,input[type=text]{background:var(--card);color:var(--ink);
  border:1px solid var(--faint);padding:5px 7px;font:600 12px/1.3 var(--mono);
  width:100%;outline:none}
select:focus,input[type=text]:focus{border-color:var(--accent)}
.modes{display:flex;gap:6px;margin-bottom:10px}
.mode{flex:1;padding:7px 6px;border:1px solid var(--faint);background:transparent;
  color:var(--mute);cursor:pointer;font:700 11.5px/1.3 var(--mono);
  text-align:center;transition:.16s}
.mode.on{background:var(--accent);border-color:var(--accent);
  color:var(--ink)}
.mode small{display:block;font:500 9.5px/1.4 var(--mono);
  letter-spacing:.3px;margin-top:2px}
.btns{display:flex;gap:8px;margin-top:4px}
button.act{flex:1;padding:9px 10px;border:1px solid var(--accent);
  background:var(--accent);color:var(--ink);cursor:pointer;
  font:800 12px/1 var(--mono);letter-spacing:1px;transition:.16s}
button.act:hover{filter:brightness(.94)}
button.act.ghost{background:transparent;color:var(--mute);
  border-color:var(--faint)}
button.act.ghost:hover{border-color:var(--ink);color:var(--ink)}
button.act:disabled{opacity:.42;cursor:not-allowed}
/* 文件浏览 */
.work{flex:1;display:flex;flex-direction:column;min-width:0}
.crumbs{flex:0 0 auto;display:flex;align-items:center;gap:6px;
  padding:8px 12px;border-bottom:1px solid var(--faint);
  background:var(--tab);font:600 11.5px/1 var(--mono)}
.crumbs .path{flex:1;overflow:hidden;text-overflow:ellipsis;
  white-space:nowrap;color:var(--ink)}
.crumbs button{background:transparent;border:1px solid var(--faint);
  color:var(--mute);cursor:pointer;font:600 11px/1 var(--mono);
  padding:4px 8px}
.crumbs button:hover{border-color:var(--ink);color:var(--ink)}
.browse{flex:1;overflow:auto;padding:10px 12px;min-height:0}
.item{display:flex;align-items:center;gap:10px;padding:6px 8px;
  border-bottom:1px dotted var(--faint);cursor:pointer;transition:.12s}
.item:hover{background:var(--tab)}
.item .ico{width:16px;text-align:center;color:var(--accent-deep);
  font-size:12px}
.item .nm{flex:1;overflow:hidden;text-overflow:ellipsis;
  white-space:nowrap;font:600 12px/1.4 var(--mono)}
.item .sz{color:var(--mute);font:600 11px/1 var(--mono)}
.item.sel{background:var(--accent);}
.item.sel .nm,.item.sel .sz,.item.sel .ico{color:var(--ink)}
.item.dir .ico{color:var(--accent-deep)}
/* 选择工具条与复选框 */
.selbar{display:flex;align-items:center;gap:6px;padding:6px 12px;
  border-bottom:1px solid var(--faint);background:var(--card);
  font:600 11.5px/1 var(--mono)}
.selbar .cnt{color:var(--accent-deep);font-weight:800}
.selbar button{background:transparent;border:1px solid var(--faint);
  color:var(--mute);cursor:pointer;font:600 11px/1 var(--mono);
  padding:4px 9px;transition:.14s}
.selbar button:hover{border-color:var(--ink);color:var(--ink)}
.item .cb{
  width:15px;height:15px;flex:0 0 15px;border:1px solid var(--faint);
  display:grid;place-items:center;color:transparent;font-size:11px;
  background:var(--card);transition:.12s;
}
.item .cb.on{background:var(--accent);border-color:var(--accent);
  color:var(--ink)}
.item.file.sel{background:var(--tab)}
.item.file.sel .nm{color:var(--accent-deep);font-weight:800}
/* 目录选择浮层（网页内嵌，与页面绑定） */
#dirModal{
  position:fixed;inset:0;z-index:200;display:none;
  background:rgba(0,0,0,.34);
  align-items:center;justify-content:center;
}
#dirModal.on{display:flex}
.dlg{
  width:min(760px,92vw);height:min(560px,86vh);
  background:var(--card);border:1px solid var(--faint);
  border-left:3px solid var(--accent);
  box-shadow:0 12px 48px rgba(0,0,0,.32);
  display:flex;flex-direction:column;
}
.dlg-h{
  display:flex;align-items:center;gap:8px;padding:9px 13px;
  background:var(--tab);border-bottom:1px solid var(--faint);
  font:700 11.5px/1 var(--mono);letter-spacing:1.4px;
  text-transform:uppercase;
}
.dlg-h .sp{flex:1}
.dlg-h button{
  background:transparent;border:1px solid var(--faint);color:var(--mute);
  cursor:pointer;font:600 11px/1 var(--mono);padding:4px 8px;
}
.dlg-h button:hover{border-color:var(--ink);color:var(--ink)}
.dlg-crumb{
  display:flex;align-items:center;gap:6px;padding:7px 13px;
  border-bottom:1px solid var(--faint);background:var(--card);
  font:600 11.5px/1 var(--mono);
}
.dlg-crumb .p{flex:1;overflow:hidden;text-overflow:ellipsis;
  white-space:nowrap}
.dlg-crumb button{
  background:transparent;border:1px solid var(--faint);color:var(--mute);
  cursor:pointer;font:600 11px/1 var(--mono);padding:4px 8px;
}
.dlg-crumb button:hover{border-color:var(--ink);color:var(--ink)}
.dlg-list{flex:1;overflow:auto;padding:6px 8px;min-height:0}
.dlg-item{
  display:flex;align-items:center;gap:9px;padding:6px 8px;
  border-bottom:1px dotted var(--faint);cursor:pointer;
}
.dlg-item:hover{background:var(--tab)}
.dlg-item .i{width:15px;text-align:center;color:var(--accent-deep)}
.dlg-item .n{flex:1;overflow:hidden;text-overflow:ellipsis;
  white-space:nowrap;font:600 12px/1.4 var(--mono)}
.dlg-item .s{color:var(--mute);font:600 11px/1 var(--mono)}
.dlg-item.cur{background:var(--accent)}
.dlg-item.cur .n,.dlg-item.cur .i,.dlg-item.cur .s{color:var(--ink)}
.dlg-f{
  display:flex;align-items:center;gap:8px;padding:9px 13px;
  border-top:1px solid var(--faint);background:var(--tab);
}
.dlg-f .picked{flex:1;overflow:hidden;text-overflow:ellipsis;
  white-space:nowrap;font:600 11.5px/1 var(--mono);color:var(--mute)}
.dlg-f button.act{flex:0 0 auto;padding:7px 14px}
/* 日志 */
.logbox{flex:0 0 190px;overflow:auto;padding:8px 12px;
  border-top:1px solid var(--faint);font:500 11.5px/1.7 var(--mono);
  background:var(--card);min-height:0}
.lg{display:flex;gap:8px}
.lg .t{color:var(--mute);flex:0 0 52px}
.lg .m{color:var(--ink);word-break:break-all}
.bar{height:4px;background:var(--faint);margin:0 12px 10px;position:relative}
.bar i{position:absolute;left:0;top:0;bottom:0;width:0;
  background:var(--accent);transition:width .3s}
.status{display:flex;align-items:center;gap:10px;padding:8px 12px;
  border-top:1px solid var(--faint);font:600 11.5px/1 var(--mono);
  background:var(--tab)}
.dot{width:8px;height:8px;border-radius:50%;background:var(--accent)}
.dot.run{animation:pulse 1.4s infinite}
@keyframes pulse{50%{opacity:.32}}
.jump{display:none;margin-left:auto;padding:6px 12px;background:var(--accent);
  color:var(--ink);border:0;cursor:pointer;font:800 11.5px/1 var(--mono);
  letter-spacing:.6px}
.jump.on{display:block}
</style></head><body>

<header>
  <div class="mark">&#8734;</div>
  <div class="brand">
    <h1>Vectorize Console</h1>
    <small>dsh-img-to-svg / 位图转分层矢量</small>
  </div>
  <div class="spacer"></div>
  <div class="themes">
    <button class="th t1 on" id="th1" title="READY"></button>
    <button class="th t2" id="th2" title="VERIFYING"></button>
    <button class="th t3" id="th3" title="DENIED"></button>
  </div>
</header>

<main>
  <aside>
    <div class="sec">
      <div class="sec-h"><i>Sec.01</i> 模式</div>
      <div class="sec-b">
        <div class="modes">
          <button class="mode on" id="mSingle">单张<small>显示绘制过程</small></button>
          <button class="mode" id="mBatch">批量<small>后台跑，不展示</small></button>
        </div>
      </div>
    </div>

    <div class="sec">
      <div class="sec-h"><i>Sec.02</i> 参数</div>
      <div class="sec-b">
        <div class="row"><label>档位</label>
          <select id="tier">
            <option value="1">1 快速</option>
            <option value="2">2 高质量</option>
            <option value="3">3 局部重绘</option>
            <option value="4" selected>4 上采样</option>
          </select></div>
        <div class="row"><label>色数 N</label>
          <select id="colors">
            <option>56</option><option>96</option>
            <option>128</option><option selected>160</option>
            <option>192</option>
          </select></div>
        <div class="row"><label>放大 k</label>
          <select id="kk">
            <option value="1">1</option><option value="2">2</option>
            <option value="3">3</option><option value="4">4</option>
            <option value="5">5</option><option value="6">6</option>
            <option value="8" selected>8</option>
          </select>
          <span id="khint" class="khint"></span></div>
        <div class="row"><label>EDGE</label>
          <select id="edge">
            <option selected>0.16</option><option>0.30</option>
            <option>0.50</option>
          </select></div>
        <div class="row"><label>输出</label>
          <input type="text" id="out" value="">
          <button class="act ghost" id="pickdir"
                  style="flex:0 0 auto;padding:5px 9px">浏览…</button></div>
        <div class="btns">
          <button class="act" id="go">开始</button>
          <button class="act ghost" id="stop">停止</button>
        </div>
      </div>
    </div>

    <div class="sec-h"><i>Sec.03</i> 进度</div>
    <div class="bar"><i id="pbar"></i></div>
    <div class="status">
      <span class="dot" id="dot"></span>
      <span id="stext">就绪</span>
      <button class="jump" id="jump">打开绘制过程 →</button>
    </div>

    <!-- 缩略图预览区：占用「就绪」下方的剩余空间；尺寸永不超过该区域 -->
    <div id="areaWrap">
      <div id="areaHead">
        <span class="sw on" id="swPreview"><i></i>显示缩略图</span>
      </div>
      <div id="areaBody">
        <span class="ph">悬停文件显示缩略图</span>
      </div>
    </div>
  </aside>

  <div class="work">
    <div class="crumbs">
      <button id="up">↑ 上级</button>
      <button id="drv">盘符</button>
      <div class="path" id="cpath">选择盘符开始浏览…</div>
    </div>
    <div class="selbar">
      <button id="selAll">全选</button>
      <button id="selInv">反选</button>
      <button id="selNone">清除</button>
      <span style="color:var(--mute)">已选</span>
      <span class="cnt" id="selCnt">0</span>
      <span style="color:var(--mute)">张</span>
    </div>
    <div class="browse" id="browse"></div>
    <div class="logbox" id="logs"></div>
  </div>
</main>

<div id="dirModal">
  <div class="dlg">
    <div class="dlg-h">
      <span>选择输出目录</span><span class="sp"></span>
      <button id="dmClose">关闭</button>
    </div>
    <div class="dlg-crumb">
      <button id="dmUp">↑ 上级</button>
      <button id="dmDrv">盘符</button>
      <div class="p" id="dmPath">—</div>
    </div>
    <div class="dlg-list" id="dmList"></div>
    <div class="dlg-f">
      <div class="picked" id="dmPicked">未选择</div>
      <button class="act" id="dmOk">选此目录</button>
    </div>
  </div>
</div>

<script>
const $=id=>document.getElementById(id);
let MODE='single', CURDIR='', PICK='';
let SEL = new Set();          // 已勾选的文件绝对路径

/* ---- 主题 ---- */
const THEMES={1:'theme-ready',2:'theme-verify',3:'theme-denied'};
function setTheme(n){
  const el=document.documentElement;
  ['theme-ready','theme-verify','theme-denied'].forEach(c=>el.classList.remove(c));
  el.classList.add(THEMES[n]||THEMES[1]);
  ['th1','th2','th3'].forEach((id,k)=>$(id).classList.toggle('on',k+1===n));
  try{localStorage.setItem('vcTheme',String(n));}catch(e){}
}
[1,2,3].forEach(n=>{$('th'+n).onclick=()=>setTheme(n);});
try{const s=parseInt(localStorage.getItem('vcTheme')||'1',10);
    if(THEMES[s])setTheme(s);}catch(e){}

/* ---- 模式 ---- */
function setMode(m){
  MODE=m;
  $('mSingle').classList.toggle('on',m==='single');
  $('mBatch').classList.toggle('on',m==='batch');
  renderBrowse();
  $('go').textContent=(m==='single')?'开始（打开绘制过程）':'开始批量';
}
$('mSingle').onclick=()=>setMode('single');
$('mBatch').onclick=()=>setMode('batch');

/* ---- 文件悬停预览：停留 0.5s 在鼠标左上方显示 15% 缩略 ---- */
const HVP={t:null,path:'',cache:new Map(),w:0,h:0,ex:0,ey:0};
function hvHide(){
  if(HVP.t){ clearTimeout(HVP.t); HVP.t=null; }
  HVP.path='';
  /* 必须重置尺寸：否则下一次悬停会先用上一张图的宽高做定位，
     浮层会在旧位置闪一下再跳到正确位置。 */
  HVP.w=0; HVP.h=0;
  const box=$('hvprev'); if(box) box.style.display='none';
}
function hvPlace(ev){
  const box=$('hvprev'); if(!box) return;
  const w=HVP.w||0, h=HVP.h||0;
  if(!w||!h) return;
  /* 需求：缩略图的**右下角始终在鼠标指针位置**。
     注意减 PAD：浮层有 padding:3px + border:1px，图片相对浮层外框内缩 4px。
     不减的话图片右下角会恒定偏离指针 +4px（审计实测 35/35 点位都有此偏差）。 */
  const PAD=0;   /* 浮层已无 padding/border，不再需要补偿 */
  let x=ev.clientX-w-PAD, y=ev.clientY-h-PAD;
  /* 唯一例外的边界保护：这样会整块出屏时就平移到屏内。
     不这么做的话，指针靠近屏幕左上角时预览完全看不见——
     那是可见性必需，不是偏离需求。平移量最小，尽量保住右下角贴合。 */
  const vw=window.innerWidth, vh=window.innerHeight;
  if(x<0) x=0;
  if(y<0) y=0;
  if(x+w>vw) x=Math.max(0,vw-w);
  if(y+h>vh) y=Math.max(0,vh-h);
  box.style.left=x+'px';
  box.style.top=y+'px';
}
/* 缩略图 blob 缓存上限。超过就淘汰最旧的并 revoke，
   否则翻大目录会一直累积（审计 M3：createObjectURL 1 处、
   revokeObjectURL 0 处）。上限取 60，约一屏文件数量级。 */
const HV_CACHE_MAX = 60;
function hvEvict(){
  while(HVP.cache.size > HV_CACHE_MAX){
    const oldest = HVP.cache.keys().next().value;
    const u = HVP.cache.get(oldest);
    try{ URL.revokeObjectURL(u); }catch(e){}
    HVP.cache.delete(oldest);
  }
}

async function hvFetch(path){
  if(HVP.cache.has(path)) return HVP.cache.get(path);
  try{
    /* auto=1：由服务端按图片方向决定目标宽
       （方图 120 / 竖图 120 宽 / 横图 200 宽，高按比例） */
    const r=await fetch('/api/thumb?src='+encodeURIComponent(path)+'&auto=1');
    if(!r.ok) return null;
    const blob=await r.blob();
    if(blob.size<80) return null;   /* 1x1 兜底图 -> 视为失败 */
    const url=URL.createObjectURL(blob);
    HVP.cache.set(path,url);
    hvEvict();                 /* 超上限就 revoke 最旧的 */
    return url;
  }catch(e){ return null; }
}
async function hvShow(ev,path){
  const box=$('hvprev'), im=$('hvprevimg');
  if(!box||!im) return;
  const url=await hvFetch(path);
  if(!url) return;
  if(HVP.path!==path) return;        /* 请求期间鼠标已移走 */
  HVP.ex=ev.clientX; HVP.ey=ev.clientY;
  /* 关键：必须**先算好位置再显示**。
     若先 display=block，此刻新图还没解码 -> naturalWidth=0 -> HVP.w/h=0
     -> hvPlace 早退（不改 left/top）-> 浮层停在**上一张图的旧坐标**上闪一下。
     首次悬停时 left/top 还是空串，fixed 元素会按静态位置渲染到页面底部。 */
  box.style.display='none';
  const reveal=()=>{
    if(HVP.path!==path) return;                /* 期间鼠标已移走 */
    HVP.w=im.naturalWidth||0;
    HVP.h=im.naturalHeight||0;
    if(!HVP.w||!HVP.h) return;                 /* 尺寸仍未知，宁可不显示 */
    hvPlace({clientX:HVP.ex,clientY:HVP.ey});  /* 先定位 */
    box.style.display='block';                 /* 后显示 -> 无闪烁 */
  };
  im.onload=reveal;
  im.src=url;
  if(im.complete && im.naturalWidth) reveal(); /* 缓存命中：立即走同一条路径 */
}
function hvBind(el,path){
  el.addEventListener('mouseenter',ev=>{
    if(HVP.t) clearTimeout(HVP.t);
    HVP.path=path;
    HVP.ex=ev.clientX; HVP.ey=ev.clientY;
    HVP.t=setTimeout(()=>{
      if(HVP.path!==path) return;
      /* 用当前记录的最新坐标，而不是 500ms 前那一下 */
      hvShow({clientX:HVP.ex,clientY:HVP.ey},path);
    },300);
  });
  el.addEventListener('mousemove',ev=>{
    /* 持续更新最新坐标：图片下载/解码期间用户可能还在移动 */
    HVP.ex=ev.clientX; HVP.ey=ev.clientY;
    const box=$('hvprev');
    if(box && box.style.display==='block') hvPlace(ev);
  });
  el.addEventListener('mouseleave',hvHide);
}

/* ---- 侧栏缩略图预览区：显示「已选中的最后一张」 ---- */
const AB={path:'',on:true,last:''};
async function areaShow(path){
  const body=$('areaBody');
  if(!body) return;
  if(!AB.on){ areaClear(); return; }
  if(!path){
    body.innerHTML='<span class="ph">悬停文件显示缩略图</span>';
    AB.path='';
    return;
  }
  const url=await hvFetch(path);
  if(!url){ body.innerHTML='<span class="ph">无法预览</span>'; return; }
  AB.path=path;
  // 复用现有 img（避免反复重建导致闪烁）
  let img=body.querySelector('img');
  if(!img){
    body.innerHTML='';
    img=document.createElement('img');
    img.alt='';
    body.appendChild(img);
  }
  img.src=url;
}
function areaClear(){
  const body=$('areaBody');
  if(!body) return;
  // **不要**清 AB.path：之前在这里清空，导致 areaRefresh 里
  // 「打开时回到当前已选图」只能拿到空串，关掉再开就回不来
  // （代码审计 M1 实测；用户体感是「关一次就得重新点文件」）。
  AB.last = AB.path || AB.last || '';
  body.innerHTML='<span class="ph">预览已关闭</span>';
}
function areaRefresh(){
  const sw=$('swPreview'), wrap=$('areaWrap');
  if(sw){
    sw.classList.toggle('on',AB.on);
    sw.setAttribute('aria-pressed',AB.on?'true':'false');
  }
  if(wrap){
    wrap.classList.toggle('off',!AB.on);
  }
  if(!AB.on){
    areaClear();
  }else{
    // 打开时回到「最后选中的图」
    const back = AB.path || AB.last || '';
    areaShow(back);
  }
}
function areaInit(){
  /* 记住开关状态（沿用主题的 localStorage 惯例，键 vcArea） */
  try{
    const v=localStorage.getItem('vcArea');
    if(v==='0') AB.on=false;
    else if(v==='1') AB.on=true;
  }catch(e){}
  const sw=$('swPreview');
  if(sw){
    sw.addEventListener('click',()=>{
      AB.on=!AB.on;
      try{ localStorage.setItem('vcArea', AB.on?'1':'0'); }catch(e){}
      areaRefresh();
    });
  }
  areaRefresh();
}

/* ---- 目录浏览 ---- */
let LAST={dirs:[],files:[]};
function fmtSize(n){
  if(n<1024)return n+' B';
  if(n<1048576)return (n/1024).toFixed(0)+' KB';
  return (n/1048576).toFixed(1)+' MB';
}
function renderBrowse(){
  const b=$('browse'); b.innerHTML='';
  if(LAST.error){
    b.innerHTML='<div style="color:#c0392b;padding:12px 4px;'+
      'font:600 12px/1.7 var(--mono)">'+LAST.error+
      '<br><span style="color:var(--mute);font-weight:500">'+
      '可点上方「盘符」重新开始浏览</span></div>';
    return;
  }
  /* 盘符兜底：若 LAST.drives 非空，先渲染成可点击条目。
     这样无论通过哪条路径拿到 drives（load('') 或首屏初始化），
     都不会出现「空目录且无法选盘符」的死局。 */
  (LAST.drives||[]).forEach(function(dp){
    const el=document.createElement('div');
    el.className='item dir';
    el.innerHTML='<span class="ico">▪</span><span class="nm"></span>';
    el.querySelector('.nm').textContent=dp;
    el.onclick=()=>load(dp);
    b.appendChild(el);
  });
  (LAST.dirs||[]).forEach(d=>{
    const el=document.createElement('div');
    el.className='item dir';
    el.innerHTML='<span class="ico">▪</span><span class="nm"></span>';
    el.querySelector('.nm').textContent=d.name;
    el.onclick=()=>load(d.path);
    b.appendChild(el);
  });
  (LAST.files||[]).forEach(f=>{
    const el=document.createElement('div');
    const on=SEL.has(f.path);
    el.className='item file'+(on?' sel':'');
    el.innerHTML='<span class="cb"></span><span class="ico">▸</span>'+
                 '<span class="nm"></span>'+
                 '<span class="sz">'+fmtSize(f.size)+'</span>';
    el.querySelector('.cb').textContent='✓';
    if(on) el.querySelector('.cb').classList.add('on');
    el.querySelector('.nm').textContent=f.name;
    el.onclick=()=>{
      if(SEL.has(f.path)) SEL.delete(f.path); else SEL.add(f.path);
      PICK = SEL.size? [...SEL][SEL.size-1] : '';
      if(!$('out').value.trim()) $('out').value=CURDIR;
      renderBrowse();
      /* 侧栏显示「已选中的最后一张」，不是悬停那张 */
      if(SEL.has(f.path)) areaShow(f.path);
      else if(PICK) areaShow(PICK);
      else areaShow('');
    };
    hvBind(el,f.path);   /* 悬停 0.3s 显示预览（按方向自适应尺寸） */
    b.appendChild(el);
  });
  updateSelCnt();
  if(!(LAST.dirs||[]).length && !(LAST.files||[]).length
     && !(LAST.drives||[]).length && !LAST.error){
    b.innerHTML='<div style="color:var(--mute)">（此目录没有子目录或图片）</div>';
  }
}
/* 「上级」可用性：盘符列表(parent===null)时置灰禁用。
   注意 parent==='' 表示「在盘符根，可回盘符列表」，那是**可点**的，
   不能用 `if(parent)` 判（空串 falsy 会误判成顶层）。 */
function syncUpState(){
  var up=$('up'); if(!up) return;
  var atTop=(LAST.parent===null||LAST.parent===undefined);
  up.disabled=!!atTop;
  up.style.opacity=atTop?'.35':'';
  up.style.cursor=atTop?'not-allowed':'';
  up.title=atTop?'已在盘符列表（最顶层）':'返回上一级';
}

async function load(path){
  const r=await fetch('/api/ls?path='+encodeURIComponent(path||''));
  const d=await r.json();
  LAST=d;
  CURDIR=d.path||''; PICK=''; SEL.clear();
  $('cpath').textContent=CURDIR||'（选择盘符）';
  syncUpState();
  if(CURDIR && !$('out').value.trim()) $('out').value=CURDIR;
  renderBrowse();
  areaShow('');            /* 换目录后侧栏清空，别显示上一个目录的图 */
}
function updateSelCnt(){
  const el=$('selCnt'); if(el) el.textContent=String(SEL.size);
}
$('selAll').onclick=()=>{
  (LAST.files||[]).forEach(f=>SEL.add(f.path));
  if(!$('out').value.trim()&&CURDIR) $('out').value=CURDIR;
  renderBrowse(); PICK=[...SEL][SEL.size-1]||''; areaShow(PICK);
};
$('selInv').onclick=()=>{
  (LAST.files||[]).forEach(f=>{
    if(SEL.has(f.path)) SEL.delete(f.path); else SEL.add(f.path);
  });
  renderBrowse(); PICK=[...SEL][SEL.size-1]||''; areaShow(PICK);
};
$('selNone').onclick=()=>{
  SEL.clear(); PICK=''; renderBrowse();
  areaShow('');            /* 清空后侧栏也要清，否则残留旧图 */
};

/* 输出目录：网页内嵌选择器（复用 /api/ls，不弹系统对话框） */
let DM = {cur:'', parent:null, picked:''};

function dmRender(){
  const b=$('dmList'); b.innerHTML='';
  $('dmPath').textContent = DM.cur || '（选择盘符）';
  if(DM.error){
    b.innerHTML='<div style="color:#c0392b;padding:14px 10px;'+
      'font:600 12px/1.7 var(--mono)">'+DM.error+
      '<br><span style="color:var(--mute);font-weight:500">'+
      '点上方「盘符」重新选择</span></div>';
    $('dmPicked').textContent='未选择';
    return;
  }
  $('dmPicked').textContent = DM.picked ? ('选中：'+DM.picked) : '未选择';
  (DM.dirs||[]).forEach(d=>{
    const el=document.createElement('div');
    el.className='dlg-item'+(DM.picked===d.path?' cur':'');
    el.innerHTML='<span class="i">▪</span><span class="n"></span>';
    el.querySelector('.n').textContent=d.name;
    el.onclick=()=>{ DM.picked=d.path; DM.cur=d.path;
                     dmLoad(d.path); };
    b.appendChild(el);
  });
  (DM.files||[]).forEach(f=>{
    const el=document.createElement('div');
    el.className='dlg-item';
    el.innerHTML='<span class="i" style="color:var(--mute)">·</span>'+
                 '<span class="n"></span><span class="s">文件</span>';
    el.querySelector('.n').textContent=f.name;
    el.style.opacity='.55';
    b.appendChild(el);
  });
  if(!(DM.dirs||[]).length && !(DM.files||[]).length){
    b.innerHTML='<div style="color:var(--mute);padding:14px 10px">'+
                '（此目录没有子目录）</div>';
  }
}

async function dmLoad(path){
  let d;
  try{
    const r=await fetch('/api/ls?path='+encodeURIComponent(path||''));
    d=await r.json();
  }catch(e){
    d={path:'', parent:null, dirs:[], files:[], error:'请求失败：'+e};
  }
  DM={cur:d.path||'', parent:d.parent||null, dirs:d.dirs||[],
      files:d.files||[], picked:DM.picked||'',
      error:d.error||null};
  if(DM.cur && !DM.error) DM.picked=DM.cur;
  dmRender();
}

/* 退到盘符列表（路径无效时用） */
async function dmDrives(){
  try{
    const r=await fetch('/api/drives'); const ds=await r.json();
    DM={cur:'', parent:null, picked:'', files:[], error:null,
        dirs:ds.map(x=>({name:x,path:x}))};
  }catch(e){
    DM={cur:'',parent:null,picked:'',files:[],dirs:[],
        error:'无可用盘符'};
  }
  dmRender();
}

async function dmOpen(){
  // 初始目录：优先当前填写的输出目录
  const init=$('out').value.trim()||CURDIR||'';
  DM.picked='';
  $('dirModal').classList.add('on');
  if(init){
    await dmLoad(init);
    if(DM.error) await dmDrives();       // 初始路径无效 -> 退盘符
  }else{
    await dmDrives();
  }
}
$('pickdir').onclick=dmOpen;
$('dmClose').onclick=()=>$('dirModal').classList.remove('on');
$('dmUp').onclick=()=>{ if(DM.parent) dmLoad(DM.parent); };
$('dmDrv').onclick=()=>dmDrives();
$('dmOk').onclick=()=>{
  const p=DM.picked||DM.cur;
  if(!p){ alert('请先选择一个目录'); return; }
  $('out').value=p;
  $('dirModal').classList.remove('on');
};
/* 点遮罩关闭 */
$('dirModal').onclick=e=>{
  if(e.target.id==='dirModal') $('dirModal').classList.remove('on');
};

/* 「上级」按钮：盘符根时 parent===''，应回到盘符列表；
   真正的根（parent===null）则禁用并给反馈。
   注意不能用 `if(LAST.parent)`——空串是 falsy，会把「回盘符列表」吞掉，
   这正是本按钮此前在 D:\ 上完全点不动的原因。 */
$('up').onclick=()=>{
  if(LAST.parent===null||LAST.parent===undefined){
    /* 已在最顶层（盘符列表），给个轻提示而不是静默 */
    if(typeof toast==='function') toast('已在盘符列表');
    return;
  }
  load(LAST.parent);          /* '' 合法 -> 回到盘符列表 */
  syncUpState();
};
$('drv').onclick=()=>{
  /* 不能用 load('')：那样只会拿到一个空目录（list_dir('') 的 dirs/files 是空的），
     页面显示「此目录没有子目录或图片」，而且回不到盘符列表。
     drives 属于另一套 UI（#dirModal / dmDrives），必须走它。 */
  /* 必须调 dmOpen 而不是 dmDrives：
     dmDrives 只拉数据 + dmRender，**不打开浮层**；
     打开浮层（classList.add('on')）是在 dmOpen 里做的。 */
  if(typeof dmOpen==='function' && document.getElementById('dirModal')){
    dmOpen();
  }else{
    load('');   /* 极端兜底：浮层不存在时至少回到空态 */
  }
};

/* 初始：列出盘符 */
fetch('/api/drives').then(r=>r.json()).then(ds=>{
  LAST={dirs:ds.map(x=>({name:x,path:x})),files:[],parent:null};
  $('cpath').textContent='选择盘符开始浏览…';
  renderBrowse();
  syncUpState();          /* 首屏也要同步，否则按钮不置灰 */
});

/* ---- 日志与状态 ---- */
let lastN=0;
async function poll(){
  try{
    const r=await fetch('/api/task'); const t=await r.json();
    const box=$('logs');
    if((t.logs_total!==undefined?t.logs_total:t.logs.length)!==lastN){
      box.innerHTML=t.logs.map(l=>
        '<div class="lg"><span class="t">'+l.t.toFixed(1)+'</span>'+
        '<span class="m"></span></div>').join('');
      const ms=box.querySelectorAll('.m');
      t.logs.forEach((l,i)=>{ if(ms[i]) ms[i].textContent=l.m; });
      box.scrollTop=box.scrollHeight;
      lastN=(t.logs_total!==undefined?t.logs_total:t.logs.length);
    }
    $('dot').className='dot'+(t.running?' run':'');
    let st='就绪';
    if(t.running){
      st = (t.mode==='batch')
        ? `批量 ${t.done}/${t.total}  ${t.current||''}`
        : '单张运行中…';
    } else if(t.mode==='batch' && t.ended_at){
      const ok=(t.results||[]).filter(x=>x.svg).length;
      st = `批量完成 ${ok}/${t.total}`;
    } else if(t.mode==='single' && t.ended_at){
      st='单张完成';
    }
    $('stext').textContent=st;
    const pct = t.total? Math.round(100*t.done/t.total):0;
    $('pbar').style.width=(t.running?(t.mode==='single'?8:pct):(t.total?100:0))+'%';
    const j=$('jump');
    if(t.mode==='single' && t.watch_port){
      j.classList.add('on');
      j.onclick=()=>window.open('http://127.0.0.1:'+t.watch_port+'/','_blank');
    } else j.classList.remove('on');
    $('go').disabled=t.running; $('stop').disabled=!t.running;
  }catch(e){}
  setTimeout(poll,500);
}
areaInit();
/* 页面卸载/刷新时释放全部 blob，避免浏览器侧泄漏 */
window.addEventListener('pagehide', function(){
  try{
    HVP.cache.forEach(function(u){
      try{ URL.revokeObjectURL(u); }catch(e){}
    });
    HVP.cache.clear();
  }catch(e){}
});
poll();

/* ---- 开始 / 停止 ---- */
$('go').onclick=async()=>{
  let out=$('out').value.trim();
  if(!out){ alert('请先选择输出目录'); return; }
  const body={
    mode:MODE, out:out,
    tier:parseInt($('tier').value,10),
    colors:parseInt($('colors').value,10),
    k:parseInt($('kk').value,10),
    edge:parseFloat($('edge').value)
  };
  if(SEL.size>0){
    body.srcs=[...SEL];                 // 多选：只跑这些
  }else if(MODE==='single'){
    alert('请先在右侧勾选至少一张图片'); return;
  }else{
    if(!CURDIR){ alert('请先进入一个目录'); return; }
    body.src=CURDIR;                    // 批量：整目录递归
  }
  const r=await fetch('/api/run',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify(body)});
  const d=await r.json();
  if(!d.ok){ alert(d.msg||'启动失败'); }
};
/* ---- k 尺度实时预估 ---- */
async function refreshKHint(){
  const el=$('khint'); if(!el) return;
  const k=parseInt($('kk').value,10)||1;
  // 优先用勾选的第一张；否则用目录里第一张图（仅单张模式）
  let one=null;
  if(SEL.size>0){ one=[...SEL][0]; }
  else if(MODE==='single' && CURDIR){ one=null; }  // 没选图就不显示
  if(!one){ el.textContent=''; el.className='khint'; return; }
  try{
    const r=await fetch('/api/plan?src='+encodeURIComponent(one)+'&k='+k);
    const d=await r.json();
    if(!d.ok){ el.textContent=''; el.className='khint'; return; }
    el.textContent='工作 '+d.text;
    el.className='khint'+(d.warn?' warn':'');
  }catch(e){ el.textContent=''; }
}
$('kk').addEventListener('change',refreshKHint);
$('tier').addEventListener('change',refreshKHint);

$('stop').onclick=async()=>{
  await fetch('/api/stop',{method:'POST'});
};
</script>
<div id="hvprev"><img id="hvprevimg" alt=""></div>
</body></html>
"""


# ======================================================================
# HTTP
# ======================================================================
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    # 客户端提前断开的各种异常（Windows 尤其爱抛 Reset/QoS）。
    _CLIENT_GONE = (BrokenPipeError, ConnectionAbortedError,
                    ConnectionResetError, TimeoutError)

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, str):
            body = body.encode("utf-8")
        # 整段包进 try：原来只有 wfile.write 在 try 里，而
        # send_header / end_headers 阶段同样会因对端断开而抛，
        # 每请求刷一屏 traceback（审计实测 RST 6/6 次）。
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except self._CLIENT_GONE:
            pass
        except Exception:
            pass

    def _origin_ok(self):
        """校验 Host / Origin / Referer，防 CSRF 与 DNS rebinding。

        本地工具只监听 127.0.0.1，但浏览器里任意网页都能向它发
        **CORS 简单请求**（text/plain、form-urlencoded 不触发预检）。
        审计实测：外域页面用 text/plain POST /api/run -> 200 且任务被启动。
        故：
          1. Host 必须是本机名（防 attacker.com 解析到 127.0.0.1）
          2. Origin 若存在必须是本机来源
          3. Referer 若存在（且无 Origin）也必须是本机来源
        """
        host = (self.headers.get("Host") or "").lower()
        hostname = host.split(":")[0]
        if hostname not in ("127.0.0.1", "localhost", "[::1]", "::1"):
            return False, f"Host 不被允许：{host}"
        for hdr in ("Origin", "Referer"):
            v = (self.headers.get(hdr) or "").strip()
            if not v:
                continue
            # 显式拒绝各种「无来源」写法。
            # 第2轮审计实测：恶意页面用 <iframe sandbox="allow-scripts">
            # 发 POST 时，浏览器送的 Origin 就是字面量 "null"，
            # 而 urlparse("null").hostname is None -> 原实现 `if hp and ...`
            # 短路放行，攻击成功（200 + 任务启动）。
            if v.lower() in ("null", "undefined", "'null'"):
                return False, f"{hdr} 为无来源（{v}），已拒绝"
            try:
                pu = urlparse(v)
                hp = (pu.hostname or "").lower()
            except Exception:
                return False, f"{hdr} 无法解析"
            # 关键：hostname 解析不出来（空）一律拒绝，不再用 `if hp and`
            # 短路。file:// / data: / blob: 等非 http(s) 方案也在这里被拒。
            if not hp:
                return False, f"{hdr} 缺少主机名（{v[:40]}），已拒绝"
            if pu.scheme and pu.scheme not in ("http", "https"):
                return False, f"{hdr} 方案不被允许：{pu.scheme}"
            if hp not in ("127.0.0.1", "localhost", "::1"):
                return False, f"{hdr} 不被允许：{hp}"
        return True, ""

    def do_GET(self):
        u = urlparse(self.path)
        qs = parse_qs(u.query)
        _ok, _why = self._origin_ok()
        if not _ok:
            self._send(403, json.dumps({"ok": False, "msg": _why},
                                       ensure_ascii=False))
            return

        if u.path in ("/", "/index.html"):
            self._send(200, HTML, "text/html; charset=utf-8")
            return

        if u.path == "/api/plan":
            # k 尺度预演：只读图尺寸做算术，不启动任何子进程。
            # 复用 watch_draw 的 _k_tier，保证与命令行 --plan 的数字一致。
            p = (qs.get("src") or [""])[0]
            try:
                kv = int((qs.get("k") or ["1"])[0])
            except (TypeError, ValueError):
                kv = 1
            try:
                from PIL import Image
                if not p or not os.path.exists(p):
                    raise ValueError("路径不存在")
                with Image.open(p) as _im:
                    W0, H0 = _im.size
                kv = max(1, min(16, kv))
                wp, hp = W0 * kv, H0 * kv
                mp = wp * hp / 1e6
                base = W0 * H0 / 1e6
                # 与 watch_draw._k_tier 完全相同的分档
                if mp < 8:
                    name, dur, warn = "轻", "通常 1-3 分钟", ""
                elif mp < 25:
                    name, dur, warn = "中", "通常 3-8 分钟", ""
                elif mp < 60:
                    name, dur, warn = "重", "5-15 分钟量级", "长任务区间"
                elif mp < 120:
                    name, dur, warn = "很重", "10-30 分钟量级", "长时间占用"
                else:
                    name, dur, warn = "极重", "可能 30 分钟以上", "建议降 k"
                payload = {
                    "ok": True,
                    "src_w": W0, "src_h": H0,
                    "work_w": wp, "work_h": hp,
                    "mp": round(mp, 1),
                    "base_mp": round(base, 2),
                    "ratio": kv * kv,
                    "tier": name, "dur": dur, "warn": warn,
                    "text": f"{wp}x{hp}（{mp:.1f} MP，是 k=1 的 {kv*kv}x）"
                            f" · {name}：{dur}"
                            + (f" · ⚠ {warn}" if warn else ""),
                }
            except Exception as e:
                payload = {"ok": False, "msg": str(e)[:160]}
            self._send(200, json.dumps(payload, ensure_ascii=False))
            return

        if u.path == "/api/thumb":
            # 文件列表悬停预览：按「原图的 pct%」生成缩略图。
            # 任何失败都回 1x1 透明 PNG（前端会因尺寸过小而放弃显示），
            # 不抛 500。失败时**会记一行日志**便于排查（此前注释误写「绝不污染日志」）。
            p = (qs.get("src") or [""])[0]
            try:
                pctv = float((qs.get("pct") or ["15"])[0])
            except (TypeError, ValueError):
                pctv = 15.0
            # nan / inf 必须显式回落 15：
            #   Python 里 min(60.0, nan) == 60.0（nan 参与比较恒 False），
            #   不判有限性的话 pct=nan 会被夹成「最大档 60%」而不是默认。
            if pctv != pctv or pctv in (float("inf"), float("-inf")):
                pctv = 15.0
            pctv = max(2.0, min(60.0, pctv))
            # auto=1：按图片方向自适应选目标宽度（前端不知道宽高，交给服务端判）
            #   方图（长短边比 <= 1.05） -> 宽 120
            #   竖图（h > w）            -> 宽 120，高按比例
            #   横图（w > h）            -> 宽 200，高按比例
            # 前端拿到 blob 后 naturalWidth/Height 即最终尺寸，无需额外请求。
            auto_v = (qs.get("auto") or ["0"])[0] == "1"
            # w= 按「目标宽度像素」缩放（方向自适应用）。w 优先于 pct。
            wv = None
            _raww = (qs.get("w") or [""])[0]
            if _raww:
                try:
                    wv = float(_raww)
                except (TypeError, ValueError):
                    wv = None
                if wv is not None and (wv != wv or wv <= 0):
                    wv = None
                if wv is not None:
                    wv = max(16.0, min(2000.0, wv))
            try:
                # 局部导入：本文件顶部没有 import io / PIL，
                # Image 只在其他函数里局部 import 过。
                # 不在这里导入会 NameError -> 走到兜底分支（实测就是那样）。
                import io as _io3
                from PIL import Image as _Img3
                with _Img3.open(p) as _im:
                    W0, H0 = _im.size
                    # 按原图 pct% 缩，再给一个很宽的夹取区间 [64, 900]。
                    #
                    # 下限为什么是 64（而不是常见的 160）：实测用户图库
                    # 1052 张，长边最小 392、中位 1500、最大 11312。
                    #   - 下限 160 会让 295 张（28%）缩不足，其中 126 张的
                    #     实际比例是 30%——是要求 15% 的两倍，肉眼明显偏大。
                    #   - 下限 64 只影响 2 张，严格 15% 达成率从 72% 升到 99.8%。
                    #   - 64px 的缩略在预览浮层里完全看得清，不会退化成色点。
                    # 同时 64 远小于图库最小长边 392，因此**不会出现「放大」**
                    # （放大才是真正违背「缩略」语义的情况）。
                    #
                    # 上限 900 仍保留：>6000 的图会缩过头（图库中仅 9 张，0.9%），
                    # 但这是为了让浮层不至于占掉半个屏幕、单张不至于上百 KB。
                    # 该取舍已写进 SKILL.md，不再隐式。
                    if auto_v:
                        # 按方向决定目标宽
                        _mx, _mn = max(W0, H0), min(W0, H0)
                        if _mn == 0:
                            _aw = 120.0
                        elif _mx / _mn <= 1.05:
                            _aw = 120.0      # 方图 120x120
                        elif H0 > W0:
                            _aw = 120.0      # 竖图：宽 120
                        else:
                            _aw = 200.0      # 横图：宽 200
                        sc = _aw / max(W0, 1)
                        tw = max(1, int(round(W0 * sc)))
                        th = max(1, int(round(H0 * sc)))
                        # 高上限：极窄高图（如 1x5000）按宽缩放会让高爆掉
                        # （实测 120x600000 = 7200 万像素、2.9 秒）。1600 足够。
                        MAXH = 1600
                        if th > MAXH:
                            sc2 = MAXH / th
                            th = MAXH
                            tw = max(1, int(round(tw * sc2)))
                    elif wv is not None:
                        # 按目标宽度：宽 = wv，高按原比例
                        sc = wv / max(W0, 1)
                        tw = max(1, int(round(W0 * sc)))
                        th = max(1, int(round(H0 * sc)))
                        # 像素总量上限：实测 w=2000 on 10x3000 会算出
                        # 2000x600000 = 12 亿像素、单请求 47 秒，能把服务打瘫。
                        # 超过 400 万像素就**等比**缩回（保形状，宽高都缩）。
                        MAXPX = 4_000_000
                        if tw * th > MAXPX:
                            sc2 = (MAXPX / (tw * th)) ** 0.5
                            tw = max(1, int(round(tw * sc2)))
                            th = max(1, int(round(th * sc2)))
                    else:
                        side = max(W0, H0) * pctv / 100.0
                        side = max(64.0, min(900.0, side))
                        sc = side / max(W0, H0, 1)
                        tw = max(1, int(round(W0 * sc)))
                        th = max(1, int(round(H0 * sc)))
                    _t = _im.convert("RGB").resize((tw, th),
                                                   _Img3.LANCZOS)
                    _buf = _io3.BytesIO()
                    _t.save(_buf, format="PNG", optimize=True)
                    _png = _buf.getvalue()
                self._send(200, _png, "image/png")
                return
            except Exception as _e3:
                # 1x1 透明 PNG 兜底（zlib 生成、PIL 已验证可解）。
                # 记一行日志便于排查，但不弹 500。
                log(f"缩略图失败({os.path.basename(p) if p else '?'}): "
                    f"{type(_e3).__name__}: {_e3}")
                self._send(200, bytes.fromhex(
                    "89504e470d0a1a0a0000000d494844520000000100000001"
                    "08060000001f15c4890000000b4944415478da63600002"
                    "0000050001e9fadcd80000000049454e44ae426082"),
                    "image/png")
                return

        if u.path == "/api/drives":
            self._send(200, json.dumps(list_drives(),
                                       ensure_ascii=False))
            return

        if u.path == "/api/ls":
            p = (qs.get("path") or [""])[0]
            try:
                payload = list_dir(p)
            except Exception as e:
                payload = _empty(p, None, f"内部错误：{e}")
            self._send(200, json.dumps(payload, ensure_ascii=False))
            return

        # DEPRECATED（保留供 API/命令行调用，UI 已改用网页内嵌浮层 #dirModal）
        # 前端 <script> 段内不再请求此路由；删此路由前请同时确认
        # _pickdir.py 无用（它负责弹 Windows 原生"选择文件夹"对话框，
        # 因存在 DPI 模糊与无 owner 窗口两个缺陷而被弃用）。
        if u.path == "/api/pickdir":
            # 已移除：原实现 subprocess.run(timeout=300) 拉起 tkinter 模态框，
            # 审计实测阻塞 >120s、真弹置顶窗口、任意网页可用 <img src> 触发。
            # 前端早已改用内嵌 #dirModal。
            self._send(410, json.dumps(
                {"ok": False, "msg": "已移除，请使用网页内嵌目录选择器"},
                ensure_ascii=False))
            return

        if u.path == "/api/task":
            with LOCK:
                snap = dict(TASK)
                snap["logs_total"] = int(
                    TASK.get("logs_total", len(TASK["logs"])))
                snap["outcome"] = TASK.get("outcome")
                snap["logs"] = TASK["logs"][-200:]
            self._send(200, json.dumps(snap, ensure_ascii=False))
            return

        self._send(404, json.dumps({"ok": False, "msg": "not found"}))

    def do_POST(self):
        global PROC
        u = urlparse(self.path)
        _ok, _why = self._origin_ok()
        if not _ok:
            self._send(403, json.dumps({"ok": False, "msg": _why},
                                       ensure_ascii=False))
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except (TypeError, ValueError):
            self._send(400, json.dumps(
                {"ok": False, "msg": "Content-Length 非法"}))
            return
        # Content-Length 上限：审计实测 CL=999999999999 直接 MemoryError，
        # 任意网页一个大 CL 就能把进程打挂。
        MAXBODY = 1 << 20          # 1 MiB，本服务 JSON 远小于此
        if n < 0 or n > MAXBODY:
            self._send(413, json.dumps(
                {"ok": False,
                 "msg": f"请求体过大或非法（{n}，上限 {MAXBODY}）"},
                ensure_ascii=False))
            return
        raw = self.rfile.read(n) if n else b"{}"

        if u.path == "/api/run":
            # 占位后任何失败路径都必须回滚 running，否则控制台会永久卡死：
            # 用户输错一次路径后，之后所有提交都被"已有任务在运行"拒绝，
            # 只能手动点停止才能恢复。
            try:
                d = json.loads(raw.decode("utf-8"))
            except Exception:
                self._send(400, json.dumps(
                    {"ok": False, "msg": "JSON 解析失败"}))
                return
            
            # 检查 + 占位在同一临界区，消除 TOCTOU
            with LOCK:
                if TASK["running"]:
                    # 409 Conflict：原来回 200，让「忙」与「成功」在 HTTP 层
                    # 无法区分，排错时很误导。
                    self._send(409, json.dumps(
                        {"ok": False, "msg": "已有任务在运行"}))
                    return
                TASK["running"] = True
            
            srcs = d.get("srcs")
            # srcs 必须是字符串数组；传 int/bool/float 会让下面的推导式抛
            # TypeError（未捕获 -> 连接直接断开，前端只看到网络错误）。
            if srcs is not None:
                if not isinstance(srcs, (list, tuple)):
                    # 此处已过占位（running=True），直接 return 会把它永久
                    # 留在 True，之后所有请求都被 409 挡死，只能重启进程。
                    with LOCK:
                        TASK.update({"running": False, "phase": "done",
                                     "ended_at": time.time()})
                    self._send(400, json.dumps(
                        {"ok": False, "msg": "srcs 必须是路径数组"},
                        ensure_ascii=False))
                    return
                srcs = [str(x) for x in srcs if isinstance(x, str)]

            # 参数解析：不能写 `d.get(x) or 默认值`——0 / 0.0 是 falsy，
            # 会被静默吞成默认值，导致下面的范围校验永不触发。
            # 审计实测：tier=0/colors=0/k=0 全返回 ok:true；
            # edge=0.0 本是合法值（不保留边缘），却被改成 0.16。
            def _num(_k, _d, _cast):
                _v = d.get(_k)
                if _v is None or _v == "":
                    return _d
                return _cast(_v)

            try:
                tier = _num("tier", 4, int)
                colors = _num("colors", 160, int)
                k = _num("k", 8, int)
                edge = _num("edge", 0.16, float)
                if not 1 <= tier <= 4:
                    raise ValueError(f"档位必须是 1-4，收到 {tier}")
                if not 2 <= colors <= 4096:
                    raise ValueError(f"色数必须在 2-4096，收到 {colors}")
                if not 1 <= k <= 32:
                    raise ValueError(f"放大倍数必须在 1-32，收到 {k}")
                if not (0.0 <= edge <= 1.0):
                    raise ValueError(f"EDGE 必须在 0-1，收到 {edge}")
            except (TypeError, ValueError) as e:
                # 参数无效回 400（原来回 200，客户端误以为"请求被接受"）；
                # 同时回滚 running，否则永久锁死。
                with LOCK:
                    TASK.update({"running": False, "phase": "done",
                                 "ended_at": time.time()})
                self._send(400, json.dumps(
                    {"ok": False, "msg": f"参数无效：{e}"},
                    ensure_ascii=False))
                return

            out = d.get("out") or ""
            ok_started = False
            mode = d.get("mode") or "single"

            # 新一轮必须清空旧日志/结果：原来全局只在模块加载时初始化一次，
            # 于是第2轮启动时还挂着上轮的 SAVED 行，叠加「结束一律显示完成」，
            # 用户会看到满屏产物却找不到文件（平台审审计 X-2 实测）。
            with LOCK:
                TASK["logs"] = []
                TASK["logs_total"] = 0
                TASK["results"] = []
                TASK["error"] = None
                TASK["outcome"] = None

            try:
                if mode == "batch":
                    # 批量模式**必须同时支持 srcs**：前端只要勾选了文件就发
                    # body.srcs（不区分模式），而原实现 batch 分支只读 src
                    # -> src="" -> 400「源目录不存在：」（冒号后还是空的）。
                    # 平台审计实测：批量+勾选 2 张必然失败，这是主力用法。
                    if srcs:
                        bad = [p for p in srcs if not os.path.isfile(p)]
                        if bad:
                            self._send(400, json.dumps(
                                {"ok": False,
                                 "msg": f"以下文件不存在：{bad[:3]}"},
                                ensure_ascii=False))
                            return
                        src = srcs
                    else:
                        src = d.get("src") or d.get("srcdir") or ""
                        if not src or not os.path.exists(src):
                            self._send(400, json.dumps(
                                {"ok": False,
                                 "msg": ("请先勾选图片，或进入一个目录再点"
                                         "「开始批量」")},
                                ensure_ascii=False))
                            return
                    threading.Thread(
                        target=run_batch,
                        args=(src, out, k, colors, edge),
                        daemon=True).start()
                else:
                    # 单张：优先 srcs（多选数组），否则 src（单路径/目录）
                    if srcs:
                        bad = [p for p in srcs if not os.path.isfile(p)]
                        if bad:
                            self._send(400, json.dumps(
                                {"ok": False,
                                 "msg": f"以下文件不存在：{bad[:3]}"},
                                ensure_ascii=False))
                            return
                        src = srcs
                    else:
                        src = d.get("src") or ""
                        if not src or not os.path.exists(src):
                            self._send(400, json.dumps(
                                {"ok": False,
                                 "msg": ("请先勾选一张图片，或进入一个目录"
                                         "再点「开始」")},
                                ensure_ascii=False))
                            return
                    threading.Thread(
                        target=run_single,
                        args=(src, out, tier, k, colors, edge),
                        daemon=True).start()
                ok_started = True
            finally:
                # 任何失败路径（含上面所有 return）都在这里回滚占位；
                # 成功起线程时 ok_started=True，生命周期交给 worker。
                if not ok_started:
                    with LOCK:
                        TASK["running"] = False
            self._send(200, json.dumps({"ok": True}))
            return

        if u.path == "/api/stop":
            with LOCK:
                TASK["running"] = False
                # 用户主动停止 -> cancelled（区别于 failed / success）
                if TASK.get("phase") not in (None, "done"):
                    TASK["outcome"] = "cancelled"
            _kill_watch()
            log("已请求停止")
            self._send(200, json.dumps({"ok": True}))
            return

        self._send(404, json.dumps({"ok": False, "msg": "not found"}))


class Server(ThreadingTCPServer):
    # 不设 allow_reuse_address：避免多个实例同时绑同一端口导致
    # 请求随机落点、状态错乱（audit 实测两个 PID 可同绑 8899）。
    # 端口占用时由 serve() 的 port+i 重试逻辑自动换端口。
    allow_reuse_address = False
    daemon_threads = True


def serve(port=8899, tries=40):
    for i in range(tries):
        try:
            srv = Server(("127.0.0.1", port + i), Handler)
            return srv, port + i
        except OSError:
            continue
    raise RuntimeError("无可用端口")


def main():
    if not os.path.exists(WATCH) or not os.path.exists(VEC):
        print("缺少依赖：")
        for lb, p in (("watch_draw.py", WATCH), ("vec.py", VEC)):
            if not os.path.exists(p):
                print(f"  {lb}: {p}")
        print(f"\n本程序目录: {THIS}")
        print(f"推导技能目录: {SKILL}")
        return 1

    srv, port = serve(8899)
    url = f"http://127.0.0.1:{port}/"
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"\n  网页界面: {url}\n")
    try:
        os.startfile(url)
    except Exception:
        pass
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
