"""gui.py -- dsh-img-to-svg 本地图形界面（非 agent 调用入口）

## 定位

用户双击 gui.py（或配套 .bat / .lnk）即可使用，无需 agent、无需记命令。

## 两种模式

### 单张模式
选择一张图片 -> 选档位与参数 -> 启动 watch_draw.py 服务 ->
自动打开浏览器，看完整绘制过程（逐层时间轴、倍速、缩放、final）。

### 批量模式（不展示过程）
选择目录或一批文件 -> 后台逐张跑 vec.py -> 实时显示进度与指标 ->
完成后汇总 MAE/p99 并给出输出目录。

## 界面

tkinter（本机可用，零额外依赖），单窗口：
  顶部：源路径选择（单张/目录）
  中部：参数（档位、色数、k 或工作宽度）
  下部：运行日志 + 进度条
  底部：开始 / 停止 / 打开输出目录

## 与 watch_draw.py 的关系

单张模式**直接调用**它，参数透传（--tier/--k/--colors/--edge/--out/--port）。
批量模式复用 vec.py，但为每张图套用同样的放大与参数逻辑，
保证"批量产物"与"单张产物"口径一致。
"""
import json
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

def _find_root():
    """工作区根：优先环境变量，否则取仓库的上一级。"""
    env = os.environ.get("IMGSVG_ROOT")
    if env and os.path.isdir(env):
        return env
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PY = sys.executable

# ---- 路径解析：以本文件自身位置为准，不依赖 DSH_HOME ----
# 本文件位于 <skill>/tools/gui.py，因此：
#   TOOLS   = <skill>/tools
#   SKILL   = <skill>
#   SCRIPTS = <skill>/scripts
THIS = os.path.dirname(os.path.abspath(__file__))
TOOLS = THIS
SKILL = os.path.dirname(THIS)
SCRIPTS = os.path.join(SKILL, "scripts")

WATCH = os.path.join(TOOLS, "watch_draw.py")
VEC = os.path.join(SCRIPTS, "vec.py")
RENDER = os.path.join(SCRIPTS, "render_check.py")

# 后备：若自身布局不成立，再尝试 DSH_HOME 下的标准位置
if not os.path.exists(WATCH) or not os.path.exists(VEC):
    _home = os.environ.get("DSH_HOME", "")
    _alt = os.path.join(_home, "skills", "dsh-img-to-svg")
    if os.path.isdir(_alt):
        SKILL = _alt
        TOOLS = os.path.join(SKILL, "tools")
        SCRIPTS = os.path.join(SKILL, "scripts")
        WATCH = os.path.join(TOOLS, "watch_draw.py")
        VEC = os.path.join(SCRIPTS, "vec.py")
        RENDER = os.path.join(SCRIPTS, "render_check.py")

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

TIER_LABEL = {
    1: "1 快速（等 1 倍矢量化）",
    2: "2 高质量（超采样 + 分块重绘）",
    3: "3 局部重绘（只修指定区域）",
    4: "4 上采样（放大后矢量化）",
}


class App:
    def __init__(self, root):
        self.root = root
        self.q = queue.Queue()
        self.proc = None
        self.running = False
        self.stop_flag = False

        root.title("dsh-img-to-svg  |  位图转分层矢量")
        root.geometry("880x620")
        root.minsize(760, 540)

        # ---------- 源选择 ----------
        f0 = ttk.LabelFrame(root, text="源文件")
        f0.pack(fill="x", padx=12, pady=(12, 6))

        self.var_mode = tk.StringVar(value="single")
        r1 = ttk.Frame(f0)
        r1.pack(fill="x", padx=10, pady=6)
        ttk.Radiobutton(r1, text="单张（显示绘制过程）",
                        variable=self.var_mode, value="single",
                        command=self._mode_changed).pack(side="left")
        ttk.Radiobutton(r1, text="批量（后台跑，不展示）",
                        variable=self.var_mode, value="batch",
                        command=self._mode_changed).pack(side="left",
                                                           padx=(18, 0))

        r2 = ttk.Frame(f0)
        r2.pack(fill="x", padx=10, pady=(0, 10))
        self.var_src = tk.StringVar()
        ttk.Entry(r2, textvariable=self.var_src).pack(
            side="left", fill="x", expand=True)
        self.btn_pick = ttk.Button(r2, text="选择…",
                                   command=self._pick_src)
        self.btn_pick.pack(side="left", padx=(6, 0))

        # ---------- 参数 ----------
        f1 = ttk.LabelFrame(root, text="参数")
        f1.pack(fill="x", padx=12, pady=6)

        g = ttk.Frame(f1)
        g.pack(fill="x", padx=10, pady=8)

        ttk.Label(g, text="档位").grid(row=0, column=0, sticky="w")
        self.var_tier = tk.StringVar(value=TIER_LABEL[4])
        cb = ttk.Combobox(g, textvariable=self.var_tier, width=30,
                          state="readonly",
                          values=[TIER_LABEL[i] for i in (1, 2, 3, 4)])
        cb.grid(row=0, column=1, sticky="w", padx=(6, 22))

        ttk.Label(g, text="色数 N").grid(row=0, column=2, sticky="w")
        self.var_n = tk.StringVar(value="160")
        ttk.Combobox(g, textvariable=self.var_n, width=8,
                     values=["56", "96", "128", "160", "192"]).grid(
            row=0, column=3, sticky="w", padx=(6, 22))

        ttk.Label(g, text="放大 k").grid(row=0, column=4, sticky="w")
        self.var_k = tk.StringVar(value="8")
        ttk.Combobox(g, textvariable=self.var_k, width=6,
                     values=["1", "2", "3", "4", "5", "6", "8"]).grid(
            row=0, column=5, sticky="w", padx=(6, 0))

        g2 = ttk.Frame(f1)
        g2.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Label(g2, text="EDGE_KEEP").pack(side="left")
        self.var_edge = tk.StringVar(value="0.16")
        ttk.Combobox(g2, textvariable=self.var_edge, width=8,
                     values=["0.16", "0.30", "0.50"]).pack(
            side="left", padx=(6, 22))
        ttk.Label(g2, text="输出目录").pack(side="left")
        self.var_out = tk.StringVar(
            value=os.path.join(_find_root(), "output", "gui"))
        ttk.Entry(g2, textvariable=self.var_out).pack(
            side="left", fill="x", expand=True, padx=(6, 6))
        ttk.Button(g2, text="选择…", command=self._pick_out).pack(
            side="left")

        # ---------- 操作 ----------
        f2 = ttk.Frame(root)
        f2.pack(fill="x", padx=12, pady=(2, 6))
        self.btn_start = ttk.Button(f2, text="开始", command=self._start)
        self.btn_start.pack(side="left")
        self.btn_stop = ttk.Button(f2, text="停止", command=self._stop,
                                   state="disabled")
        self.btn_stop.pack(side="left", padx=(8, 0))
        ttk.Button(f2, text="打开输出目录",
                   command=self._open_out).pack(side="left", padx=(8, 0))

        self.pb = ttk.Progressbar(f2, mode="determinate", length=260)
        self.pb.pack(side="right")

        # ---------- 日志 ----------
        f3 = ttk.LabelFrame(root, text="运行状态")
        f3.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.txt = tk.Text(f3, height=14, wrap="none",
                           font=("Consolas", 10))
        sb = ttk.Scrollbar(f3, command=self.txt.yview)
        self.txt.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.txt.pack(side="left", fill="both", expand=True, padx=(8, 0),
                      pady=8)

        self._log(f"技能目录: {SKILL}")
        self._log(f"解释器: {PY}")
        self._log("就绪。选单张会打开浏览器看绘制过程；选批量则后台跑。")
        self.root.after(120, self._drain)

    # ---------------- 基础 ----------------
    def _log(self, s):
        self.q.put(s)

    def _drain(self):
        try:
            while True:
                s = self.q.get_nowait()
                self.txt.insert("end", s + "\n")
                self.txt.see("end")
        except queue.Empty:
            pass
        self.root.after(120, self._drain)

    def _mode_changed(self):
        batch = self.var_mode.get() == "batch"
        self.btn_pick.configure(
            text="选择目录…" if batch else "选择图片…")

    def _pick_src(self):
        if self.var_mode.get() == "batch":
            d = filedialog.askdirectory(title="选择图片目录")
            if d:
                self.var_src.set(d)
        else:
            p = filedialog.askopenfilename(
                title="选择图片",
                filetypes=[("图片", "*.png *.jpg *.jpeg *.webp *.bmp "
                                     "*.tif *.tiff"),
                           ("全部", "*.*")])
            if p:
                self.var_src.set(p)

    def _pick_out(self):
        d = filedialog.askdirectory(title="选择输出目录")
        if d:
            self.var_out.set(d)

    def _open_out(self):
        d = self.var_out.get().strip()
        if d and os.path.isdir(d):
            os.startfile(d)
        else:
            messagebox.showinfo("提示", "输出目录还不存在")

    # ---------------- 启动 ----------------
    def _start(self):
        if self.running:
            return
        src = self.var_src.get().strip()
        if not src:
            messagebox.showwarning("缺少源", "请先选择单张图片或目录")
            return
        if not os.path.exists(src):
            messagebox.showerror("路径无效", f"不存在：\n{src}")
            return

        tier = int(self.var_tier.get()[0])
        n = int(self.var_n.get())
        k = int(self.var_k.get())
        edge = float(self.var_edge.get())
        outdir = self.var_out.get().strip()
        os.makedirs(outdir, exist_ok=True)

        self.running = True
        self.stop_flag = False
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")

        if self.var_mode.get() == "batch":
            t = threading.Thread(target=self._run_batch,
                                 args=(src, outdir, tier, n, k, edge),
                                 daemon=True)
        else:
            t = threading.Thread(target=self._run_single,
                                 args=(src, outdir, tier, n, k, edge),
                                 daemon=True)
        t.start()

    def _stop(self):
        self.stop_flag = True
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.kill()
            except Exception:
                pass
        self._log("[停止] 已请求终止当前任务")

    def _finish(self):
        self.running = False
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")

    # ---------------- 单张 ----------------
    def _run_single(self, src, outdir, tier, n, k, edge):
        try:
            base = os.path.splitext(os.path.basename(src))[0]
            out_svg = os.path.join(outdir, base + ".svg")
            port = str(9600 + int(time.time()) % 200)
            cmd = [PY, "-u", "-X", "utf8", WATCH, src,
                   "--tier", str(tier), "--k", str(k),
                   "--colors", str(n), "--edge", str(edge),
                   "--out", out_svg, "--port", port, "--keepalive"]
            self._log(f"[单张] {os.path.basename(src)}")
            self._log(f"  档位={tier}  N={n}  k={k}  EDGE={edge}")
            self._log(f"  输出: {out_svg}")
            self._log("  启动服务，浏览器将自动打开…")
            self.pb.configure(mode="indeterminate")
            self.pb.start(60)
            _env_g = {k2: v2 for k2, v2 in os.environ.items()
                      if not k2.startswith("VEC_")}
            _env_g["PYTHONIOENCODING"] = "utf-8"
            self.proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                env=_env_g, cwd=TOOLS,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            # 日志转发放后台线程（子进程画完不退出，不能等 EOF）
            import threading as _th
            _stop_evt = _th.Event()
            
            def _pump():
                try:
                    for _ln in self.proc.stdout:
                        if _stop_evt.is_set():
                            break
                        _ln = _ln.rstrip()
                        if _ln:
                            self._log('  ' + _ln)
                except Exception:
                    pass
            
            _th.Thread(target=_pump, daemon=True).start()
            
            # 轮询 /state 的 done 判完成（watch_draw 画完不退出）
            import json as _json
            import urllib.request as _urq
            _t0 = time.time()
            while True:
                time.sleep(0.5)
                if self.stop_flag:
                    break
                try:
                    with _urq.urlopen(f'http://127.0.0.1:{port}/state',
                                      timeout=2) as _r:
                        _st = _json.loads(_r.read().decode('utf-8', 'replace'))
                    if _st.get('done'):
                        break
                except Exception:
                    # 服务还没起来或已退出：看进程是否还活
                    if self.proc.poll() is not None:
                        break
                if time.time() - _t0 > 6 * 3600:
                    break
            _stop_evt.set()
        except Exception as e:
            self._log(f"[错误] {e}")
        finally:
            self.root.after(0, self._finish)

    # ---------------- 批量 ----------------
    def _collect(self, src):
        if os.path.isfile(src):
            return [src]
        out = []
        for root, _, fs in os.walk(src):
            for f in sorted(fs):
                if os.path.splitext(f)[1].lower() in IMG_EXT:
                    out.append(os.path.join(root, f))
        return out

    def _run_batch(self, src, outdir, tier, n, k, edge):
        try:
            files = self._collect(src)
            total = len(files)
            self._log(f"[批量] 发现 {total} 张图片")
            if not total:
                self._log("[批量] 目录内没有可处理图片")
                return
            self.root.after(0, lambda: self.pb.configure(
                mode="determinate", maximum=total, value=0))
            results = []
            for i, f in enumerate(files, 1):
                if self.stop_flag:
                    self._log(f"[批量] 在第 {i} 张处停止")
                    break
                name = os.path.basename(f)
                base = os.path.splitext(name)[0]
                out_svg = os.path.join(outdir, base + ".svg")
                self._log(f"[{i}/{total}] {name}")
                t0 = time.time()
                ok = self._vec_one(f, out_svg, outdir, k, n, edge)
                dt = time.time() - t0
                if ok:
                    mb = os.path.getsize(out_svg) / 1048576
                    self._log(f"    -> {mb:.1f} MB  {dt:.0f}s")
                    results.append({"file": name, "svg": out_svg,
                                    "mb": round(mb, 1),
                                    "sec": round(dt, 1)})
                else:
                    self._log(f"    -> 失败 {dt:.0f}s")
                    results.append({"file": name, "svg": None,
                                    "err": "vec 失败"})
                self.root.after(0, lambda v=i: self.pb.configure(value=v))

            good = [r for r in results if r.get("svg")]
            self._log("")
            self._log(f"[批量完成] 成功 {len(good)} / {total}")
            rep = os.path.join(outdir, "_batch_report.json")
            json.dump(results, open(rep, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
            self._log(f"清单: {rep}")
        except Exception as e:
            self._log(f"[错误] {e}")
        finally:
            self.root.after(0, self._finish)

    def _vec_one(self, src, out_svg, outdir, k, n, edge):
        """对单张图跑 vec.py。

        k>1 时先把源图物理放大（LANCZOS）存临时文件，
        再矢量化 —— 这样绕开 VEC_W 的 4000 上限，k 才真正生效。
        """
        from PIL import Image
        tmpdir = os.path.join(outdir, "_tmp")
        os.makedirs(tmpdir, exist_ok=True)
        target = src
        if k > 1:
            im = Image.open(src).convert("RGB")
            W, H = im.size
            big = os.path.join(tmpdir,
                               os.path.basename(src) + f"_x{k}.png")
            im.resize((W * k, H * k), Image.LANCZOS).save(big)
            target = big
        env = {kk: vv for kk, vv in os.environ.items()
               if not kk.startswith("VEC_")}
        env.update({"VEC_N": str(n), "VEC_EDGE": str(edge),
                    "VEC_SEED": "42", "VEC_W": "0",
                    "VEC_TITLE": os.path.splitext(
                        os.path.basename(src))[0]})
        try:
            r = subprocess.run([PY, VEC, target, out_svg],
                               capture_output=True, text=True,
                               errors="replace", env=env, cwd=SCRIPTS,
                               timeout=7200)
            ok = r.returncode == 0 and os.path.exists(out_svg)
            if not ok:
                tail = (r.stderr or "").strip().splitlines()[-1:] or ["?"]
                self._log(f"    vec: {tail[0][:90]}")
            return ok
        finally:
            if target != src:
                try:
                    os.remove(target)
                except OSError:
                    pass


def main():
    miss = [(lb, p) for p, lb in ((WATCH, "watch_draw.py"),
                                  (VEC, "vec.py"))
            if not os.path.exists(p)]
    if miss:
        print("缺少依赖文件，无法启动：")
        for lb, p in miss:
            print(f"  {lb}: {p}")
        print()
        print(f"本程序所在目录: {THIS}")
        print(f"推导的技能目录: {SKILL}")
        print("请确认 gui.py 位于 <skill>/tools/ 下，且 scripts/ 与 tools/ 同级。")
        return 1
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except Exception:
        pass
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
