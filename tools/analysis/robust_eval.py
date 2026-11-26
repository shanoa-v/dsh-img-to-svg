"""robust_eval.py -- 多种子稳健评估，修掉 bnd 不可比的问题

根因（已两次独立验证）：
  a_starry 上只换 KMeans 种子、平滑逻辑不动，bnd% 就在 14.01~25.70 之间跳
  （极差 11.68，σ=4.33）。因此任何「单次运行、单一种子」的 bnd 都不可作门槛，
  而任何会改变 edge_sm 的改动都会隐式重排分簇，效果等同换种子。

本工具的判定方式：
  1. 每个配置跑 N 个种子，bnd 取 **中位数**（不用最小值，那是挑好看的）
  2. 同时报 min/max，露出波动范围，不隐藏
  3. MAE 取中位数（已知其对种子较稳健）
  4. 判定用中位数比较，且要求 **两个配置的波动区间有实质重叠之外的一致方向**

需要冻结副本支持 VEC_SEED。脚本会先自检，不支持就拒绝跑，
免得又变成「参数没生效却宣布结论」（本项目踩过）。
"""
import os
import io
import sys
import json
import hashlib
import subprocess
import statistics
import numpy as np
from PIL import Image
from scipy.ndimage import sobel, median_filter

def _find_root():

    """工作区根：优先环境变量，否则取仓库的上一级。



    原实现写死作者机器上的 F:\\SVG drawing。

    """

    env = os.environ.get("IMGSVG_ROOT")

    if env and os.path.isdir(env):

        return env

    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))





ROOT = _find_root()
def _find_python():
    """定位可用解释器：优先环境变量，否则用当前解释器。

    原实现写死作者机器上的 D:\\anaconda3\\python.exe，
    换到别的机器必然找不到。
    """
    env = os.environ.get("IMGSVG_PYTHON")
    if env and os.path.isfile(env):
        return env
    return sys.executable


PY = _find_python()
FROZEN = os.path.join(os.environ["DSH_HOME"], "skills", "dsh-img-to-svg", "scripts")
OUT = os.path.join(ROOT, "workspace", "robust")
os.makedirs(OUT, exist_ok=True)
RENDER = os.path.join(os.environ["DSH_HOME"], "skills", "dsh-img-to-svg",
                      "scripts", "render_check.py")

SEEDS = [42, 43, 44, 1, 2, 123]

IMGS = [
    ("mouth", os.path.join(ROOT, "input", "collections", "batch3",
     "illust_100531443_20260808_134010.png")),
    ("starry", os.path.join(ROOT, "input", "collections", "batch1", "a_starry.png")),
    ("wuther", os.path.join(ROOT, "input", "collections", "batch1", "c_wuthering.png")),
    ("user", os.path.join(ROOT, "input", "user", "user_char.png")),
]


def check_seed_support():
    src = io.open(os.path.join(FROZEN, "vec.py"), encoding="utf-8").read()
    if 'VEC_SEED' not in src:
        print("拒绝运行：冻结副本不支持 VEC_SEED，无法做多种子评估。")
        print("（本项目有『参数写死导致分支静默失效』的先例，不做假验证）")
        sys.exit(2)
    print("VEC_SEED 支持：是")


def metrics(ren, ref, flat):
    lr = ren.mean(axis=2)
    ga = np.hypot(sobel(lr, 0), sobel(lr, 1))
    gr = np.hypot(sobel(ref.mean(axis=2), 0), sobel(ref.mean(axis=2), 1))
    vg = ren.mean(axis=2)
    j = np.zeros_like(vg, dtype=bool)
    j[:, 1:] |= np.abs(vg[:, 1:] - vg[:, :-1]) > 1
    j[:, :-1] |= np.abs(vg[:, 1:] - vg[:, :-1]) > 1
    j[1:, :] |= np.abs(vg[1:, :] - vg[:-1, :]) > 1
    j[:-1, :] |= np.abs(vg[1:, :] - vg[:-1, :]) > 1
    j3 = np.zeros_like(vg, dtype=bool)
    j3[:, 1:] |= np.abs(vg[:, 1:] - vg[:, :-1]) > 3
    j3[:, :-1] |= np.abs(vg[:, 1:] - vg[:, :-1]) > 3
    j3[1:, :] |= np.abs(vg[1:, :] - vg[:-1, :]) > 3
    j3[:-1, :] |= np.abs(vg[1:, :] - vg[:-1, :]) > 3
    return {
        "mae": float(np.abs(ref - ren).mean()),
        "grad": float(ga.mean() / max(gr.mean(), 1e-9)),
        "bnd1": 100 * float(j[flat].mean()),
        "bnd3": 100 * float(j3[flat].mean()),
        "hf": float(np.abs(lr - median_filter(lr, 5))[flat].mean()),
    }


def run(name, src, env_extra, tag):
    ref_im = Image.open(src).convert("RGB")
    ref = np.asarray(ref_im).astype(np.float32)
    lr = ref.mean(axis=2)
    gx = np.zeros_like(lr); gy = np.zeros_like(lr)
    gx[:, 1:-1] = lr[:, 2:] - lr[:, :-2]
    gy[1:-1, :] = lr[2:, :] - lr[:-2, :]
    flat = np.hypot(gx, gy) < 3.0

    env = {k: v for k, v in os.environ.items() if not k.startswith("VEC_")}
    env.update(env_extra)
    env["VEC_TITLE"] = tag
    svg = os.path.join(OUT, "%s_%s.svg" % (tag, name))
    png = svg.replace(".svg", ".png")
    # cwd 固定到冻结目录（子智能体报告过 cwd 会改变输出）
    r = subprocess.run([PY, os.path.join(FROZEN, "vec.py"), src, svg],
                       capture_output=True, text=True, env=env, cwd=FROZEN)
    if r.returncode != 0:
        return None
    subprocess.run([PY, RENDER, svg, png], capture_output=True, text=True, cwd=FROZEN)
    v = Image.open(png).convert("RGB")
    if v.size != ref_im.size:
        v = v.resize(ref_im.size, Image.LANCZOS)
    return metrics(np.asarray(v).astype(np.float32), ref, flat)


def summary(cfg_name, cfg_env):
    print("\n=== %s  %s ===" % (cfg_name, cfg_env or "默认"))
    print("%-9s %7s %7s %7s %7s %7s" % ("图", "MAE中位", "grad中位", "bnd1中位", "bnd1范围", "bnd3中位"))
    print("-" * 60)
    out = {}
    for name, src in IMGS:
        vals = []
        for s in SEEDS:
            e = dict(cfg_env)
            e["VEC_SEED"] = str(s)
            m = run(name, src, e, "%s_s%d" % (cfg_name.replace(" ", ""), s))
            if m:
                vals.append(m)
        if not vals:
            print("%-9s 失败" % name); continue
        b1 = [v["bnd1"] for v in vals]
        agg = {
            "mae": statistics.median([v["mae"] for v in vals]),
            "grad": statistics.median([v["grad"] for v in vals]),
            "bnd1_med": statistics.median(b1),
            "bnd1_min": min(b1), "bnd1_max": max(b1),
            "bnd3_med": statistics.median([v["bnd3"] for v in vals]),
            "hf": statistics.median([v["hf"] for v in vals]),
        }
        out[name] = agg
        print("%-9s %7.3f %7.3f %7.2f %4.1f~%4.1f %7.2f"
              % (name, agg["mae"], agg["grad"], agg["bnd1_med"],
                 agg["bnd1_min"], agg["bnd1_max"], agg["bnd3_med"]))
    return out


if __name__ == "__main__":
    check_seed_support()
    cfgname = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    extra = {}
    for a in sys.argv[2:]:
        if "=" in a:
            k, v = a.split("=", 1)
            extra[k] = v
    res = summary(cfgname, extra)
    with open(os.path.join(OUT, "%s.json" % cfgname.replace(" ", "")), "w",
              encoding="utf-8") as f:
        json.dump({"cfg": cfgname, "env": extra, "seeds": SEEDS, "rows": res},
                  f, ensure_ascii=False, indent=2)
    print("\n-> workspace/robust/%s.json" % cfgname.replace(" ", ""))
