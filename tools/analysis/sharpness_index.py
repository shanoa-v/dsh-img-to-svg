"""sharpness_index.py -- 锐度指标（与 MAE 不同维度）

为什么需要这个：MAE 奖励平滑。一张糊掉但"整体色差小"的图 MAE 很好，
这正是嘴部丢失却被 MAE 判为"可接受"的原因。要往锐化走，必须先能测锐度。

指标定义：
  gradE    全图平均梯度能量（Sobel 幅值均值）→ 越高越锐
  grad_ratio  gradE(渲染) / gradE(源图)  → 1.0 表示锐度与源图一致
              <1 糊了，>1 过锐/有伪影
  edge_w    边缘过渡宽度（沿强边缘法向的 10%-90% 上升距离）
              越小越锐利；用简化代理：强边缘像素占其邻域的比例
  hf_all    全图高频能量（不限于平坦区，与现有 hfKept 区分）
"""
import os
import sys
import json
import subprocess
import numpy as np
from PIL import Image
from scipy.ndimage import sobel, median_filter, label

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
SK = os.path.join(os.environ["DSH_HOME"], "skills", "dsh-img-to-svg")
OUT = os.path.join(ROOT, "workspace", "sharp")
os.makedirs(OUT, exist_ok=True)

IMGS = [
    ("mouth", os.path.join(ROOT, "input", "collections", "batch3",
     "illust_100531443_20260808_134010.png")),
    ("portrait", os.path.join(ROOT, "input", "collections", "batch3",
     "illust_88927530_20260808_134042.png")),
    ("wuther", os.path.join(ROOT, "input", "collections", "batch1", "c_wuthering.png")),
    ("user", os.path.join(ROOT, "input", "user", "user_char.png")),
]


def metrics(a, ref):
    """a=渲染, ref=源图, 都是 float32 RGB"""
    ga = np.hypot(sobel(a.mean(axis=2), 0), sobel(a.mean(axis=2), 1))
    gr = np.hypot(sobel(ref.mean(axis=2), 0), sobel(ref.mean(axis=2), 1))
    gradE_a = float(ga.mean())
    gradE_r = float(gr.mean())
    hf_a = float(np.abs(a.mean(axis=2) - median_filter(a.mean(axis=2), 5)).mean())
    hf_r = float(np.abs(ref.mean(axis=2) - median_filter(ref.mean(axis=2), 5)).mean())
    mae = float(np.abs(ref - a).mean())
    return {
        "gradE": round(gradE_a, 2),
        "grad_ratio": round(gradE_a / max(gradE_r, 1e-6), 3),
        "hf_all": round(hf_a, 3),
        "hf_ratio": round(hf_a / max(hf_r, 1e-6), 3),
        "mae": round(mae, 3),
    }


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    extra = {}
    for a in sys.argv[2:]:
        if "=" in a:
            k, v = a.split("=", 1)
            extra[k] = v
    print("=== 锐度评估  tag=%s  参数=%s ===" % (tag, extra or "默认"))
    print("%-10s %8s %9s %8s %9s %7s"
          % ("图", "gradE", "grad比值", "hf", "hf比值", "MAE"))
    print("-" * 58)
    rows = {}
    for name, src in IMGS:
        if not os.path.exists(src):
            continue
        ref_im = Image.open(src).convert("RGB")
        ref = np.asarray(ref_im).astype(np.float32)
        env = {k: v for k, v in os.environ.items() if not k.startswith("VEC_")}
        env.update(extra)
        svg = os.path.join(OUT, "%s_%s.svg" % (tag, name))
        png = svg.replace(".svg", ".png")
        r = subprocess.run([PY, os.path.join(SK, "scripts", "vec.py"), src, svg],
                           capture_output=True, text=True, env=env)
        if r.returncode != 0:
            print("%-10s 失败" % name); continue
        subprocess.run([PY, os.path.join(SK, "scripts", "render_check.py"), svg, png],
                       capture_output=True, text=True)
        v = Image.open(png).convert("RGB")
        if v.size != ref_im.size:
            v = v.resize(ref_im.size, Image.LANCZOS)
        m = metrics(np.asarray(v).astype(np.float32), ref)
        rows[name] = m
        print("%-10s %8.2f %9.3f %8.3f %9.3f %7.3f"
              % (name, m["gradE"], m["grad_ratio"], m["hf_all"],
                 m["hf_ratio"], m["mae"]))
    with open(os.path.join(OUT, "%s.json" % tag), "w", encoding="utf-8") as f:
        json.dump({"tag": tag, "params": extra, "rows": rows}, f,
                  ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
