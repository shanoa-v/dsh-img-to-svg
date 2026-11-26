"""stratified_sample.py -- 分层抽样：覆盖尺寸 × 复杂度两个维度

## 为什么分层

530 张图尺寸跨度 0.15 ~ 58.82 MP（580 倍），复杂度（8级色数）41~407（10 倍）。
简单随机抽样会漏掉极端值 —— 而极端值正是失败模式所在。

## 分层维度

  尺寸：small(<0.5MP) / mid(0.5-2MP) / large(2-6MP) / huge(>6MP)
  复杂度：简单(<120) / 中(120-200) / 复杂(>200)

## 抽样规则

每个 (尺寸层 × 复杂度层) 格子最多抽 2 张，保证覆盖全谱；
再补足到目标数量，用组内随机。

## 输出

  output/dataset_profile/sample.json   抽样清单
"""
import os
import json
import numpy as np

def _find_root():

    """工作区根：优先环境变量，否则取仓库的上一级。



    原实现写死作者机器上的 F:\\SVG drawing。

    """

    env = os.environ.get("IMGSVG_ROOT")

    if env and os.path.isdir(env):

        return env

    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))





ROOT = _find_root()
OUT = os.path.join(ROOT, "output", "dataset_profile")
PROFILE = os.path.join(OUT, "profile.json")

SIZE_BINS = [(0, 0.5, "small"), (0.5, 2.0, "mid"), (2.0, 6.0, "large"), (6.0, 1e9, "huge")]
COL_BINS = [(0, 120, "simple"), (120, 200, "mid"), (200, 1e9, "complex")]


def size_bin(mp):
    for lo, hi, name in SIZE_BINS:
        if lo <= mp < hi:
            return name
    return "huge"


def col_bin(c):
    for lo, hi, name in COL_BINS:
        if lo <= c < hi:
            return name
    return "complex"


prof = json.load(open(PROFILE, encoding="utf-8"))
rng = np.random.default_rng(20251026)

# 目标：每组按格子抽，总额控制在 60 张左右（够统计又不至于跑太久）
TARGET_PER_GROUP = {"Noyu": 42, "pixiv": 80, "梅原生": 60}   # Noyu 全量（只有42张）

sample = []
for gname, recs in prof.items():
    if not recs:
        continue
    # 分格子
    cells = {}
    for r in recs:
        key = (size_bin(r["mp"]), col_bin(r["colors8"]))
        cells.setdefault(key, []).append(r)

    picked = set()
    target = TARGET_PER_GROUP.get(gname, 40)

    # 第一轮：每格最多 2 张（优先最极端的）
    for key in sorted(cells.keys()):
        group = sorted(cells[key], key=lambda r: -r["colors8"])
        take = group[:2] if len(group) >= 2 else group
        for r in take:
            picked.add(r["file"])

    # 第二轮：补足到目标数量（随机）
    if len(picked) < target:
        rest = [r for r in recs if r["file"] not in picked]
        rng.shuffle(rest)
        for r in rest[:target - len(picked)]:
            picked.add(r["file"])

    for r in recs:
        if r["file"] in picked:
            sample.append({
                "group": gname, "file": r["file"],
                "path": os.path.join("input", "new_train", gname, r["file"]),
                "w": r["w"], "h": r["h"], "mp": r["mp"],
                "colors8": r["colors8"],
                "size_bin": size_bin(r["mp"]),
                "col_bin": col_bin(r["colors8"]),
                "flat_pct": r["flat_pct"], "detail_pct": r["detail_pct"],
            })

with open(os.path.join(OUT, "sample.json"), "w", encoding="utf-8") as f:
    json.dump(sample, f, ensure_ascii=False, indent=1)

# 打印覆盖情况
print("=== 抽样结果 %d 张 ===" % len(sample))
print("%-10s %6s %8s %8s %8s %8s"
      % ("组", "数量", "small", "mid", "large", "huge"))
for g in ("Noyu", "pixiv", "梅原生"):
    sub = [s for s in sample if s["group"] == g]
    if not sub:
        continue
    cnt = {b: sum(1 for s in sub if s["size_bin"] == b)
           for b in ("small", "mid", "large", "huge")}
    print("%-10s %6d %8d %8d %8d %8d"
          % (g, len(sub), cnt["small"], cnt["mid"], cnt["large"], cnt["huge"]))

print("\n%-10s %9s %9s %9s" % ("组", "simple", "mid", "complex"))
for g in ("Noyu", "pixiv", "梅原生"):
    sub = [s for s in sample if s["group"] == g]
    if not sub:
        continue
    cnt = {b: sum(1 for s in sub if s["col_bin"] == b)
           for b in ("simple", "mid", "complex")}
    print("%-10s %9d %9d %9d" % (g, cnt["simple"], cnt["mid"], cnt["complex"]))

print("\n极值覆盖检查：")
mp_sorted = sorted(sample, key=lambda s: s["mp"])
print("  最小 %.2f MP  %s" % (mp_sorted[0]["mp"], mp_sorted[0]["file"][:36]))
print("  最大 %.2f MP  %s" % (mp_sorted[-1]["mp"], mp_sorted[-1]["file"][:36]))
col_sorted = sorted(sample, key=lambda s: s["colors8"])
print("  最简 %d 色   %s" % (col_sorted[0]["colors8"], col_sorted[0]["file"][:36]))
print("  最繁 %d 色   %s" % (col_sorted[-1]["colors8"], col_sorted[-1]["file"][:36]))

print("\n抽样清单 -> %s" % os.path.join(OUT, "sample.json"))
