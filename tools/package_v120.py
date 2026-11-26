"""package_v120.py -- 打包 v1.2.0 成果

## 内容

  final_v1.2.0/梅原生/
    svg/   95 个（矢量文件）
    png/   95 个（渲染校验，体积大，压缩时排除）
    cmp/   95 个（三联对比图）
    _summary.json  _rows.jsonl
    README.md

## 体积

SVG 约 6.1 GB。为控制压缩包体积，PNG 校验图不打包（可由 SVG 重新渲染）。

## 用法

    python package_v120.py                 # 默认输出（已存在则拒绝）
    python package_v120.py --out <路径>     # 自定义输出
    python package_v120.py --force          # 明确允许覆盖

**安全性**：本脚本会写入一个 1.8 GB 级的成品包。早期版本用
`ZipFile(OUT, "w")` 直接覆盖同名目标，**无任何检查**——任何人无参跑一次
就会静默销毁上一份成品包（实测中断后 zip 中央目录未写完，
`testzip()` 报 `BadZipFile: File is not a zip file`，**原始字节不可复原**）。
因此现在默认**拒绝覆盖已存在的目标**，必须显式 `--force`。
"""
import argparse
import os
import zipfile
import hashlib
import time
import json

# 默认取「技能目录的上两级」，而不是硬编码作者机器上的路径。
# 可用环境变量 DSH_IMGSVG_ROOT 覆盖。
ROOT = os.environ.get("DSH_IMGSVG_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", ".."))
BASE = os.path.join(ROOT, "output", "final_v1.2.0", "梅原生")
DEF_OUT = os.path.join(ROOT, "final_v1.2.0_梅原生_全量重绘_20251027.zip")
RES = os.path.join(ROOT, "output", "research_v1.2.0")


def main():
    ap = argparse.ArgumentParser(
        description="打包 v1.2.0 成果（约 1.8 GB）")
    ap.add_argument("--out", default=DEF_OUT,
                    help="输出 zip 路径（默认 %(default)s）")
    ap.add_argument("--force", action="store_true",
                    help="允许覆盖已存在的目标（默认拒绝，防误删成品包）")
    args = ap.parse_args()
    OUT = args.out

    # ---- 覆盖保护 ----
    # 早期版本无此检查，无参跑一次就会把上一份 1.79 GB 成品包截断覆盖。
    # 中断后 zip 中央目录未写完 -> testzip 报 BadZipFile，原始字节不可复原。
    if os.path.exists(OUT) and not args.force:
        sz = os.path.getsize(OUT) / 1073741824
        raise SystemExit(
            "目标已存在：%s（%.2f GB）\n"
            "本脚本会**覆盖**它，且一旦中断旧文件即不可恢复。\n"
            "如确认要覆盖，请加 --force；或改用 --out 指定别的路径。"
            % (OUT, sz))

    t0 = time.time()
    n = 0
    total_in = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        # 文档
        for doc in ("README.md", "_summary.json"):
            p = os.path.join(BASE, doc)
            if os.path.exists(p):
                z.write(p, doc)
                n += 1
        # 研究
        if os.path.isdir(RES):
            for f in os.listdir(RES):
                p = os.path.join(RES, f)
                if os.path.isfile(p):
                    z.write(p, "research/" + f)
                    n += 1
        # SVG + 对比图
        for sub in ("svg", "cmp"):
            d = os.path.join(BASE, sub)
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                p = os.path.join(d, f)
                if not os.path.isfile(p):
                    continue
                z.write(p, f"{sub}/{f}")
                total_in += os.path.getsize(p)
                n += 1
                if n % 50 == 0:
                    print(f"  已打包 {n} 个文件，用时 {time.time() - t0:.0f}s",
                          flush=True)
    mb = os.path.getsize(OUT) / 1048576
    print()
    print(f"文件 {n} 个")
    print(f"原始 {total_in / 1024:.0f} MB  压缩后 {mb:.0f} MB")
    print(f"SHA256 {hashlib.sha256(open(OUT, 'rb').read()).hexdigest()[:16]}")
    print(f"路径 {OUT}")
    print(f"耗时 {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
