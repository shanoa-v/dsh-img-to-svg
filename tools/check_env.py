"""check_env.py -- 环境自检

脱离 agent 使用时，先跑一次这个，确认缺什么：
    python check_env.py

检查项：
  1. Python 版本（需 3.10+）
  2. 第三方依赖（numpy / Pillow / scikit-learn / scikit-image）
  3. 关键文件是否齐全（scripts/vec.py、tools/watch_draw.py 等）
  4. 路径解析方式（自身推导 or DSH_HOME 后备）
"""
import importlib
import os
import sys

THIS = os.path.dirname(os.path.abspath(__file__))
SKILL = os.path.dirname(THIS)
SCRIPTS = os.path.join(SKILL, "scripts")

print("=" * 60)
print("dsh-img-to-svg 环境自检")
print("=" * 60)

# 1. Python 版本
v = sys.version_info
ok_py = v >= (3, 10)
print(f"\n[1] Python {v.major}.{v.minor}.{v.micro}  "
      f"{'OK' if ok_py else '过低，需要 3.10+'}")
print(f"    解释器: {sys.executable}")

# 2. 依赖
print("\n[2] 第三方依赖")
DEPS = [("numpy", "numpy"), ("PIL", "Pillow"),
        ("sklearn", "scikit-learn"), ("skimage", "scikit-image"),
        ("cairosvg", "cairosvg（SVG 渲染，缺了逐层帧会空白）")]
missing = []
for mod, pkg in DEPS:
    try:
        importlib.import_module(mod)
        print(f"    {pkg:<18} OK")
    except Exception:
        print(f"    {pkg:<18} 缺失")
        missing.append(pkg)

# 3. 关键文件
print("\n[3] 关键文件")
NEED = ["scripts/vec.py", "scripts/render_check.py",
        "scripts/block_tracer.py", "tools/watch_draw.py",
        "tools/webui.py", "requirements.txt"]
bad = []
for rel in NEED:
    p = os.path.join(SKILL, rel)
    e = os.path.exists(p)
    print(f"    {rel:<32} {'OK' if e else '缺失'}")
    if not e:
        bad.append(rel)

# 4. 路径解析
print("\n[4] 路径解析")
self_ok = os.path.isdir(os.path.join(SKILL, "scripts"))
print(f"    技能根目录: {SKILL}")
print(f"    自身推导:   {'可用' if self_ok else '不可用'}")
print(f"    DSH_HOME:   {os.environ.get('DSH_HOME') or '(未设置)'}")

# 结论
print("\n" + "=" * 60)
if not ok_py or missing or bad:
    print("结论: 需要修复")
    if not ok_py:
        print(f"  - Python 版本过低（当前 {v.major}.{v.minor}）")
    if missing:
        print(f"  - 缺依赖: {', '.join(missing)}")
        print(f"    安装: \"{sys.executable}\" -m pip install -r "
              f"\"{os.path.join(SKILL, 'requirements.txt')}\"")
    if bad:
        print(f"  - 缺文件: {', '.join(bad)}")
else:
    print("结论: 环境完整，可直接使用")
    print("  启动网页界面: 双击 tools/启动网页界面.bat")
print("=" * 60)
