"""lint_locks.py -- 扫描"持锁调用会再取锁的函数"（防死锁守卫）

## 为什么需要它

本项目历史上出现过 4 次同一类死锁：
`with LOCK:` 块内调用了 `log()`，而 `log()` 内部也 `with LOCK:`。
`threading.Lock` 不可重入，同线程二次获取 -> 永久死锁 -> 服务冻结。

修一处很可能漏另一处，所以扫全库。

## 实现要点（v3）

用 **AST 遍历**，不用行文本正则。原因：文本扫描会把注释/文档字符串里
提到 `log(` 的地方误报成危险点（本脚本 v1 就误报了自己文档里的示例）。

**v3 相比 v2 修了两个漏检**（第 1 轮独立审计实测发现）：

1. **v2 只认 `LOCK`，不认 `_RL`**
   `watch_draw.py:91` 定义的是 `_RL = threading.Lock()`，
   而 `_record_beat` / `_record_bye` / `_watchdog` / `run_pipeline` 都在用它。
   v2 对"持 `_RL` 调取 `_RL` 的函数"这类死锁**完全看不见**。
   → v3 用 `LOCK_NAMES` 集合，默认 `{"LOCK", "_RL", ...}`，也支持 `--lock` 追加。

2. **v2 没有传递闭包**
   `with LOCK: a()` 而 `a()` 里调 `b()`、`b()` 才 `with LOCK` —— v2 判 a 不在
   `lockfns` 里就放过了。→ v3 用迭到不动点的传递闭包。

## 自检

自检样本现在覆盖 4 类（v2 只有 2 类，所以"自检 PASS"给了虚假安全感）：
- POS_DIRECT   持锁调 log
- POS_RL       持 `_RL` 调取 `_RL` 的函数
- POS_CLOSURE  传递闭包（with LOCK -> a -> b -> with LOCK）
- NEG           不持锁调 log

## 用法
    python tools/lint_locks.py [目录] [--lock NAME ...]
退出码 0 = 干净，1 = 有问题，2 = 自检失败
"""
import ast
import os
import sys

# 本项目实际用到的锁名。加新锁时在这里登记，或命令行 --lock 追加。
LOCK_NAMES = {"LOCK", "_RL", "_LOCK", "RL", "LOCK2", "_LOCK2"}

# 已知的"内部会取锁"的函数名（补漏：有些函数名看不出来但确实取锁）
KNOWN_LOCK_TAKERS = "log"


def _is_lock_with(node):
    """判断 with 的 context 是不是某把锁。

    支持 with LOCK / with _RL / with LOCK as l / with self.LOCK。
    """
    if not isinstance(node, ast.With):
        return False
    for item in node.items:
        e = item.context_expr
        if isinstance(e, ast.Name) and e.id in LOCK_NAMES:
            return True
        if isinstance(e, ast.Attribute) and e.attr in LOCK_NAMES:
            return True
    return False


def _own_nodes(fnode):
    """遍历函数 node 的子树，但**不进入嵌套函数/类定义体**。

    否则外层函数会把内层函数里的 `with LOCK` 误算成自己的。
    """
    stack = list(ast.iter_child_nodes(fnode))
    while stack:
        n = stack.pop()
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                          ast.Lambda, ast.ClassDef)):
            continue
        yield n
        stack.extend(ast.iter_child_nodes(n))


def _callee_name(call):
    f = call.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def _lock_functions(tree):
    """返回"调用它就会取锁"的函数名集合（含传递闭包）。

    迭代到不动点：with LOCK: a() 而 a() 调 b()、b() 才 with LOCK，
    也要把 a 算进去。
    """
    direct = set()
    calls = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body_nodes = list(_own_nodes(node))
        if any(_is_lock_with(n) for n in body_nodes):
            direct.add(node.name)
        cs = set()
        for n in body_nodes:
            if isinstance(n, ast.Call):
                cn = _callee_name(n)
                if cn:
                    cs.add(cn)
        calls.setdefault(node.name, set()).update(cs)

    lockfns = set(direct)
    changed = True
    while changed:
        changed = False
        for fn, cs in calls.items():
            if fn in lockfns:
                continue
            if cs & lockfns:
                lockfns.add(fn)
                changed = True
    return lockfns


def scan_source(text, label=""):
    """返回 [(行号, 描述)]"""
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        return [(getattr(e, "lineno", 0), f"语法错误，跳过: {e}")]

    lockfns = _lock_functions(tree)
    hits = []
    for node in ast.walk(tree):
        if not _is_lock_with(node):
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                nm = _callee_name(child)
                if nm in KNOWN_LOCK_TAKERS:
                    hits.append((child.lineno,
                                 f"持锁直接调 {nm}() -> 会二次取锁"))
                elif nm and nm in lockfns:
                    hits.append((child.lineno,
                                 f"持锁调 {nm}() -> 其内部（可能是间接地）取锁"))
    # 同位置去重
    seen = set()
    out = []
    for ln, msg in hits:
        if (ln, msg) not in seen:
            seen.add((ln, msg))
            out.append((ln, msg))
    return sorted(out)


def scan_file(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            txt = fh.read()
    except OSError:
        return []
    return scan_source(txt)


# ---------------------------------------------------------------- 自检

POS_DIRECT = """import threading
LOCK=threading.Lock()
def log(m):
    with LOCK:
        pass
def bad():
    with LOCK:
        log('x')
"""

POS_RL = """import threading
_RL=threading.Lock()
def rec(m):
    with _RL:
        pass
def bad2():
    with _RL:
        rec('x')
"""

POS_CLOSURE = """import threading
LOCK=threading.Lock()
def inner():
    with LOCK:
        pass
def mid():
    inner()
def outer():
    with LOCK:
        mid()
"""

NEG = """import threading
LOCK=threading.Lock()
def log(m):
    with LOCK:
        pass
def good():
    with LOCK:
        pass
    log('x')
"""

NEG_NESTED = """import threading
LOCK=threading.Lock()
def outer():
    def inner2():
        with LOCK:
            pass
    pass
"""


def _selfcheck():
    """全部通过返回 True。任一项不达标说明扫描器不可信。"""
    cases = [
        ("POS_DIRECT", POS_DIRECT, True),
        ("POS_RL", POS_RL, True),
        ("POS_CLOSURE", POS_CLOSURE, True),
        ("NEG", NEG, False),
        ("NEG_NESTED", NEG_NESTED, False),
    ]
    all_ok = True
    print("自检：")
    for name, src, expect_hit in cases:
        n = len(scan_source(src))
        want = ">0" if expect_hit else "=0"
        ok = (n > 0) if expect_hit else (n == 0)
        flag = "OK" if ok else "FAIL"
        print(f"  {name:<12} 命中 {n} 处（应 {want}）-> {flag}")
        if not ok:
            all_ok = False
    return all_ok


def main():
    args = [a for a in sys.argv[1:]]
    # --lock NAME 可追加自定义锁名
    i = 0
    roots = []
    while i < len(args):
        if args[i] == "--lock" and i + 1 < len(args):
            LOCK_NAMES.add(args[i + 1])
            i += 2
            continue
        roots.append(args[i])
        i += 1

    if not roots:
        roots = [os.path.dirname(os.path.abspath(__file__))]

    if not _selfcheck():
        print()
        print("!! 扫描器自检失败，结果无意义。请先修扫描器。")
        return 2

    print()
    total = 0
    scanned = 0
    for root in roots:
        if os.path.isfile(root):
            paths = [root]
        else:
            paths = []
            for dp, _, fns in os.walk(root):
                for fn in sorted(fns):
                    if fn.endswith(".py"):
                        paths.append(os.path.join(dp, fn))
        for p in paths:
            scanned += 1
            hits = scan_file(p)
            if hits:
                print(f"{p}")
                for ln, msg in hits:
                    print(f"  L{ln}: {msg}")
                total += len(hits)

    print()
    print(f"已扫描 {scanned} 个 .py 文件")
    if total:
        print(f"发现 {total} 处持锁调用危险点")
        return 1
    print("未发现持锁调用危险点")
    return 0


if __name__ == "__main__":
    sys.exit(main())
