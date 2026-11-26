"""_pickdir.py -- 弹出 Windows 原生"选择文件夹"对话框（资源管理器风格）

## 为什么单独一个进程

tkinter 的对话框必须在拥有 Tk 的线程里跑。webui 的 HTTP 请求在子线程，
直接在子线程里建 Tk 会偶发崩溃，所以用独立短进程：
    webui  ->  subprocess  ->  本脚本  ->  原生对话框
结果通过 stdout 的 JSON 返回，避免中文路径编码问题（ensure_ascii=False 但
读取端按 utf-8 解码）。

## 用法

    python _pickdir.py [初始目录]

输出（stdout 单行 JSON）：

    {"path": "F:\\out"}     选中
    {"path": null}          取消
"""
import json
import os
import sys


def _enable_dpi_awareness():
    """声明进程 DPI 感知，必须在创建任何窗口之前调用。

    实测本机进程默认 Unaware（=0），导致对话框按 96 DPI 渲染后被系统
    位图拉伸，文字发虚。这里按优先级尝试三种声明方式。
    """
    if os.name != "nt":
        return
    try:
        import ctypes
    except Exception:
        return

    # 1) Per-Monitor V2（Windows 10 1703+，效果最好）
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return
    except Exception:
        pass

    # 2) Per-Monitor（Windows 8.1+）
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
        return
    except Exception:
        pass

    # 3) 系统级（Vista+ 都有）
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def _tune_tk_scaling(root):
    """让 Tk 按真实 DPI 缩放控件尺寸（否则控件会偏小）。"""
    try:
        import ctypes
        dpi = ctypes.windll.user32.GetDpiForSystem()
        if dpi and dpi > 0:
            root.tk.call("tk", "scaling", dpi / 72.0)
    except Exception:
        pass


def main():
    initial = sys.argv[1] if len(sys.argv) > 1 else ""

    # 关键：在 import tkinter / 创建 Tk 之前声明感知
    _enable_dpi_awareness()

    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as e:
        print(json.dumps({"path": None, "err": f"tkinter 不可用: {e}"},
                         ensure_ascii=False))
        return 0

    root = tk.Tk()
    _tune_tk_scaling(root)
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass

    start = initial
    if not start or not os.path.exists(start):
        start = os.getcwd()

    try:
        p = filedialog.askdirectory(
            parent=root, initialdir=start, mustexist=True,
            title="选择输出目录")
    except Exception as e:
        print(json.dumps({"path": None, "err": str(e)},
                         ensure_ascii=False))
        try:
            root.destroy()
        except Exception:
            pass
        return 0

    try:
        root.destroy()
    except Exception:
        pass

    print(json.dumps({"path": p or None}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
