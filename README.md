# dsh-img-to-svg

把位图（插画 / 海报 / 立绘 / 截图 / 照片）复刻成分层矢量 SVG，追求最大化视觉还原度，
附带实时预览、定量相似度评估与差异热力图诊断。

> **本项目仅为个人娱乐产物。**
> 它源于作者自己想把一堆图片转成 SVG 的私心，不是产品，没有路线图，没有支持承诺。
> **可能会较长时间不更新**，也可能会突然大改。请把它当作「一份能跑的个人脚本」看待，
> 而不是「一个维护中的开源项目」。欢迎自取使用，但不建议用于任何有交付时限的场合。

---

## 它做什么

输入一张图，输出一份 SVG。不是简单的「描边 + 填色」，而是：

1. **平滑** —— 保边滤波压掉噪声，避免把噪点描成碎屑
2. **量化** —— 按色阶聚类，把连续色调压成有限色板
3. **分层描摹** —— 每个色块独立追轮廓，按面积/层次排序，逐层堆叠
4. **权威出图** —— 由 `vec.py` 统一生成最终 SVG，保证「看到的过程」与「交付的文件」同源

技法细节记录在 `references/` 下（算法、参数、踩坑、各阶段技术报告）。

---

## 快速上手

### 1. 装依赖

```bash
python -m pip install -r requirements.txt
```

要求 Python ≥ 3.9。核心依赖是 `numpy` / `Pillow` / `scikit-image` / `scikit-learn` / `scipy`，
渲染校验额外用 `cairosvg`（不满意可以跳过那一步，直接用浏览器看 SVG）。

> 若 `pip` 卡住，试试镜像：
> `pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com`

### 2. 自检环境

```bash
python tools/check_env.py
```

会逐项报告缺哪个包。

### 3. 转一张图

**命令行（最简）**

`vec.py` 的两个位置参数是 **输入图** 和 **输出 SVG**，调参走环境变量：

```bash
python scripts/vec.py 你的图.png 输出.svg
```

想改参数就加环境变量（Windows PowerShell 用 `$env:VEC_N="32"`，cmd 用 `set VEC_N=32`）：

| 变量 | 默认 | 作用 |
|---|---|---|
| `VEC_N` | 56 | 色数 |
| `VEC_EPS` | 0.30 | RDP 简化容差，越大越省节点也越糙 |
| `VEC_MINAREA` | 5 | 小于该面积的色块丢弃 |
| `VEC_EDGE` | 0.16 | 边缘保留强度 |
| `VEC_W` | 0 | 工作宽度，0 表示按原尺寸 |

Linux / macOS / Git Bash：

```bash
VEC_N=32 VEC_EPS=0.18 python scripts/vec.py 你的图.png 输出.svg
```

PowerShell：

```powershell
$env:VEC_N="32"; $env:VEC_EPS="0.18"
python scripts/vec.py 你的图.png 输出.svg
```

> 注：`tools/` 下另有一批带 `--colors` 这类命令行开关的封装脚本
> （`max_quality.py`、`quality_boost.py`、`redraw_meiyuan.py` 等），
> 它们内部会转成 `VEC_*` 环境变量再调 `vec.py`。用哪个都行。

**网页控制台（推荐，可以看绘制过程）**

```bash
python tools/webui.py
```

浏览器会自动打开。左侧选档位与参数，右侧选图，点「开始」。
单张模式会实时显示分层绘制动画；批量模式后台跑，不占前台。

**图形界面（tkinter，不依赖浏览器）**

```bash
python tools/gui.py
```

**双击启动（Windows）**

`tools/启动网页界面.bat` 或 `tools/启动图形界面.bat`。
启动器会依次探测几个候选解释器，选第一个能 `import` 齐依赖的。

---

## 四个档位

| 档位 | 名字 | 说明 | 典型用途 |
|---|---|---|---|
| 1 | 快速 | `k=1`，一倍矢量化，看过程 | 日常转图，几十秒 |
| 2 | 高质量 | 更大工作尺寸 + 更多色数 | 成品交付 |
| 3 | 局部重绘 | 针对特定区域加强 | 局部糊的地方 |
| 4 | 上采样 | 先放大 `k` 倍再矢量化，不计时间 | 极致还原，可能几十分钟 |

`k` 是放大倍数。档位 4 的意义是「让模型自己看图决定 k」，所以参数留给你/agent 判断。
网页控制台里 `k` 可以直接下拉选。

想知道某个 `k` 会跑出多大工作量，先预演：

```bash
python tools/watch_draw.py 你的图.png --plan
```

只读图尺寸做算术，0.5 秒返回一张 1..8 的对照表，不启服务、不写文件。

---

## 参数怎么选

`--colors`（色数 N）和 `--edge`（边缘保留强度）是最影响观感的两个：

| 图类型 | N | EDGE |
|---|---|---|
| 粉彩 / 柔和厚涂 | 56–64 | 0.08–0.12 |
| 一般插画 | 48–80 | 0.16 |
| 线稿 / 高对比 | 32–56 | 0.30–0.50 |

判断依据不是感觉，而是先量四个统计量（饱和度分布、亮度中位数、梯度均值、最高单色占比）。
`tools/dataset_profile.py` 可以帮你出这些数。

细节与实测数据见 `references/parameters.md`。

---

## 交付前必做

**不要只信相似度数字。** 这个项目里出现过多次「相似度 99% 但小字全糊」的情况。

1. **渲染校验** —— `python scripts/render_check.py 输出.svg`
2. **定量评估** —— `python scripts/compare.py 输出.svg 标签 原图.png`

   `compare.py` 要 **3 个参数**：`<svg>` `<tag>` `<参考图>`。
   中间那个 `tag` 是个短标签，只用于给差异图命名。
   它会输出 MAE / RMSE / PSNR，并生成差异热力图。
3. **目视确认** —— 裁几个关键区域放大 2–3 倍看：主标题笔画断不断、logo 小字能不能读、
   人脸五官有没有糊、最小字号处（版权行、角标）——**这里通常会糊，该说就说**

`references/pitfalls.md` 里记了 30+ 条实测踩过的坑。

---

## 目录结构

```
.
├── SKILL.md                  # 完整使用文档（也很长，等同手册）
├── requirements.txt
├── scripts/
│   ├── vec.py                # 核心转换器（权威出图）
│   ├── block_tracer.py       # 按连通块描摹
│   ├── live_vec_server.py    # 实时预览服务
│   ├── render_check.py       # 渲染校验
│   └── compare.py            # 相似度评估
├── tools/
│   ├── webui.py              # 网页控制台（主力界面）
│   ├── gui.py                # tkinter 图形界面
│   ├── watch_draw.py         # 逐步回放绘制过程
│   ├── max_quality.py        # 档位 2
│   ├── quality_boost.py      # 档位 3
│   ├── redraw_meiyuan.py     # 档位 4
│   └── ...                   # 诊断、评估、批量、打包等辅助脚本
├── references/               # 算法与实测报告
└── assets/                   # 示例与页面备注
```

---

## 环境变量

改造后不再依赖作者本机路径。需要覆盖时：

| 变量 | 作用 | 默认 |
|---|---|---|
| `IMGSVG_ROOT` | 工作区根（输出、素材、日志的基准目录） | 仓库的上一级 |
| `IMGSVG_PYTHON` | 子进程调用的解释器 | 当前解释器 |

---

## 已知限制

这份代码是「个人用着顺手」的产物，有些地方不完美，说在前面：

- **不是所有脚本都能 `import`。** `tools/` 下有几个是纯 CLI 脚本（`dataset_profile.py`、
  `validate_target_width.py`、`verify_seam_bleed.py` 等 12 个），它们在模块顶层就干活，
  `import` 会直接开始跑。**请用 `python tools/xxx.py` 调用**，不要 import。
- **只测过 Windows。** 路径处理、`.bat` 启动器、DPI 相关都有 Windows 假定。
  Linux/macOS 理论上能跑核心的 `vec.py`，但没验证过。
- **`cairosvg` 在 Windows 上偶发找不到 cairo 库。** 修法见 `requirements.txt` 末尾的疑难节；
  实在不行跳过渲染校验，浏览器看 SVG 一样。
- **大图吃内存。** 1920×1080 分层描摹峰值约 2–3 GB；档位 4 开 `k=8` 能到几千万像素，慎用。
- **色块接缝。** 相邻色块各自描边会产生亚像素渗色，已用「同色细描边 0.6」缓解，
  详见 `references/SEAM_BLEED_INVESTIGATION.md`。
- **文字会变成路径。** 不可再编辑，这是矢量化本身的代价。

---

## 许可

MIT。见 [LICENSE](LICENSE)。

---

## 致谢与免责

技法上受启发于 potrace、vtracer 等成熟工具的思路，但这个实现是独立写的，
针对「插画/立绘」这个特定场景做了取舍，不追求通用位图矢量化。

本项目按「现状」提供，不附带任何担保。用它处理他人作品前请自行确认授权。
