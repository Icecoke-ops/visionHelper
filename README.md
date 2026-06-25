## visionHelper

> **项目号**：`VH-2026-001`
> **版本号**：`1.0.1`
> **作者**：IceCoke

visionHelper 是一个轻量级的视觉辅助工具集，围绕 YOLO 数据生产与训练流程，
把**视频抽帧、图片去重、数据标注统计、自动标注、标注清除、YOLO 数据集导出
与模型训练**等常用步骤整合到同一个图形界面中，同时也以稳定的命令行/API
形式独立提供，便于在脚本或 CI 中复用。

项目划分为两个互不依赖的子包：

- `scripts/`：纯命令行/编程接口的核心工具包。
  - 统一入口为 `python scripts/vh.py <subcommand> <action> [options]`，所有参数
    显式通过 `--` / `-` 传递，无位置参数、无列表展开。
  - 按流程组织为 `scripts/common/`、`scripts/images/`、`scripts/datasets/`、
    `scripts/train/`、`scripts/predict/`、`scripts/deploy/` 六个子包；
    `scripts/cli.py` 只做路由，具体业务逻辑与参数解析放在各子包模块中。
  - `scripts/api.py` 提供稳定的编程接口，方法体内懒加载重型依赖，保证
    `import scripts` 自身**零副作用、零重依赖**。
- `gui/`：基于 PyQt5 的桌面图形界面，通过 **子进程** 调用
  `python scripts/vh.py <subcommand> <action> ...` 执行后台任务，从而把
  `torch` / `ultralytics` 等重型依赖完全隔离在用户自选的 Python 解释器中。

> **设计约束**：`scripts` 与 `gui` 之间不存在反向依赖，`scripts` 可独立使用，
> 不引入任何 GUI 相关依赖；`gui` 也不直接 `import scripts.api`，避免把深度
> 学习栈拉进 GUI 主进程。

---

## 更新日志（自 v1.0.0 以来）

> 仓库尚未维护独立的 CHANGELOG 文件，以下变更基于工作区相对于
> `v1.0.0`（commit `462f99e`）的实际差异整理。

### 架构重构

- **统一三级 CLI 入口**：所有工具统一通过 `python scripts/vh.py <subcommand> <action>`
  调用，例如 `python scripts/vh.py images import`、`python scripts/vh.py datasets stats`。
  旧 `python -m scripts.<tool>` 风格已废弃。
- **按流程重组后端**：后端代码按数据生产流程整理为 `scripts/images/`、
  `scripts/datasets/`、`scripts/train/`、`scripts/predict/`、`scripts/deploy/`
  以及公共的 `scripts/common/`；`scripts/cli.py` 仅做路由，业务逻辑下沉到
  各子包模块。
- **`scripts/__init__.py` 与公共子包 `__init__.py` 零副作用**：禁止在 import
  时顺带拉起重依赖，避免 GUI 打包态下误触 `torch` / `cv2` 导致闪退。
- **`scripts/config.py`**：集中存放后端公共常量（任务类型、扩展名、CLI 协议
  标记、进度日志默认参数、shape_type 集合、优化器集合等），与 `gui/config.py`
  完全解耦。
- **`scripts/logging_utils.py`**：统一的 `log()` 函数与
  `ProgressLogger`（用整行 `print` 取代 `tqdm`，适配 GUI 子进程日志面板，
  支持 `VH_NO_PROGRESS` 环境变量节流）。
- **`scripts.api` 全面增强**：所有方法体内 *懒导入* 重型依赖，调用前做
  轻量参数校验（路径存在性、阈值范围、互斥选项），并新增
  `annotation_stats` / `annotation_label_stats` / `annotation_stats_cli`
  / `clear_annotations` / `check_annotation_type` /
  `check_image_annotation_type` / `discover_trained_models` 等 API。

### 新增功能

- **标注清除工具（`scripts/clear_annotations.py` + `scripts/core/clear_annotations.py`）**：
  按"自动 / 自动后矫正 / 手动"三种类型选择性删除 X-AnyLabeling JSON。
  必须显式指定至少一个 `--include-*` 开关，否则拒绝执行（防误删）。
  同步集成到"数据标注统计"GUI 页面。
- **图片去重 phash 后端**：除原有 ViT 深度特征外新增感知哈希 (`backend="phash"`)
  路径，仅依赖 Pillow/numpy（可选 scipy），适合大批量粗筛或纯 CPU 环境，
  通过 `--backend phash --hash-size 16` 启用。
- **视频抽帧 `seek_mode` 与 `overwrite`**：新增 `decode_all` / `seek` 两种
  跳帧策略；默认 `decode_all` 顺序解码保证精度，`seek` 直接 seek 到下一目标帧
  显著加速；增加 `--overwrite` 控制是否覆盖同名文件，否则自动追加 `_1/_2`。
- **导出 YOLO 数据集 `copy_mode`**：支持 `copy` / `link`（硬链接，失败回退复制）
  / `symlink`（符号链接），便于在不复制大数据集的前提下完成划分。
- **GUI 启动引导页（`gui/welcome_page.py`）**：启动后先展示最近工作目录列表，
  支持新增 / 移除 / 一键打开，选定后再进入主界面。最多保留 20 条历史。
- **GUI 全局上下文（`gui/context.py`）**：`AppContext` 集中保存
  `work_dir` / `python_env` 并通过 Qt 信号通知页面，取代此前"沿 parent 链
  查找"的写法。
- **GUI 子进程参数构造（`gui/_proc.py`）**：`build_script_argv()` 把 Python
  风格 kwargs 自动转成 `--kebab-case` 形式 CLI argv，统一空值/布尔/列表处理；
  `infer_script_name()` 便于日志文件命名。
- **GUI 配置持久化（`gui/settings.py`）**：封装基于 `QSettings` 的读写
  （`work_dir` / `python_env` / `recent_work_dirs` / `window_geometry`），
  避免在各页面散落键名常量。
- **关于对话框**：`gui/app.py` 内置 `AboutDialog`，展示项目号 / 版本 /
  Python & PyQt 版本 / 操作系统 / 解释器路径，并支持一键复制环境信息。

### 工程化

- **测试套件（`tests/`）**：基于 pytest，覆盖
  `annotation_type` / `common_iters` / `deduplicate_phash` /
  `export_yolo_dataset` 等核心模块。配置位于 `pyproject.toml`
  （`[tool.pytest.ini_options]`）。
- **`pyproject.toml`**：当前仅用于聚合 pytest 配置，项目本身仍以脚本方式运行。
- **`requirements-dev.txt`**：开发 / 测试依赖（pytest）。
- **`requirements-gui.txt`**：GUI 独立打包专用依赖（仅含 PyQt5），
  避免 PyInstaller 把 torch / ultralytics 等重库收集进 exe。
- **打包脚本**：
  - `build.sh`（Linux）
  - `build.bat`（Windows）
  - `build-macos.sh`（macOS，自动适配 `.app` Bundle 与目录式产物）
  - 共用 `visionHelper.spec`：`console=False`、显式
    `excludes` 掉重型依赖，发布产物形如
    `dist/visionHelper/{visionHelper(.exe), _internal/, scripts/}`，
    `scripts/` 仅原样拷贝，运行期由用户选择的解释器加载。

---

## 任务类型与标注约定

visionHelper 支持以下 4 种 YOLO 任务类型，**自动标注、数据集导出、模型训练**
3 个工具均通过 `--task` 参数（或 GUI 下拉框）切换：

| 任务         | task 值     | X-AnyLabeling JSON 形式                                                    | YOLO 模型后缀 |
| ------------ | ----------- | -------------------------------------------------------------------------- | ------------- |
| 目标检测     | `detect`    | `shapes[*].shape_type == "rectangle"`                                      | _无_          |
| 旋转框       | `obb`       | `shapes[*].shape_type == "rotation"`（4 个点）                              | `-obb`        |
| 实例分割     | `segment`   | `shapes[*].shape_type == "polygon"`                                        | `-seg`        |
| 图像分类     | `classify`  | 顶层 `flags: {类别名: bool, ...}`，其中**第一个值为 True 的 key** 即类别  | `-cls`        |

**重要限制：图像分类目前仅支持单标签**

- 一张图片只能对应一个分类，自动标注与数据集导出均按 X-AnyLabeling JSON 顶层
  `flags` 中**第一个 `True`** 的 key 处理；其他 `True` 的 key 会被忽略。
- 自动标注 classify 任务时只刷新顶层 `flags` 字段，不会改动 `shapes`；
  反之 detect/obb/segment 写入时只刷新 `shapes`、保留原有 `flags`。
- classify 数据集采用 ImageFolder 结构：`output_dir/images/{train,test}/<class_name>/<image>`；
  训练时 `data` 参数会自动指向 `images/` 根目录而非 `data.yaml`。

### 标注类型判定

`scripts/common/annotation_type.py` 根据 JSON 中是否包含 `auto_annotated_time`
字段，以及该时间戳与文件 `mtime` 的差距，判定标注属于：

- `manual`（手动标注）
- `auto`（自动标注，且未被人工二次修改）
- `auto_corrected`（自动标注后被人工修改）

容差秒数由 `--tolerance-seconds` / `tolerance_seconds` 控制（默认 2.0s），
被 **自动标注** 与 **标注清除** 等工具共用。

---

## 功能

### 核心工具（`scripts/`）

> 表格中"CLI 命令"列给出新的统一入口示例；所有参数均为显式 `--` 选项。

| 工具 | CLI 命令 | 实现位置 |
| --- | --- | --- |
| 视频抽帧 | `python scripts/vh.py images import` | `scripts/images/import_.py` |
| 图片去重 | `python scripts/vh.py images dedup` | `scripts/images/dedup.py` |
| 标注统计 | `python scripts/vh.py datasets stats` | `scripts/datasets/stats.py` |
| 自动标注 | `python scripts/vh.py datasets auto` | `scripts/datasets/auto.py` |
| 标注清除 | `python scripts/vh.py datasets clear` | `scripts/datasets/clear.py` |
| YOLO 数据集导出 | `python scripts/vh.py datasets export` | `scripts/datasets/export.py` |
| 模型训练 | `python scripts/vh.py train run` | `scripts/train/train.py` |
| 标注类型判定 | — | `scripts/common/annotation_type.py` |

各工具的能力概述：

- **视频抽帧 (`python scripts/vh.py images import`)**
  - 按指定帧间隔从视频中抽取画面并保存为图片。
  - 支持 JPEG、PNG、WebP、BMP、TIFF 等格式。
  - 支持指定起止时间、图片质量、文件名前缀、`--seek-mode {decode_all,seek}`、`--overwrite` 等参数。

- **图片去重 (`python scripts/vh.py images dedup`)**
  - 提供两种特征后端：
    - `vit`（默认，精度高）：基于 ViT / DINOv2 等深度特征，依赖 `torch` / `transformers`。
    - `phash`（轻量）：基于感知哈希，仅依赖 Pillow/numpy（可选 scipy），适合 CPU 与大规模粗筛。
  - 自动找出重复图片，支持原地删除（`--delete`）或移动到指定目录（`--move-to`）。

- **标注统计 (`python scripts/vh.py datasets stats`)**
  - 遍历目录，根据同名 X-AnyLabeling JSON 标注文件统计各类 `shape_type` 的图片数量、按标签的实例数量、
    以及 **手动 / 自动 / 自动后矫正** 三种标注类型的分布。
  - stdout 中以 `===VH_STATS_BEGIN===` / `===VH_STATS_END===` 之间的 JSON 块输出机器可读结果，
    供 GUI 通过 `AnnotationAPI.annotation_stats_cli` 直接解析。

- **自动标注 (`python scripts/vh.py datasets auto`)**
  - 使用已训练好的 YOLO 模型对图片自动生成 X-AnyLabeling JSON 标注（兼容 LabelMe 格式）。
  - 支持 4 种任务类型：`detect` / `obb` / `segment` / `classify`，可设置置信度阈值与 IoU 阈值。
  - 通过 `--include-unannotated` / `--include-auto` / `--include-auto-corrected` / `--include-manual`
    4 个开关控制处理范围（未标注 / 已自动标注 / 自动标注后矫正 / 手动标注）。
  - 自动写入 `auto_annotated_time` 字段以区分自动 / 手动标注。
  - `detect` / `obb` / `segment` 写入时仅刷新 `shapes` 字段、保留原有顶层 `flags`；
    `classify` 写入时仅刷新顶层 `flags` 字段、保留原有 `shapes`。

- **标注清除 (`python scripts/vh.py datasets clear`)**
  - 按 **自动 / 自动后矫正 / 手动** 三种类型批量清除目录顶层的 X-AnyLabeling JSON。
  - 必须显式指定 `--include-auto` / `--include-auto-corrected` / `--include-manual` 中至少一个，
    否则工具会拒绝执行（防误删）。
  - 通过 `--tolerance-seconds` 复用与自动标注一致的时间容差判定。

- **YOLO 数据集导出 (`python scripts/vh.py datasets export`)**
  - 将工作目录下已标注的图片导出为 YOLO 格式数据集，支持 `detect` / `obb` / `segment` / `classify`。
  - `detect` / `obb` / `segment`：标准 YOLO 结构（`images/{train,test}` + `labels/{train,test}`）。
  - `classify`：ImageFolder 结构（`images/{train,test}/<class>/<image>`）。
  - 仅划分训练集与测试集，不生成单独验证集；`data.yaml` 中 `val` 指向测试集以满足 Ultralytics 校验。
  - 支持 `--copy-mode {copy,link,symlink}` 控制落盘方式，便于在大数据集上避免重复拷贝。

- **模型训练 (`python scripts/vh.py train run`)**
  - 基于 Ultralytics YOLO 对导出的数据集进行训练。
  - 支持 `detect` / `obb` / `segment` / `classify`，根据任务自动为模型名追加后缀
    （`-obb` / `-seg` / `-cls`）。
  - 可指定 `epochs` / `imgsz` / `batch` / `device` / `optimizer` / `lr0` / `patience` / `workers` /
    `resume` 等参数；`classify` 任务的 `data` 自动指向数据集 `images/` 根目录而非 `data.yaml`。

### 公共基础模块

- **`scripts/api.py`**：统一对外 API。
  以类方法形式暴露各能力（`VideoAPI` / `ImageAPI` / `AnnotationAPI` / `TrainingAPI`），
  方法体内 *懒导入* 重型依赖、调用前做轻量参数校验。
- **`scripts/common/utils.py`**：内部公共工具（图像/标注扩展名常量、`is_image_file`、
  `load_annotation`、`resolve_image_path`、`discover_trained_models`、迭代器等），
  避免在各工具间重复实现。
- **`scripts/common/config.py`**：后端公共常量（版本号、任务类型、扩展名、CLI 输出协议标记、
  进度日志参数等），零副作用、零重依赖。
- **`scripts/common/logging.py`**：统一 `log()` 与 `ProgressLogger`（替代 tqdm，适配 GUI 日志面板）。
- **`scripts/cli.py`**：统一命令行路由，解析全局选项后分发到各子包入口。

### 图形界面（`gui/`）

- **主程序 (`gui/app.py`)**
  - 基于 PyQt5 构建的跨平台桌面 GUI。
  - 启动后先展示 `WelcomePage`，选定工作目录后再进入主界面。
  - 主界面顶部为"工作目录 + Python 环境"信息条，菜单栏提供：
    *视频抽帧 / 标注统计 / 模型训练 / 自动标注 / 关闭项目 / 关于*。
  - 通过子进程调用 `python scripts/vh.py <subcommand> <action> ...` 执行任务，并弹出日志窗口实时展示输出。

- **欢迎页 (`gui/welcome_page.py`)**
  - 展示最近工作目录列表（默认最多 20 条），每项可点击进入或一键移除，
    新增目录通过弹出 `QFileDialog` 选择。

- **全局上下文 (`gui/context.py`)**
  - `AppContext` 集中保存 `work_dir` / `python_env`，通过 Qt 信号通知各子页面。

- **页面基类 (`gui/base_pages.py`)**
  - `BasePage`：统一卡片式容器与 `_work_dir` / `_python_env` 访问。
  - `BaseTaskPage`：在 `BasePage` 基础上扩展表单控件、文件选择、子进程启动能力。

- **公共控件 (`gui/widgets.py`)**
  - `LabeledSpinBox` / `LabeledDoubleSpinBox` / `PrimaryButton` / `SecondaryButton` /
    `LinkButton` / `IconButton` / `MutedLabel` / `SectionTitle` / `HSeparator` 等。

- **视频抽帧 + 图片去重 (`gui/video_frame_page.py`)**
  - 一页双卡片，整合视频抽帧与图片去重两个相邻流程。

- **数据标注统计 (`gui/data_annotation_page.py`)**
  - 展示目录级标注统计：图片总数、已标注、目标检测/OBB/多边形数量，按标签实例统计，
    以及手动 / 自动 / 自动后矫正分布。
  - 集成 **标注清除** 操作面板，可选择性删除目录顶层指定类型的 JSON。

- **自动标注 (`gui/auto_annotate_page.py`)**
  - 自动扫描工作目录下 `runs/` 中已训练的模型，选择后对图片进行自动标注。
  - 任务类型可选 `detect` / `obb` / `segment` / `classify`。
  - 提供"处理范围"4 个复选框（未标注 / 自动标注 / 自动标注后矫正 / 手动标注），可任意组合，
    默认仅勾选"未标注"。

- **模型训练 (`gui/model_training_page.py`)**
  - 整合 YOLO 数据集导出与模型训练两步流程，任务类型可选
    `detect` / `obb` / `segment` / `classify`。
  - 切换任务类型时，"基模型"下拉框会自动刷新为对应任务的权重
    （例如 classify 显示 `yolov8n-cls` / `yolo11n-cls` 等），无需手工拼接后缀。

- **日志弹窗 (`gui/run_log_dialog.py`)**
  - 实时展示子进程 stdout / stderr，并自动写入日志文件，便于事后排查。

- **子进程参数构造 (`gui/_proc.py`)**
  - `build_script_argv()` 把 Python 风格 kwargs 转换为
    `python scripts/vh.py <subcommand> <action> --kebab-case` 形式的 argv，
    自动跳过 `None` / 空串、处理 `bool` / `list`。

- **配置持久化 (`gui/settings.py`)**
  - 封装 `QSettings`，提供 `load_/save_work_dir`、`load_/save_python_env`、
    `load_/save_recent_dirs`、`promote_recent_dir`、`load_/save_window_geometry` 等便捷函数。

---

## 环境

推荐使用独立 Python 环境运行 `scripts/`（深度学习相关功能依赖较重）：

```bash
python
```

GUI 自身仅依赖 PyQt5，可以使用更轻量的环境（仅安装 `requirements-gui.txt`）来构建发布版本；
运行时再让用户在顶部"Python 环境"输入框里指定已安装重型依赖的解释器即可。

---

## 安装依赖

### 默认（开发 / 完整运行）

```bash
pip install -r requirements.txt
```

`requirements.txt` 包括：

- `torch` / `torchvision`
- `transformers`
- `pillow` / `numpy` / `tqdm`
- `opencv-python`
- `PyQt5`
- `ultralytics`

### 开发 / 测试

```bash
pip install -r requirements-dev.txt
pytest               # 等价于 pytest -q（见 pyproject.toml）
```

### GUI 独立打包

```bash
python -m venv .venv-gui
source .venv-gui/bin/activate          # Windows: .venv-gui\Scripts\activate
pip install -r requirements-gui.txt
pip install pyinstaller
# Linux:
./build.sh
# Windows:
build.bat
# macOS:
./build-macos.sh
```

打包产物位于 `dist/visionHelper/`，包含 `visionHelper(.exe)`、`_internal/`
以及原样拷贝的 `scripts/` 源码目录。运行时在 GUI 顶部"Python 环境"输入框中指定
已安装重型依赖的解释器即可。

---

## 快速开始

### 视频抽帧

```bash
python scripts/vh.py images import \
    --input /path/to/video.mp4 \
    --output /path/to/output \
    --frame-step 5 \
    --ext jpg \
    --quality 95 \
    --prefix frame \
    --seek-mode decode_all
```

### 图片去重

```bash
# ViT 后端（精度高，建议有 GPU）
python scripts/vh.py images dedup \
    --input /path/to/images \
    --threshold 0.95 \
    --model google/vit-base-patch16-224 \
    --batch-size 8

# phash 后端（轻量，纯 CPU 也可跑）
python scripts/vh.py images dedup \
    --input /path/to/images \
    --backend phash \
    --hash-size 16 \
    --threshold 0.95
```

### 标注统计

```bash
python scripts/vh.py datasets stats \
    --input /path/to/annotated_images
```

### 自动标注

```bash
python scripts/vh.py datasets auto \
    --input /path/to/images \
    --model /path/to/best.pt \
    --task detect \
    --threshold 0.25 \
    --iou 0.45 \
    --include-unannotated
```

### 标注清除（谨慎！会直接删除 JSON）

```bash
# 仅清除"自动标注"产生的 JSON
python scripts/vh.py datasets clear \
    --input /path/to/images \
    --include-auto
```

### 导出 YOLO 数据集

```bash
python scripts/vh.py datasets export \
    --input /path/to/annotated_images \
    --output /path/to/.dataset \
    --task detect \
    --train-ratio 0.8 \
    --test-ratio 0.2 \
    --copy-mode copy
```

### 训练模型

```bash
python scripts/vh.py train run \
    --data /path/to/.dataset/data.yaml \
    --task detect \
    --model yolov8n \
    --epochs 100 \
    --imgsz 640 \
    --batch 16
```

### 统一 API 调用

```python
from scripts.api import VideoAPI, ImageAPI, AnnotationAPI, TrainingAPI

# 视频抽帧
VideoAPI.extract_video_frames(
    input_video="/path/to/video.mp4",
    output_dir="/path/to/output",
    frame_step=5,
)

# 图片去重（phash 后端）
result = ImageAPI.deduplicate(
    folder="/path/to/images",
    threshold=0.95,
    backend="phash",
    hash_size=16,
)

# 标注统计（进程内）
stats = AnnotationAPI.annotation_stats("/path/to/annotated_images")
labels = AnnotationAPI.annotation_label_stats("/path/to/annotated_images")

# 标注统计（子进程，GUI 推荐）
payload = AnnotationAPI.annotation_stats_cli(
    folder="/path/to/annotated_images",
    python_executable="/path/to/python",
)

# 标注清除（仅清除自动标注 JSON）
AnnotationAPI.clear_annotations(
    folder="/path/to/annotated_images",
    include_auto=True,
)

# 自动标注
AnnotationAPI.auto_annotate(
    work_dir="/path/to/images",
    model_path="/path/to/best.pt",
    task="detect",
    threshold=0.25,
    include_unannotated=True,
)

# 导出 YOLO 数据集
TrainingAPI.export_yolo_dataset(
    input_dir="/path/to/annotated_images",
    output_dir="/path/to/.dataset",
    task="detect",
    copy_mode="copy",
)

# 训练模型
TrainingAPI.train_model(
    dataset_yaml="/path/to/.dataset/data.yaml",
    task="detect",
    model="yolov8n",
    epochs=100,
)
```

### 启动图形界面

```bash
# 开发态
python -m gui.app
# 或者
python gui_main.py
```

启动后先在欢迎页选择或新增工作目录，然后顶部菜单栏可在
*视频抽帧 / 标注统计 / 模型训练 / 自动标注* 页面之间切换；
"关闭项目"可回到欢迎页重新选择工作目录。

---

## 项目结构

```text
visionHelper/
├── README.md                      # 项目说明（即本文件）
├── LICENSE
├── pyproject.toml                 # pytest 配置聚合
├── requirements.txt               # 完整运行依赖
├── requirements-gui.txt           # GUI 独立打包依赖（仅 PyQt5）
├── requirements-dev.txt           # 开发 / 测试依赖
├── gui_main.py                    # PyInstaller 入口
├── visionHelper.spec              # PyInstaller spec
├── build.sh / build.bat / build-macos.sh   # 三平台打包脚本
├── scripts/                       # 核心工具包（命令行 / API）
│   ├── __init__.py                    # 零副作用
│   ├── vh.py                          # 统一 CLI 入口
│   ├── cli.py                         # 统一命令行路由
│   ├── api.py                         # 对外统一 API 接口
│   ├── common/                        # 公共基础模块
│   │   ├── __init__.py                    # 零副作用
│   │   ├── config.py                      # 后端公共常量
│   │   ├── utils.py                       # 内部公共工具
│   │   ├── logging.py                     # 统一日志 / 进度工具
│   │   └── annotation_type.py             # 标注类型判定
│   ├── images/                        # 图片资源流程
│   │   ├── __init__.py                    # 零副作用
│   │   ├── import_.py                     # 视频抽帧
│   │   └── dedup.py                       # 图片去重
│   ├── datasets/                      # 数据集制作流程
│   │   ├── __init__.py                    # 零副作用
│   │   ├── stats.py                       # 标注统计
│   │   ├── auto.py                        # 自动标注
│   │   ├── clear.py                       # 标注清除
│   │   └── export.py                      # YOLO 数据集导出
│   ├── train/                         # 模型训练流程
│   │   ├── __init__.py                    # 零副作用
│   │   └── train.py                       # 模型训练
│   ├── predict/                       # 模型预测流程（预留）
│   │   ├── __init__.py                    # 零副作用
│   │   └── predict.py                     # 模型预测
│   └── deploy/                        # 模型部署流程（预留）
│       ├── __init__.py                    # 零副作用
│       └── deploy.py                      # 导出部署模型
├── gui/                           # 图形界面包（PyQt5）
│   ├── __init__.py
│   ├── app.py                         # PyQt5 主程序 + 关于对话框
│   ├── welcome_page.py                # 启动引导页（最近工作目录）
│   ├── context.py                     # 全局应用上下文（AppContext）
│   ├── settings.py                    # QSettings 持久化封装
│   ├── _proc.py                       # 子进程 argv 构造工具
│   ├── base_pages.py                  # 页面基类（BasePage / BaseTaskPage）
│   ├── widgets.py                     # 公共自定义控件
│   ├── theme.py                       # 全局主题（颜色、字体、QSS）
│   ├── config.py                      # GUI 全局常量与配置
│   ├── video_frame_page.py            # 视频抽帧 + 图片去重页面
│   ├── data_annotation_page.py        # 数据标注统计 + 标注清除页面
│   ├── auto_annotate_page.py          # 自动标注页面
│   ├── model_training_page.py         # 模型训练（导出 + 训练）页面
│   └── run_log_dialog.py              # 运行日志弹窗
└── tests/                         # pytest 测试套件
    ├── conftest.py
    ├── test_annotation_type.py
    ├── test_common_iters.py
    ├── test_deduplicate_phash.py
    └── test_export_yolo_dataset.py
```

---

## 编码规范

- 使用 Python 3 语法。
- 文件头使用 `#!/usr/bin/env python3` 和 `# -*- coding: utf-8 -*-`。
- 为模块和公开函数编写清晰的 docstring。
- 优先使用 `pathlib.Path` 处理文件路径。
- 关键函数包含参数校验和异常处理。
- 业务实现放在 `scripts/<subpackage>/<feature>.py` 中；`scripts/cli.py`
  仅做路由，不实现业务逻辑；新增工具时请遵循按流程分包的目录结构。
- 每个工具通过 `scripts.api` 中的 API 接口方法对外暴露；
  方法体内 *懒导入* 重型依赖，避免 `import scripts.api` 自身就触发
  `torch` / `ultralytics` 等加载。
- `scripts/` 与 `gui/` 互不依赖：
  - `scripts` 不应 `import gui`；
  - `gui` 不应 `import scripts.api` 或 `scripts.*` 中的重型模块，
    所有耗时操作都通过 `python scripts/vh.py <subcommand> <action> ...` 子进程触发。
- `scripts/__init__.py` 与各子包 `__init__.py` 必须保持零副作用，
  禁止在 import 时顺带 import 任何重型子模块。
- 模型训练依赖 `ultralytics`，使用前请确认已安装。
