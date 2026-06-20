## visionHelper

> **项目号**：`VH-2026-001`
> **版本号**：`1.0.0`
> **作者**：IceCoke

visionHelper 是一个轻量级的视觉辅助工具集，目前包含视频抽帧、图片去重、数据标注统计、自动标注、YOLO 数据集导出与模型训练等功能，并提供基于 PyQt5 的图形界面。

项目划分为两个互不依赖的子包：

- `scripts/`：纯命令行/编程接口的核心工具包，可直接通过 `python -m scripts.xxx` 调用，也可以通过 `scripts.api` 暴露的统一 API 在 Python 代码中调用。
- `gui/`：基于 PyQt5 的桌面图形界面，通过子进程调用 `scripts` 模块完成实际工作。

> **设计约束**：`scripts` 与 `gui` 之间不存在反向依赖，`scripts` 可独立使用，不引入任何 GUI 相关依赖。

## 任务类型与标注约定

visionHelper 支持以下 4 种 YOLO 任务类型，自动标注、数据集导出、模型训练 3 个工具均通过 `--task` 参数（或 GUI 下拉框）切换：

| 任务         | task 值     | LabelMe JSON 形式                                                          | YOLO 模型后缀 |
| ------------ | ----------- | -------------------------------------------------------------------------- | ------------- |
| 目标检测     | `detect`    | `shapes[*].shape_type == "rectangle"`                                      | _无_          |
| 旋转框       | `obb`       | `shapes[*].shape_type == "rotation"`（4 个点）                              | `-obb`        |
| 实例分割     | `segment`   | `shapes[*].shape_type == "polygon"`                                        | `-seg`        |
| 图像分类     | `classify`  | 顶层 `flags: {类别名: bool, ...}`，其中**第一个值为 True 的 key** 即类别  | `-cls`        |

**重要限制：图像分类目前仅支持单标签**

- 一张图片只能对应一个分类，自动标注与数据集导出均按 LabelMe JSON 顶层 `flags` 中**第一个 `True`** 的 key 处理；其他 `True` 的 key 会被忽略。
- 自动标注 classify 任务时只刷新顶层 `flags` 字段，不会改动 `shapes`；反之 detect/obb/segment 写入时只刷新 `shapes`、保留原有 `flags`。
- classify 数据集采用 ImageFolder 结构：`output_dir/images/{train,test}/<class_name>/<image>`；训练时 `data` 参数会自动指向 `images/` 根目录而非 `data.yaml`。

## 功能


### 核心工具（`scripts/`）

- **视频抽帧 (`scripts/extract_video_frames.py`)**
  - 按指定帧间隔从视频中抽取画面并保存为图片。
  - 支持 JPEG、PNG、WebP、BMP、TIFF 等格式。
  - 支持指定起止时间、图片质量、文件名前缀等参数。

- **图片去重 (`scripts/deduplicate_images.py`)**
  - 基于 ViT（Vision Transformer）特征对图片进行相似度计算。
  - 自动找出重复图片并支持删除或移动到指定目录。

- **标注统计 (`scripts/annotation_stats.py`)**
  - 遍历目录，根据同名的 LabelMe JSON 标注文件统计各类 shape_type 的图片数量、按标签的实例数量等。

- **标注类型判断 (`scripts/annotation_type.py`)**
  - 根据 JSON 中是否包含 ``auto_annotated_time`` 字段及其与文件修改时间的差距，
    判断标注为手动标注、自动标注或自动标注并手动矫正。

- **自动标注 (`scripts/auto_annotate.py`)**
  - 使用已训练好的 YOLO 模型对图片自动生成 LabelMe JSON 标注。
  - 支持 4 种任务类型：目标检测（detect）、旋转框（obb）、实例分割（segment）、图像分类（classify），可设置置信度阈值与 IoU 阈值。
  - 通过 ``include_unannotated`` / ``include_auto`` / ``include_auto_corrected`` / ``include_manual`` 4 个开关控制处理范围（未标注 / 已自动标注 / 自动标注后矫正 / 手动标注）。
  - 自动写入 ``auto_annotated_time`` 字段以区分自动/手动标注。
  - detect/obb/segment 写入时仅刷新 ``shapes`` 字段、保留原有顶层 ``flags``；classify 写入时仅刷新顶层 ``flags`` 字段、保留原有 ``shapes``。

- **YOLO 数据集导出 (`scripts/export_yolo_dataset.py`)**
  - 将工作目录下已标注的图片导出为 YOLO 格式数据集。
  - 支持 4 种任务类型：目标检测（detect）、旋转框（obb）、实例分割（segment）、图像分类（classify）。
  - detect/obb/segment 输出标准 YOLO 结构（`images/{train,test}` + `labels/{train,test}`）；classify 输出 ImageFolder 结构（`images/{train,test}/<class>/<image>`）。
  - 仅划分为训练集与测试集，不生成单独验证集；`data.yaml` 中 `val` 指向测试集以满足 Ultralytics 训练校验需求。

- **模型训练 (`scripts/train_model.py`)**
  - 基于 Ultralytics YOLO 对导出的数据集进行自动训练。
  - 支持 detect / obb / segment / classify 4 种任务，自动根据任务类型为模型名追加后缀（``-obb`` / ``-seg`` / ``-cls``）。
  - 可指定模型、epoch、imgsz、batch 等参数；classify 任务会将 ``data`` 参数指向数据集 ``images/`` 根目录而非 ``data.yaml``。


- **统一 API (`scripts/api.py`)**
  - 将各工具以 `VideoAPI`、`ImageAPI`、`AnnotationAPI`、`TrainingAPI` 类方法的形式暴露，便于上层调用与测试。

- **公共工具 (`scripts/_common.py`)**
  - 集中提供图像/标注扩展名常量、`is_image_file`、`load_annotation`、`resolve_image_path` 等内部辅助函数，避免在各工具间重复实现。

### 图形界面（`gui/`）

- **主程序 (`gui/app.py`)**
  - 基于 PyQt5 构建的跨平台桌面 GUI。
  - 通过顶部菜单栏切换首页、视频抽帧、图片去重、数据标注、自动标注、模型训练等子页面。
  - 通过子进程调用 `scripts` 模块执行任务，并弹出日志窗口实时展示输出。

- **页面基类 (`gui/base_pages.py`)**
  - `BasePage`：统一卡片式容器与 ``_work_dir`` 工作目录访问。
  - `BaseTaskPage`：在 `BasePage` 基础上扩展表单控件、文件选择、子进程启动能力。

- **公共控件 (`gui/widgets.py`)**
  - `LabeledSpinBox`、`LabeledDoubleSpinBox`：带文字标签的整型/浮点数值输入框，供多个页面复用。

- **数据标注统计 (`gui/data_annotation_page.py`)**
  - 在 GUI 中展示目录级标注统计：图片总数、已标注、目标检测/OBB/多边形数量，以及按标签的实例统计。

- **自动标注 (`gui/auto_annotate_page.py`)**
  - 自动扫描工作目录下 `runs/` 中已训练的模型，选择后对图片进行自动标注。
  - 任务类型可选 detect / obb / segment / classify。
  - 提供"处理范围"4 个复选框：**未标注**、**自动标注**、**自动标注后矫正**、**手动标注**，可任意组合，决定哪些图片会被本次自动标注覆盖（默认仅勾选"未标注"）。

- **模型训练 (`gui/model_training_page.py`)**
  - 整合 YOLO 数据集导出与模型训练两步流程，任务类型可选 detect / obb / segment / classify。
  - 切换任务类型时，"基模型"下拉框会自动刷新为对应任务的权重（例如 classify 显示 ``yolov8n-cls`` / ``yolo11n-cls`` 等），无需手工拼接后缀。



## 环境

项目使用以下 Python 环境：

```bash
/home/zh/.anaconda3/envs/vision/bin/python
```

## 安装依赖

```bash
/home/zh/.anaconda3/envs/vision/bin/pip install -r requirements.txt
```

依赖包包括：

- `torch`
- `torchvision`
- `transformers`
- `pillow`
- `numpy`
- `tqdm`
- `opencv-python`
- `PyQt5`
- `ultralytics`

## 快速开始

### 视频抽帧

```bash
/home/zh/.anaconda3/envs/vision/bin/python -m scripts.extract_video_frames \
    /path/to/video.mp4 /path/to/output \
    --frame-step 5 \
    --ext jpg \
    --quality 95 \
    --prefix frame
```

### 图片去重

```bash
/home/zh/.anaconda3/envs/vision/bin/python -m scripts.deduplicate_images \
    /path/to/images \
    --threshold 0.95 \
    --model google/vit-base-patch16-224 \
    --batch-size 8
```

### 自动标注

```bash
/home/zh/.anaconda3/envs/vision/bin/python -m scripts.auto_annotate \
    /path/to/images \
    /path/to/best.pt \
    --task detect \
    --threshold 0.25 \
    --iou 0.45
```

### 导出 YOLO 数据集

```bash
/home/zh/.anaconda3/envs/vision/bin/python -m scripts.export_yolo_dataset \
    /path/to/annotated_images \
    /path/to/.dataset \
    --task detect \
    --train-ratio 0.8 \
    --test-ratio 0.2
```

### 训练模型

```bash
/home/zh/.anaconda3/envs/vision/bin/python -m scripts.train_model \
    /path/to/.dataset/data.yaml \
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

# 图片去重
result = ImageAPI.deduplicate(
    folder="/path/to/images",
    threshold=0.95,
)

# 标注统计
stats = AnnotationAPI.annotation_stats("/path/to/annotated_images")

# 导出 YOLO 数据集
TrainingAPI.export_yolo_dataset(
    input_dir="/path/to/annotated_images",
    output_dir="/path/to/.dataset",
    task="detect",
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
/home/zh/.anaconda3/envs/vision/bin/python -m gui.app
```

启动后，通过顶部菜单栏切换首页、视频抽帧、图片去重、数据标注、自动标注与模型训练等页面。

## 项目结构

```text
visionHelper/
├── .clinerules                    # Cline 项目规则
├── README.md                      # 项目说明
├── requirements.txt               # Python 依赖
├── scripts/                       # 核心工具包（命令行 / API）
│   ├── __init__.py
│   ├── _common.py                     # 内部公共工具（图像扩展名、标注加载等）
│   ├── annotation_stats.py            # 标注统计
│   ├── annotation_type.py             # 标注类型判断
│   ├── api.py                         # 统一对外 API 接口
│   ├── auto_annotate.py               # 自动标注工具
│   ├── deduplicate_images.py          # 图片去重工具
│   ├── export_yolo_dataset.py         # YOLO 数据集导出工具
│   ├── extract_video_frames.py        # 视频抽帧工具
│   └── train_model.py                 # 模型训练工具
└── gui/                           # 图形界面包（PyQt5）
    ├── __init__.py
    ├── app.py                         # PyQt5 主程序
    ├── auto_annotate_page.py          # 自动标注页面
    ├── base_pages.py                  # 页面基类
    ├── config.py                      # GUI 全局常量与配置
    ├── data_annotation_page.py        # 数据标注统计页面
    ├── image_deduplicate_page.py      # 图片去重页面
    ├── model_training_page.py         # 模型训练页面
    ├── run_log_dialog.py              # 运行日志弹窗
    ├── video_frame_page.py            # 视频抽帧页面
    └── widgets.py                     # 公共自定义控件
```

## 编码规范

- 使用 Python 3 语法。
- 文件头使用 `#!/usr/bin/env python3` 和 `# -*- coding: utf-8 -*-`。
- 为模块和公开函数编写清晰的 docstring。
- 优先使用 `pathlib.Path` 处理文件路径。
- 关键函数包含参数校验和异常处理。
- 每个工具通过 `scripts.api` 中的 API 接口方法对外暴露。
- `scripts/` 与 `gui/` 互不依赖：`scripts` 不应该 `import gui`，反之亦然。
- 模型训练依赖 `ultralytics`，使用前请确认已安装。
