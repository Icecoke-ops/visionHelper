"""
visionHelper 对外统一 API 接口。

以类方法的形式暴露项目内的核心能力，便于上层（GUI、脚本、测试）统一调用。

设计要点
========

* **零副作用导入**：所有重型依赖（torch、ultralytics、opencv、transformers 等）
  都通过方法体内的 *懒导入* 在真正调用时才被加载，因此
  ``import scripts.api`` 自身是无副作用的，即使部分依赖缺失，
  也不会影响未使用功能的可用性。
* **轻量参数校验**：每个 API 方法在调用核心实现前会做一次便宜的边界检查
  （路径存在性、阈值范围、互斥选项等），尽早抛出异常。
* **签名稳定**：方法签名与默认值保持向后兼容，调用方升级版本无需修改代码。

用法示例
========

::

    from scripts.api import VideoAPI, ImageAPI

    VideoAPI.extract_video_frames(
        input_video="/path/to/video.mp4",
        output_dir="/path/to/output",
        frame_step=5,
    )

    result = ImageAPI.deduplicate(
        folder="/path/to/images",
        threshold=0.95,
        delete=False,
    )
"""

from scripts.api.video import VideoAPI
from scripts.api.annotation import AnnotationAPI
from scripts.api.training import TrainingAPI
from scripts.api.predict import PredictAPI
from scripts.api.image import ImageAPI

__all__ = [
    "VideoAPI",
    "AnnotationAPI",
    "TrainingAPI",
    "PredictAPI",
    "ImageAPI",
]
