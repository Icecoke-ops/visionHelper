#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# visionHelper GUI 打包脚本
#
# 用法：
#   ./build.sh                       # 使用默认 PyInstaller 环境（系统 PATH 中的 pyinstaller）
#   PYINSTALLER=/path/to/pyinstaller ./build.sh
#
# 产物位于 ./dist/visionHelper/，已自动把仓库根的 scripts/ 拷贝进去，
# 形成 "exe + scripts/源码" 的最终发布目录布局：
#
#   dist/visionHelper/
#   ├── visionHelper            ← GUI 可执行文件
#   ├── _internal/              ← PyInstaller 运行时
#   └── scripts/                ← 原样拷贝的脚本源码（运行期由用户的 Python 解释器加载）
#
# 设计上保持 scripts/ 不被打进 exe；用户在 GUI 顶部"Python 环境"选项里
# 选好已安装 torch / ultralytics 等依赖的解释器即可。
# ----------------------------------------------------------------------------

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYINSTALLER="${PYINSTALLER:-pyinstaller}"
SPEC_FILE="visionHelper.spec"
DIST_DIR="$ROOT_DIR/dist/visionHelper"

if ! command -v "$PYINSTALLER" >/dev/null 2>&1; then
    echo "[ERROR] 未找到可执行文件 '$PYINSTALLER'。请先安装 PyInstaller："
    echo "        pip install pyinstaller"
    echo "        或通过环境变量指定，如：PYINSTALLER=/path/to/pyinstaller ./build.sh"
    exit 1
fi

echo "[1/3] 清理旧构建产物 ..."
rm -rf "$ROOT_DIR/build" "$ROOT_DIR/dist"

echo "[2/3] 运行 PyInstaller ($SPEC_FILE) ..."
"$PYINSTALLER" --noconfirm --clean "$SPEC_FILE"

if [[ ! -d "$DIST_DIR" ]]; then
    echo "[ERROR] 预期的输出目录不存在：$DIST_DIR"
    exit 1
fi

echo "[3/3] 拷贝 scripts/ 到发布目录 ..."
# 移除可能由 PyInstaller 误带入的 scripts 残留，然后用源码目录原样覆盖
rm -rf "$DIST_DIR/scripts"
cp -r "$ROOT_DIR/scripts" "$DIST_DIR/scripts"

# 顺手清理 __pycache__，让发布物干净
find "$DIST_DIR/scripts" -type d -name "__pycache__" -prune -exec rm -rf {} +

cat <<EOF

✅ 构建完成
   产物目录：$DIST_DIR
   入口文件：$DIST_DIR/visionHelper

发布说明：
  • scripts/ 已与可执行文件同级放置，运行期通过 PYTHONPATH 注入即可定位。
  • 启动 GUI 后，请在顶部"Python 环境"选择一个安装好 torch / ultralytics
    / transformers / opencv 等依赖的 Python 解释器（绝对路径）。
  • 若双击运行报错或无窗口，可在终端直接执行可执行文件查看 stderr：
        $DIST_DIR/visionHelper

EOF
