#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# visionHelper GUI 打包脚本 (macOS)
#
# 与 build.sh 的差异
# ------------------
#
# 1. macOS 上 PyInstaller 可能产出两种形态：
#       a. dist/visionHelper/                 ← 普通目录式产物（与 Linux 一致）
#       b. dist/visionHelper.app/             ← .app Bundle（windowed/onefile=False
#          时 PyInstaller 默认会同时生成 .app）
#    本脚本会自动探测 .app 是否存在，存在则把 scripts/ 拷贝到
#    ``visionHelper.app/Contents/MacOS/scripts/``，以匹配 ``gui/config.py``
#    中 ``Path(sys.executable).parent`` 的运行期定位逻辑。
#    同时若目录式产物存在，也会同步补上 scripts/，方便在终端直接调试。
#
# 2. 使用 BSD find 兼容的清理写法（macOS 自带 BSD find，不是 GNU find）。
#
# 3. 提示文字针对 macOS 习惯（``open <app>`` / 终端直接运行可执行文件）。
#
# 用法：
#   ./build-macos.sh
#   PYINSTALLER=/path/to/pyinstaller ./build-macos.sh
#
# 设计上保持 scripts/ 不被打进 exe；用户在 GUI 顶部"Python 环境"选项里
# 选好已安装 torch / ultralytics 等依赖的解释器即可。
# ----------------------------------------------------------------------------

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYINSTALLER="${PYINSTALLER:-pyinstaller}"
SPEC_FILE="visionHelper.spec"
DIST_ROOT="$ROOT_DIR/dist"
DIST_DIR="$DIST_ROOT/visionHelper"
DIST_APP="$DIST_ROOT/visionHelper.app"

if ! command -v "$PYINSTALLER" >/dev/null 2>&1; then
    echo "[ERROR] 未找到可执行文件 '$PYINSTALLER'。请先安装 PyInstaller："
    echo "        pip install pyinstaller"
    echo "        或通过环境变量指定，如：PYINSTALLER=/path/to/pyinstaller ./build-macos.sh"
    exit 1
fi

echo "[1/3] 清理旧构建产物 ..."
rm -rf "$ROOT_DIR/build" "$ROOT_DIR/dist"

echo "[2/3] 运行 PyInstaller ($SPEC_FILE) ..."
"$PYINSTALLER" --noconfirm --clean "$SPEC_FILE"

# 至少要存在一种产物
if [[ ! -d "$DIST_DIR" && ! -d "$DIST_APP" ]]; then
    echo "[ERROR] 既未找到目录式产物 $DIST_DIR，也未找到 .app 产物 $DIST_APP"
    exit 1
fi

# ----- 工具函数：把 scripts/ 同步到指定目录，并清理 __pycache__ -----
sync_scripts() {
    local target_parent="$1"
    if [[ -z "$target_parent" || ! -d "$target_parent" ]]; then
        return 0
    fi
    echo "       → 同步 scripts/ 到 $target_parent"
    rm -rf "$target_parent/scripts"
    cp -R "$ROOT_DIR/scripts" "$target_parent/scripts"
    # BSD find 兼容写法：不依赖 -prune -exec ... +
    find "$target_parent/scripts" -type d -name "__pycache__" -exec rm -rf {} +
}

echo "[3/3] 拷贝 scripts/ 到发布产物 ..."

# (a) 目录式产物（若存在）
if [[ -d "$DIST_DIR" ]]; then
    sync_scripts "$DIST_DIR"
fi

# (b) .app Bundle（若存在）
#     gui/config.py::app_root() 在 frozen 态返回 Path(sys.executable).parent，
#     即 visionHelper.app/Contents/MacOS/，因此 scripts/ 必须放在该目录下。
APP_MACOS_DIR=""
if [[ -d "$DIST_APP" ]]; then
    APP_MACOS_DIR="$DIST_APP/Contents/MacOS"
    if [[ ! -d "$APP_MACOS_DIR" ]]; then
        echo "[ERROR] .app 结构异常：缺少 $APP_MACOS_DIR"
        exit 1
    fi
    sync_scripts "$APP_MACOS_DIR"
fi

cat <<EOF

✅ 构建完成
EOF

if [[ -d "$DIST_DIR" ]]; then
    cat <<EOF
   目录式产物：$DIST_DIR
   入口文件：  $DIST_DIR/visionHelper
EOF
fi

if [[ -d "$DIST_APP" ]]; then
    cat <<EOF
   .app Bundle：$DIST_APP
   双击启动或：  open "$DIST_APP"
   终端排错：    "$DIST_APP/Contents/MacOS/visionHelper"
EOF
fi

cat <<EOF

发布说明：
  • scripts/ 已与可执行文件同级放置（.app 内部位于 Contents/MacOS/scripts/），
    运行期通过 PYTHONPATH 注入即可定位。
  • 启动 GUI 后，请在顶部"Python 环境"选择一个安装好 torch / ultralytics
    / transformers / opencv 等依赖的 Python 解释器（绝对路径）。
  • 若双击 .app 启动后无窗口或闪退，请在终端直接运行可执行文件查看 stderr。
  • 首次运行 .app 若被 Gatekeeper 拦截，可执行：
        xattr -dr com.apple.quarantine "$DIST_APP"

EOF
