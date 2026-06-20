@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
REM ============================================================================
REM visionHelper GUI 打包脚本 (Windows)
REM
REM 用法：
REM   build.bat                        使用默认 PyInstaller (系统 PATH 中的 pyinstaller)
REM   set PYINSTALLER=C:\path\to\pyinstaller.exe ^&^& build.bat
REM
REM 产物位于 .\dist\visionHelper\，已自动把仓库根的 scripts\ 拷贝进去，
REM 形成 "exe + scripts\源码" 的最终发布目录布局：
REM
REM   dist\visionHelper\
REM   ├── visionHelper.exe         ← GUI 可执行文件
REM   ├── _internal\               ← PyInstaller 运行时
REM   └── scripts\                 ← 原样拷贝的脚本源码（运行期由用户的 Python 解释器加载）
REM
REM 设计上保持 scripts\ 不被打进 exe；用户在 GUI 顶部"Python 环境"选项里
REM 选好已安装 torch / ultralytics 等依赖的解释器即可。
REM ============================================================================

REM 切换到脚本所在目录
set "ROOT_DIR=%~dp0"
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"
pushd "%ROOT_DIR%" >nul

REM 允许通过环境变量 PYINSTALLER 覆盖默认命令
if "%PYINSTALLER%"=="" set "PYINSTALLER=pyinstaller"
set "SPEC_FILE=visionHelper.spec"
set "DIST_DIR=%ROOT_DIR%\dist\visionHelper"

REM 校验 PyInstaller 是否可用
where "%PYINSTALLER%" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] 未找到可执行文件 "%PYINSTALLER%"。请先安装 PyInstaller：
    echo         pip install pyinstaller
    echo         或通过环境变量指定，如：set PYINSTALLER=C:\path\to\pyinstaller.exe
    popd >nul
    exit /b 1
)

echo [1/3] 清理旧构建产物 ...
if exist "%ROOT_DIR%\build" rmdir /s /q "%ROOT_DIR%\build"
if exist "%ROOT_DIR%\dist"  rmdir /s /q "%ROOT_DIR%\dist"

echo [2/3] 运行 PyInstaller (%SPEC_FILE%) ...
"%PYINSTALLER%" --noconfirm --clean "%SPEC_FILE%"
if errorlevel 1 (
    echo [ERROR] PyInstaller 构建失败。
    popd >nul
    exit /b 1
)

if not exist "%DIST_DIR%" (
    echo [ERROR] 预期的输出目录不存在：%DIST_DIR%
    popd >nul
    exit /b 1
)

echo [3/3] 拷贝 scripts\ 到发布目录 ...
REM 移除可能由 PyInstaller 误带入的 scripts 残留，然后用源码目录原样覆盖
if exist "%DIST_DIR%\scripts" rmdir /s /q "%DIST_DIR%\scripts"
xcopy "%ROOT_DIR%\scripts" "%DIST_DIR%\scripts\" /E /I /Q /Y >nul
if errorlevel 1 (
    echo [ERROR] 拷贝 scripts\ 失败。
    popd >nul
    exit /b 1
)

REM 顺手清理 __pycache__，让发布物干净
for /d /r "%DIST_DIR%\scripts" %%D in (__pycache__) do (
    if exist "%%D" rmdir /s /q "%%D"
)

echo.
echo ✅ 构建完成
echo    产物目录：%DIST_DIR%
echo    入口文件：%DIST_DIR%\visionHelper.exe
echo.
echo 发布说明：
echo   • scripts\ 已与可执行文件同级放置，运行期通过 PYTHONPATH 注入即可定位。
echo   • 启动 GUI 后，请在顶部"Python 环境"选择一个安装好 torch / ultralytics
echo     / transformers / opencv 等依赖的 Python 解释器（绝对路径）。
echo   • 若双击运行报错或无窗口，可在终端直接执行可执行文件查看 stderr：
echo         "%DIST_DIR%\visionHelper.exe"
echo.

popd >nul
endlocal
exit /b 0
