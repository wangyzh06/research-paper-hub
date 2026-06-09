@echo off
chcp 65001 >nul
echo ========================================
echo  ResearchPaperHub v4.0 - 安装向导
echo ========================================
echo.

REM 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python！
    echo.
    echo 请先安装 Python 3.10 或更高版本：
    echo   https://www.python.org/downloads/
    echo.
    echo 安装时请勾选 "Add Python to PATH"
    echo.
    pause
    exit /b 1
)

echo [1/4] 检查 Python 版本...
python --version
echo.

REM 创建虚拟环境
echo [2/4] 创建虚拟环境...
if not exist "venv" (
    python -m venv venv
    echo 虚拟环境创建完成
) else (
    echo 虚拟环境已存在
)
echo.

REM 激活虚拟环境并安装依赖
echo [3/4] 安装依赖（首次安装可能需要几分钟）...
call venv\Scripts\activate.bat
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ -q
echo 依赖安装完成
echo.

REM 创建启动脚本
echo [4/4] 创建启动脚本...
(
echo @echo off
echo chcp 65001 ^>nul
echo echo 正在启动 ResearchPaperHub...
echo call venv\Scripts\activate.bat
echo streamlit run app.py --server.port 8501 --server.headless true
echo pause
) > start_app.bat

echo.
echo ========================================
echo  安装完成！
echo ========================================
echo.
echo 使用方法：
echo   1. 双击 start_app.bat 启动应用
echo   2. 浏览器会自动打开 http://localhost:8501
echo   3. 关闭命令窗口停止服务
echo.
echo 首次启动需要加载 AI 模型，请耐心等待
echo.
pause
