@echo off
chcp 65001 >nul
echo ========================================
echo  ResearchPaperHub v4.0 - 打包工具
echo ========================================
echo.

REM 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到 Python，请先安装 Python 3.10+
    pause
    exit /b 1
)

REM 安装 PyInstaller
echo [1/3] 安装 PyInstaller...
pip install pyinstaller -i https://mirrors.aliyun.com/pypi/simple/ -q

REM 打包
echo [2/3] 正在打包（可能需要几分钟）...
pyinstaller --onefile --name ResearchPaperHub ^
    --add-data "app.py;." ^
    --add-data "engine.py;." ^
    --add-data "data;data" ^
    --add-data "requirements.txt;." ^
    --hidden-import streamlit ^
    --hidden-import sqlalchemy ^
    --hidden-import httpx ^
    --hidden-import fitz ^
    --hidden-import numpy ^
    --hidden-import pyvis ^
    --hidden-import networkx ^
    --hidden-import plotly ^
    --hidden-import sklearn ^
    --hidden-import dotenv ^
    --collect-all streamlit ^
    --collect-all sqlalchemy ^
    run.py

if errorlevel 1 (
    echo 打包失败！
    pause
    exit /b 1
)

REM 完成
echo [3/3] 打包完成！
echo.
echo 可执行文件位置: dist\ResearchPaperHub.exe
echo.
echo 使用方法:
echo   1. 将 dist\ResearchPaperHub.exe 复制到目标电脑
echo   2. 双击运行
echo   3. 首次运行会自动安装依赖（需要网络）
echo   4. 浏览器会自动打开 http://localhost:8501
echo.
pause
