"""
ResearchPaperHub v4.0 - Windows 桌面应用入口
双击运行即可启动，自动打开浏览器
"""
import os
import sys
import subprocess
import webbrowser
import time
import signal
import threading

# 获取应用目录（打包后为 exe 所在目录，开发时为脚本所在目录）
if getattr(sys, 'frozen', False):
    APP_DIR = os.path.dirname(sys.executable)
else:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))

os.chdir(APP_DIR)

# 依赖检查
REQUIRED_PACKAGES = [
    'streamlit', 'sqlalchemy', 'httpx', 'PyMuPDF', 'numpy',
    'python-dotenv', 'pyvis', 'networkx', 'plotly', 'scikit-learn'
]

def check_dependencies():
    """检查并安装缺失的依赖"""
    missing = []
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg.replace('-', '_').replace('PyMuPDF', 'fitz'))
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"正在安装缺失依赖: {', '.join(missing)}")
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '--quiet',
            '-i', 'https://mirrors.aliyun.com/pypi/simple/',
            *missing
        ], check=True)
        print("依赖安装完成")

def start_streamlit():
    """启动 Streamlit 服务器"""
    app_py = os.path.join(APP_DIR, 'app.py')
    if not os.path.exists(app_py):
        print(f"错误: 找不到 {app_py}")
        input("按回车键退出...")
        sys.exit(1)

    # 启动 Streamlit
    cmd = [
        sys.executable, '-m', 'streamlit', 'run', app_py,
        '--server.port=8501',
        '--server.address=localhost',
        '--server.headless=true',
        '--browser.gatherUsageStats=false'
    ]

    print("=" * 50)
    print("  ResearchPaperHub v4.0")
    print("  智能学术研究助手")
    print("=" * 50)
    print()
    print("正在启动服务...")
    print("启动后将自动打开浏览器")
    print("关闭此窗口将停止服务")
    print()

    # 启动 Streamlit 进程
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=APP_DIR
    )

    # 等待服务启动
    time.sleep(3)

    # 打开浏览器
    url = "http://localhost:8501"
    print(f"正在打开浏览器: {url}")
    webbrowser.open(url)

    print()
    print("服务已启动！")
    print("按 Ctrl+C 或关闭此窗口停止服务")
    print()

    # 等待进程结束
    try:
        process.wait()
    except KeyboardInterrupt:
        print("\n正在停止服务...")
        process.terminate()
        process.wait(timeout=5)
        print("服务已停止")

def main():
    """主函数"""
    print("检查依赖...")
    check_dependencies()
    start_streamlit()

if __name__ == '__main__':
    main()
