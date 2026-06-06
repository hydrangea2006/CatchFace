#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
一键启动所有组件
按顺序启动：服务器 → 算法端 → 前端
"""

import subprocess
import sys
import os
import time
import threading
import webbrowser

# ========== 配置区域 ==========
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 各组件路径
SERVER_PATH = os.path.join(PROJECT_ROOT, "Flask_server", "app.py")
ALGORITHM_PATH = os.path.join(PROJECT_ROOT, "visual_Algorithm", "visual.py")
FRONTEND_PATH = os.path.join(PROJECT_ROOT, "frontend")

# 是否自动打开浏览器（前端）
AUTO_OPEN_BROWSER = True

# 启动顺序和延迟（秒）
DELAY_BEFORE_ALGORITHM = 3   # 服务器启动后等待3秒再启动算法端
DELAY_BEFORE_FRONTEND = 5    # 算法端启动后等待5秒再启动前端

# ========== 颜色输出 ==========
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_info(msg):
    print(f"{Colors.CYAN}[INFO]{Colors.ENDC} {msg}")

def print_success(msg):
    print(f"{Colors.GREEN}[SUCCESS]{Colors.ENDC} {msg}")

def print_error(msg):
    print(f"{Colors.RED}[ERROR]{Colors.ENDC} {msg}")

def print_warning(msg):
    print(f"{Colors.YELLOW}[WARNING]{Colors.ENDC} {msg}")

def print_step(step, msg):
    print(f"\n{Colors.BOLD}{Colors.BLUE}━━━ 步骤 {step} ━━━{Colors.ENDC}")
    print(f"{Colors.BLUE}{msg}{Colors.ENDC}\n")


# ========== 检查依赖 ==========
def check_dependencies():
    """检查必要的依赖是否已安装"""
    print_step("0", "检查依赖")
    
    # 检查 Python 包
    try:
        import flask
        import flask_socketio
        import socketio
        print_success("✓ Flask-SocketIO 已安装")
    except ImportError as e:
        print_error(f"缺少依赖: {e}")
        print_info("请运行: pip install flask flask-socketio python-socketio eventlet")
        return False
    
    # 检查前端依赖（可选）
    if os.path.exists(os.path.join(FRONTEND_PATH, "package.json")):
        node_modules = os.path.join(FRONTEND_PATH, "node_modules")
        if not os.path.exists(node_modules):
            print_warning("前端依赖未安装，请在前端目录运行: npm install")
            print_info("前端将不会自动启动")
            return True  # 不强制要求前端依赖
        else:
            print_success("✓ 前端依赖已安装")
    
    return True


# ========== 启动服务器 ==========
def start_server():
    """启动 Flask-SocketIO 服务器"""
    print_step("1", "启动后端服务器")
    
    print_info(f"服务器地址: {SERVER_URL}")
    
    # 使用 subprocess 启动服务器
    try:
        # Windows 使用 python，Linux/Mac 使用 python3
        python_cmd = sys.executable
        process = subprocess.Popen(
            [python_cmd, SERVER_PATH],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        print_success("服务器进程已启动")
        return process
        
    except Exception as e:
        print_error(f"启动服务器失败: {e}")
        return None


# ========== 启动算法端 ==========
def start_algorithm():
    """启动算法端（面部捕捉）"""
    print_step("2", "启动算法端（面部捕捉）")
    
    if not os.path.exists(ALGORITHM_PATH):
        print_error(f"找不到算法端文件: {ALGORITHM_PATH}")
        print_warning("请确认 visual_Algorithm/visual.py 存在")
        return None
    
    print_info(f"算法端路径: {ALGORITHM_PATH}")
    
    try:
        python_cmd = sys.executable
        process = subprocess.Popen(
            [python_cmd, ALGORITHM_PATH],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        print_success("算法端进程已启动")
        return process
        
    except Exception as e:
        print_error(f"启动算法端失败: {e}")
        return None


# ========== 启动前端 ==========
def start_frontend():
    """启动 React 前端"""
    print_step("3", "启动前端")
    
    if not os.path.exists(FRONTEND_PATH):
        print_error(f"找不到前端路径: {FRONTEND_PATH}")
        return None
    
    # 检查是否已安装依赖
    node_modules = os.path.join(FRONTEND_PATH, "node_modules")
    if not os.path.exists(node_modules):
        print_error("前端依赖未安装！")
        print_info("请先运行: cd frontend && npm install")
        return None
    
    print_info(f"前端路径: {FRONTEND_PATH}")
    
    # 打开浏览器
    if AUTO_OPEN_BROWSER:
        webbrowser.open("http://localhost:3000")
        print_info("已打开浏览器: http://localhost:3000")
    
    try:
        # 使用 npm start 启动
        process = subprocess.Popen(
            ["npm", "start"],
            cwd=FRONTEND_PATH,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            shell=True  # Windows 需要
        )
        
        print_success("前端进程已启动")
        return process
        
    except Exception as e:
        print_error(f"启动前端失败: {e}")
        return None


# ========== 监控进程 ==========
def monitor_processes(processes, names):
    """监控所有进程，如果有进程退出，打印提示"""
    while True:
        for proc, name in zip(processes, names):
            if proc and proc.poll() is not None:
                print_warning(f"{name} 已退出 (退出码: {proc.returncode})")
                return False
        time.sleep(1)
    return True


# ========== 打印服务器日志 ==========
def print_server_logs(process):
    """实时打印服务器日志"""
    if not process:
        return
    
    for line in process.stdout:
        if "连接" in line or "断开" in line:
            print(f"{Colors.GREEN}[服务器]{Colors.ENDC} {line.strip()}")
        elif "错误" in line or "ERROR" in line:
            print(f"{Colors.RED}[服务器]{Colors.ENDC} {line.strip()}")
        elif "警告" in line or "WARNING" in line:
            print(f"{Colors.YELLOW}[服务器]{Colors.ENDC} {line.strip()}")
        else:
            # 可选：不打印普通日志
            pass


# ========== 主函数 ==========
def main():
    """主函数：一键启动所有组件"""
    print(f"\n{Colors.BOLD}{Colors.HEADER}")
    print("=" * 60)
    print("         面部捕捉系统 - 一键启动")
    print("=" * 60)
    print(f"{Colors.ENDC}")
    
    # 检查依赖
    if not check_dependencies():
        print_error("依赖检查失败，请安装后重试")
        return
    
    print_info("开始启动各组件...\n")
    
    # 1. 启动服务器
    server_process = start_server()
    if not server_process:
        print_error("服务器启动失败，终止启动流程")
        return
    
    print_info(f"等待 {DELAY_BEFORE_ALGORITHM} 秒让服务器完全启动...")
    time.sleep(DELAY_BEFORE_ALGORITHM)
    
    # 2. 启动算法端
    algorithm_process = start_algorithm()
    if not algorithm_process:
        print_warning("算法端启动失败，继续启动前端...")
    
    print_info(f"等待 {DELAY_BEFORE_FRONTEND} 秒...")
    time.sleep(DELAY_BEFORE_FRONTEND)
    
    # 3. 启动前端
    frontend_process = start_frontend()
    if not frontend_process:
        print_warning("前端启动失败")
    
    # 打印启动完成信息
    print("\n" + "=" * 60)
    print_success("所有组件已启动！")
    print("=" * 60)
    print_info("服务器地址: http://localhost:5000")
    print_info("前端地址:   http://localhost:3000")
    print_info("健康检查:   http://localhost:5000/health")
    print("\n按 Ctrl+C 停止所有进程...\n")
    
    # 启动日志监控线程
    log_thread = threading.Thread(target=print_server_logs, args=(server_process,), daemon=True)
    log_thread.start()
    
    # 等待用户中断
    try:
        processes = [server_process, algorithm_process, frontend_process]
        names = ["服务器", "算法端", "前端"]
        while monitor_processes(processes, names):
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print_warning("正在停止所有进程...")
        
        # 终止所有子进程
        for proc, name in zip(processes, names):
            if proc:
                proc.terminate()
                print_info(f"已停止 {name}")
        
        print_success("所有进程已停止")
        print("=" * 60)


if __name__ == "__main__":
    main()
