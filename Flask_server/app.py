from flask import Flask, request
from flask_socketio import SocketIO, emit
import logging
import time
from datetime import datetime


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = '123456'

# 启用 CORS 允许局域网访问
socketio = SocketIO(app, 
                   cors_allowed_origins="*",
                   logger=True, 
                   engineio_logger=True,
                   async_mode='gevent')


# ========== Socket 事件处理 ==========

@socketio.on('connect')
def handle_connect():
    """客户端连接时的处理"""
    client_id = request.sid
    logger.info(f'[连接] 客户端 {client_id} 已连接')
    emit('connection_response', {
        'data': 'Connected to server', 
        'status': 'ok',
        'message': '面部捕捉服务器已连接'
    })


@socketio.on('disconnect')
def handle_disconnect():
    """客户端断开连接时的处理"""
    logger.info(f'[断开] 客户端 {request.sid} 已断开')


@socketio.on('blendshapes')
def handle_blendshapes(data):
    """
    接收算法端传来的 MediaPipe blendshape 数据
    广播给所有前端
    
    """
    try:
        # 获取原始 blendshape 列表
        blendshapes_dict = data.get('blendshapes', {})

        head_data = data.get('head', {})
        
        
        # 获取时间戳
        timestamp = data.get('timestamp', int(time.time() * 1000))
        
        # 准备广播的数据
        broadcast_data = {
            'blendshapes': blendshapes_dict,   # 直接转发字典
            'timestamp': timestamp,
            'head': head_data,                  # 可选，转发头部数据
            'source': request.sid
        }
        
        # 广播给所有连接的客户端（包括发送者）
        emit('face_data', broadcast_data, broadcast=True)
        
        # 可选：打印日志（每10帧打印一次避免刷屏，这里简单打印）
        # logger.info(f'[广播] 表情数据已广播 - 帧时间戳: {timestamp}')
        
    except Exception as e:
        logger.error(f'处理数据出错: {e}')


@socketio.on('ping')
def handle_ping():
    """心跳检测 - 客户端发送ping，服务器回复pong"""
    emit('pong', {
        'timestamp': datetime.now().timestamp()
    })



# ========== HTTP 路由 ==========

@app.route('/')
def index():
    """首页 - 服务器状态页面"""
    return {
        'status': 'running',
        'service': 'Face Tracking Server',
        'message': 'Socket.IO 服务运行中',
        'websocket_endpoint': '/socket.io',
        'port': 5000
    }


@app.route('/health')
def health_check():
    """健康检查接口 - 用于监控服务器是否正常运行"""
    return {
        'status': 'ok',
        'timestamp': datetime.now().isoformat()
    }


# @app.route('/clients')
# def get_clients():
#     """获取当前连接的客户端数量（简单实现）"""
#     # 注意：这个方法需要更复杂的实现才能获取准确数量
#     # 这里返回一个简单的信息
#     return {
#         'message': '当前连接数需要通过Socket.IO的manager获取',
#         'note': '可以使用 socketio.server.manager.rooms 查看'
#     }


# ========== 启动服务器 ==========
if __name__ == '__main__':
    HOST = '0.0.0.0'      # 允许局域网访问
    PORT = 5000           # 固定端口
    
    print("\n" + "="*60)
    print("面部捕捉服务器启动")
    print("="*60)
    print(f"本机访问: http://localhost:{PORT}")
    print(f"局域网访问: http://<你的IP地址>:{PORT}")
    print(f"WebSocket端点: ws://<你的IP>:{PORT}/socket.io")
    print("="*60)
    print("\n等待客户端连接...\n")
    
    socketio.run(app, host=HOST, port=PORT, debug=True)