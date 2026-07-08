import socketio
import time

class FaceNetSender:
    def __init__(self, server_ip="127.0.0.1", port=5000, limit_fps=60):
        self.sio = socketio.Client(
            reconnection=True,
            reconnection_attempts=0,      # 0 = 无限重试
            reconnection_delay=1,
            reconnection_delay_max=5,
        )
        self.url = f"http://{server_ip}:{port}"
        self.is_connect = False
        self.min_interval = 1.0 / limit_fps
        self.last_send_ts = 0

        @self.sio.event
        def connect():
            self.is_connect = True
            print(f"[网络发送]成功连接面部服务:{self.url}")

        @self.sio.event
        def connection_service(data):
            print("[服务端回执]", data["message"])

        @self.sio.event
        def disconnect():
            self.is_connect = False
            print("[网络发送]断开服务连接")

    def connect(self):
        try:
            self.sio.connect(self.url, transports=["websocket"])
            return True
        except Exception as e:
            print("[连接失败]", e, "- 将自动重试")
            return False

    def send_face_pack(self, arkit_dict):
        """直接传入arkit_response字典一键发送"""
        if not self.is_connect:
            return
        now = time.time()
        # 帧率限流
        if now - self.last_send_ts < self.min_interval:
            return
        self.last_send_ts = now

        send_data = {
            "blendshapes": arkit_dict["blendshapes"],
            "head": arkit_dict["head"],
            "timestamp": arkit_dict["timestamp"]
        }
        self.sio.emit("blendshapes", send_data)

    def release(self):
        if self.is_connect:
            self.sio.disconnect()