import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
from ldtransformer import land_mark_transformer
class FaceCatcher:
    def __init__(self):
        # 1. 配置 MediaPipe 的人工智能模型参数
        base_options = python.BaseOptions(model_asset_path="face_landmarker_v2_with_blendshapes.task")
        self.options = vision.FaceLandmarkerOptions(
            base_options = base_options,
            output_face_blendshapes = False,
            output_facial_transformation_matrixes = False,
            refine_face_landmarks = True,
            running_mode = vision.RunningMode.LIVE_STREAM,  # 开启异步直播流模式
            result_callback = self.result_callback,         # 指定计算完后的“交货”回调函数
            num_faces = 1
        )
        
        # 2. 实例化一个一欧元滤波器，用来专门给鼻尖（4号点）去噪防抖
        
        

    def start_stream(self):
        # 创建面部关键点检测器
        self.face_landmarker = vision.FaceLandmarker.create_from_options(self.options)
        # 打开电脑的默认摄像头（0代表第一个摄像头）
        self.cap = cv2.VideoCapture(0)
        
    
        try:
            while self.cap.isOpened():
                success, image = self.cap.read()
                if not success:
                    print("无法从摄像头获取画面...")
                    continue
                    
                # MediaPipe 需要 RGB 格式，而 OpenCV 默认读出的是 BGR 格式，所以需要转换
                rbg_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # 将转换后的图像包装成 MediaPipe 专属的 Image 对象
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rbg_img)
                
                # 获取当前精确的毫秒级时间戳（MediaPipe 异步流要求必须提供递增的时间戳）
                timestamp = int(time.time() * 1000)
                
                # 【核心动作】把图像异步扔给 AI 模型去检测，AI 算完后会自动调用 result_callback
                self.face_landmarker.detect_async(mp_image, timestamp)
                
                # 稍微让 CPU 歇 10 毫秒，防止 while 死循环把电脑处理器跑满
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\n正在接收退出指令...")
        finally:
            # 无论如何，最后都要释放摄像头并关闭 AI 实例，否则后台线程会卡死
            self.cap.release()
            self.face_landmarker.close()
            print("程序已安全关闭。")

    # 核心数据处理工厂（回调函数）：只要 AI 识别出一帧结果，就会自动进到这里
    def result_callback(self, result, output_image, s_timestamp):
        # 安全检查：如果画面里没有人脸，result 里面没东西，就直接返回，防止程序崩溃
        if not result or not result.face_landmarks:
            return
            
        # 1. 拿到检测到的第一张脸的 478 个原始关键点
        raw_face_points = result.face_landmarks[0]
        
if __name__ == "__main__":
    catcher = FaceCatcher()
    catcher.start_stream()