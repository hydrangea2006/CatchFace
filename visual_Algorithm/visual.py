import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

class FaceCatcher:
    def __init__(self):
        self.done_frame = None
        base_options = python.BaseOptions(model_asset_path="face_landmarker_v2_with_blendshapes.task")
        self.options = vision.FaceLandmarkerOptions(
            base_options = base_options,
            output_face_blendshapes = True,
            output_facial_transformation_matrixes = True,
            running_mode = vision.RunningMode.VIDEO,  
            num_faces = 2
        )

    def start_stream(self):
        video_path = '除雪机.mp4'
        self.face_landmarker = vision.FaceLandmarker.create_from_options(self.options)
        self.cap = cv2.VideoCapture(video_path)
        
        while self.cap.isOpened():
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            wait_time = int(1000 / fps) if fps > 0 else 25
            success, image = self.cap.read()
            if not success:
                print("unable to get image from camera")
                continue
                
            rbg_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rbg_img)
            
            timestamp = int(time.time() * 1000)
            result = self.face_landmarker.detect_for_video(mp_image, timestamp)            
            annotated_frame = image.copy() # OpenCV 直接用原图（BGR）画图
            h, w, _ = annotated_frame.shape            
            if result.face_landmarks:
                for face_landmarks in result.face_landmarks:
                    for landmark in face_landmarks:
                        cx, cy = int(landmark.x * w), int(landmark.y * h)
                        # OpenCV 默认是 BGR，(0, 255, 200) 在这里是浅绿色/青色
                        cv2.circle(annotated_frame, (cx, cy), 1, (0, 255, 200), -1)
            
            cv2.imshow("test", annotated_frame)
                
            if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                break
                
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    catcher = FaceCatcher()
    catcher.start_stream()