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
            running_mode = vision.RunningMode.LIVE_STREAM,  
            result_callback = self.result_callback,
            num_faces = 1
        )

    def start_stream(self):
        self.face_landmarker = vision.FaceLandmarker.create_from_options(self.options)
        self.cap = cv2.VideoCapture(0)
        
        while self.cap.isOpened():
            success, image = self.cap.read()
            if not success:
                print("unable to get image from camera")
                continue
                
            rbg_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rbg_img)
            
            timestamp = int(time.time() * 1000)
            self.face_landmarker.detect_async(mp_image, timestamp)
            
            if self.done_frame is not None:
                cv2.imshow("test", self.done_frame)
            else:
                cv2.imshow("test", image)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        self.cap.release()
        cv2.destroyAllWindows()

    def result_callback(self, result, output_image, s_timestamp):
        rbg_frame = output_image.numpy_view()
        annotated_frame = rbg_frame.copy()
        
        h, w, _ = annotated_frame.shape
        
        
        if result.face_landmarks:
            for face_landmarks in result.face_landmarks:
                for landmark in face_landmarks:
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    

                    cv2.circle(annotated_frame, (cx, cy), 1, (0, 255, 200), -1)
                
        self.done_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

if __name__ == "__main__":
    catcher = FaceCatcher()
    catcher.start_stream()