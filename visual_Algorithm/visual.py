import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
class FaceCatcher:
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path="face_landmarker_v2_with_blendshapes.task")
        self.options = vision.FaceLandmarkerOptions(
            base_options = base_options,
            output_face_blendshapes = True,
        )