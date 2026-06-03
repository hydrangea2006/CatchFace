import numpy as np
from oefilter import oefilter 
import time
class land_mark_transformer:
    FACE_POINTS = {
    "eye": {
        # 索引顺序：[外眼角, 内眼角, 上眼睑外/中/内, 下眼睑外/中/内]
        "LEFT":  [130, 133, 160, 159, 158, 144, 145, 153],
        "RIGHT": [263, 362, 387, 386, 385, 373, 374, 380],
    },
    "brow": {
        # 眉毛边缘的 8 个水平与垂直控制点
        "LEFT":  [35, 244, 63, 105, 66, 229, 230, 231],
        "RIGHT": [265, 464, 293, 334, 296, 449, 450, 451],
    },
    "pupil": {
        # 468/473 为左右虹膜几何中心（瞳孔），其余为边界点。必须在 Options 中开启 refine_face_landmarks
        "LEFT":  [468, 469, 470, 471, 472],
        "RIGHT": [473, 474, 475, 476, 477],
    },
        
    "head_rigid": {
        "LEFT_FOREHEAD":  21,   # 左额头上方骨骼点
        "RIGHT_FOREHEAD": 251,  # 右额头上方骨骼点
        "RIGHT_JAW":      397,  # 右下颌骨边缘点
        "LEFT_JAW":       172,  # 左下颌骨边缘点
    },
    "mouth": {
        "EYE_INNER_L": 133,  # 左眼内眼角（作为嘴部距离解算的归一化绝对基准物）
        "EYE_INNER_R": 362,  # 右眼内眼角
        "EYE_OUTER_L": 130,  # 左眼外眼角
        "EYE_OUTER_R": 263,  # 右眼外眼角
        
        "UPPER_LIP":   13,   # 上唇内侧正中心点
        "LOWER_LIP":   14,   # 下唇内侧正中心点
        "CORNER_L":    61,   # 左嘴角肌肉锚点
        "CORNER_R":    291,  # 右嘴角肌肉锚点
    }
    }
    def __init__(self, mincutoff=1.2, beta=0.005, dcutoff=1.0):
        # 1. 提取所有要用到的 34 个不重复索引
        all_indices = set()
        for side in ["LEFT", "RIGHT"]:
            all_indices.update(self.FACE_POINTS["eye"][side])
            all_indices.update(self.FACE_POINTS["brow"][side])
            all_indices.update(self.FACE_POINTS["pupil"][side])
        all_indices.update(self.FACE_POINTS["head_rigid"].values())
        all_indices.update(self.FACE_POINTS["mouth"].values())
        
        self.needed_indices = sorted(list(all_indices))
        
        #  一个点只需要一个 oefilter 实例
        self.filters = {}
        for idx in self.needed_indices:
            self.filters[idx] = oefilter(mincutoff=mincutoff, beta=beta, dcutoff=dcutoff)
            
        # 记录上一帧的时间戳，用来给你的滤波器计算 te (dt)
        self.last_time = None

    def get_smooth_mesh(self, raw_face_landmarks):
        # 1. 计算时间差 te (单位：秒)
        current_time = time.time()
        if self.last_time is None:
            # 第一帧时，给一个默认的、极小的帧间隔时间（比如 30 帧的 0.033 秒）
            te = 0.033
        else:
            te = current_time - self.last_time
        
        # 防止极端情况下 te 为 0 导致源码中 (x - self.x) / te 报除以零错误
        if te <= 0:
            te = 0.001
            
        self.last_time = current_time # 刷新时间，留给下一帧用

        # 2. 用 Numpy 切出 34 个点的原始数据
        all_pts = np.array([[lm.x, lm.y, lm.z] for lm in raw_face_landmarks])
        raw_34_points = all_pts[self.needed_indices]
        
        smooth_mesh_mock = {}
        
        # 用源码里的 __call__ 进行引用
        for i, idx in enumerate(self.needed_indices):
            raw_xyz = raw_34_points[i] # 拿到当前点的原始 [x, y, z]
            
            # 直接把实例当函数调：对象名(x, y, z, te)
            # 它会直接吐出洗干净的三个轴坐标
            smooth_x, smooth_y, smooth_z = self.filters[idx](raw_xyz[0], raw_xyz[1], raw_xyz[2], te)
            
            # 装进大字典
            smooth_mesh_mock[idx] = [smooth_x, smooth_y, smooth_z]
            
        return smooth_mesh_mock
    def transformor(self, smooth_mesh_mock):
        eyeInnerCornerL = smooth_mesh_mock[133] #左眼内眼角
        eyeInnerCornerR = smooth_mesh_mock[362] #右眼内眼角
        eyeOuterCornerL = smooth_mesh_mock[130]  #左眼外眼角
        eyeOuterCornerR = smooth_mesh_mock[263]  #右眼外眼角
        #计算双眼内眼角距离和外眼角总宽度(这两个的数值在人脸靠近/远离时会成比例缩放，是优秀的参照物)

        eyeInnerDistance = np.linalg.norm(eyeInnerCornerL - eyeInnerCornerR) #计算两点之间的欧式距离
        eyeOuterDistance = np.linalg.norm(eyeOuterCornerL - eyeInnerCornerR)