import numpy as np
from oefilter import oefilter 
import time
class land_mark_transformer:
#   # L/R 结尾代表左右。
    # INNER/OUTER 代表内外侧。
    # UPPER/LOWER 代表上下。
    # MID 代表正中心。
    FACE_POINTS = {
        "eye": {
            # 眼部点位
            "OUTER_L":       130,  # 左眼外眼角
            "INNER_L":       133,  # 左眼内眼角
            "UPPER_OUTER_L": 160,  # 左上眼睑外侧
            "UPPER_MID_L":   159,  # 左上眼睑中心
            "UPPER_INNER_L": 158,  # 左上眼睑内侧
            "LOWER_OUTER_L": 144,  # 左下眼睑外侧
            "LOWER_MID_L":   145,  # 左下眼睑中心
            "LOWER_INNER_L": 153,  # 左下眼睑内侧

            "OUTER_R":       263,  # 右眼外眼角
            "INNER_R":       362,  # 右眼内眼角
            "UPPER_OUTER_R": 387,  # 右上眼睑外侧
            "UPPER_MID_R":   386,  # 右上眼睑中心
            "UPPER_INNER_R": 385,  # 右上眼睑内侧
            "LOWER_OUTER_R": 373,  # 右下眼睑外侧
            "LOWER_MID_R":   374,  # 右下眼睑中心
            "LOWER_INNER_R": 380,  # 右下眼睑内侧
        },
        
        "brow": {
        #原有的眉毛点位
            "OUTER_L":       35,   # 左眉最外侧端点
            "UPPER_OUTER_L": 244,  # 左眉上缘外侧
            "UPPER_MID_L":   63,   # 左眉上缘中心
            "UPPER_INNER_L": 105,  # 左眉上缘内侧
            "LOWER_INNER_L": 66,   # 左眉下缘内侧
            "LOWER_MID_L":   229,  # 左眉下缘中心
            "LOWER_OUTER_L": 230,  # 左眉下缘外侧
            "INNER_L":       231,  # 左眉最内侧端点（眉头）

            "OUTER_R":       265,  # 右眉最外侧端点
            "UPPER_OUTER_R": 464,  # 右眉上缘外侧
            "UPPER_MID_R":   293,  # 右眉上缘中心
            "UPPER_INNER_R": 334,  # 右眉上缘内侧
            "LOWER_INNER_R": 296,  # 右眉下缘内侧
            "LOWER_MID_R":   449,  # 右眉下缘中心
            "LOWER_OUTER_R": 450,  # 右眉下缘外侧
            "INNER_R":       451,  # 右眉最内侧端点（眉头）

            # 眉心核心控制点
            "GLABELLA":      9,    # 印堂点（双眉正中心鼻梁上方，算 browInnerUp 的绝对基准）
            "INNER_BASE_L":  107,  # 左眉头下压锚点（算眉毛下压/愤怒）
            "INNER_BASE_R":  336,  # 右眉头下压锚点
        },
        
        "pupil": {
            # 瞳孔点位
            "CENTER_L":      468,  # 左虹膜几何中心（瞳孔）
            "BORDER_1_L":    469,  # 左虹膜边缘点 1
            "BORDER_2_L":    470,  # 左虹膜边缘点 2
            "BORDER_3_L":    471,  # 左虹膜边缘点 3
            "BORDER_4_L":    472,  # 左虹膜边缘点 4

            "CENTER_R":      473,  # 右虹膜几何中心（瞳孔）
            "BORDER_1_R":    474,  # 右虹膜边缘点 1
            "BORDER_2_R":    475,  # 右虹膜边缘点 2
            "BORDER_3_R":    476,  # 右虹膜边缘点 3
            "BORDER_4_R":    477,  # 右虹膜边缘点 4
        },
        
        "head_rigid": {
            # --- 你原有的头部刚性点位 ---
            "FOREHEAD_L":    21,   # 左额头上方骨骼点
            "FOREHEAD_R":    251,  # 右额头上方骨骼点
            "JAW_R":         397,  # 右下颌骨边缘点
            "JAW_L":         172,  # 左下颌骨边缘点
        },
        
        "mouth": {
            # --- 你原有的嘴部基准与核心点位 ---
            "REF_EYE_INNER_L": 133,  # 嘴部计算复用的左内眼角参照物
            "REF_EYE_INNER_R": 362,  # 嘴部计算复用的右内眼角参照物
            "REF_EYE_OUTER_L": 130,  # 左外眼角参照物
            "REF_EYE_OUTER_R": 263,  # 右外眼角参照物
            
            "LIP_INNER_UPPER": 13,   # 上唇内侧正中心点（算张嘴 jawOpen）
            "LIP_INNER_LOWER": 14,   # 下唇内侧正中心点
            "CORNER_L":        61,   # 左嘴角锚点
            "CORNER_R":        291,  # 右嘴角锚点
            
            # 嘴唇外侧、嘴角四周边缘点
            "LIP_OUTER_UPPER": 0,    # 上唇外侧最顶端中心（算抿嘴 roll/提唇 upperUp）
            "LIP_OUTER_LOWER": 17,   # 下唇外侧最底端中心（算包唇 roll/下翻 lowerDown）
            "LIP_UPPER_SIDE_L": 37,  # 左上唇外侧拉高点（算单边龇牙/蔑视）
            "LIP_UPPER_SIDE_R": 267, # 右上唇外侧拉高点
            "LIP_LOWER_SIDE_L": 84,  # 左下唇外侧拉低点（算单边拉低露出下牙）
            "LIP_LOWER_SIDE_R": 314, # 右下唇外侧拉低点
            "CORNER_PURS_L":   206,  # 左嘴角外侧肌肉锚点（算抿嘴 press/酒窝 dimple）
            "CORNER_PURS_R":   426,  # 右嘴角外侧肌肉锚点
            "LIP_UNDER_SHRUG": 18,   # 下唇下方凹陷处（算耸下巴 shrugLower）
        },

        "jaw": {
            # 下巴骨骼错位点
            "TIP":            152,  # 下巴最底部的刚性骨骼尖端（算下巴左右歪斜 jawLeft/Right 和前伸 jawForward）
        },

        "cheek": {
        # 脸颊与眼底肌肉点
            "PUFF_L":         50,   # 左脸颊肉感最高点（算鼓腮帮子 cheekPuff）
            "PUFF_R":         280,  # 右脸颊肉感最高点
            "SQUINT_L":       205,  # 左眼眶正下方脸颊点（算大笑肉上挤 cheekSquint）
            "SQUINT_R":       425,  # 右眼眶正下方脸颊点
        },

        "nose": {
            # 鼻部控制点
            "BRIDGE":         6,    # 鼻梁正中心刚性死点（算皱鼻子 base）
            "TIP":            4,    # 鼻尖（作为脸颊点算空间鼓气距离的中心锚点）
            "SNEER_W_L":      129,  # 左鼻翼拉高点（算皱鼻子 noseSneer）
            "SNEER_W_R":      358,  # 右鼻翼拉高点
        }
    }
    def __init__(self, mincutoff=1.2, beta=0.005, dcutoff=1.0):
        # 1. 提取所有要用到的不重复索引
        all_indices = set()
        all_indices.update(self.FACE_POINTS["eye"].values())
        all_indices.update(self.FACE_POINTS["brow"].values())
        all_indices.update(self.FACE_POINTS["pupil"].values())
        all_indices.update(self.FACE_POINTS["head_rigid"].values())
        all_indices.update(self.FACE_POINTS["mouth"].values())
        all_indices.update(self.FACE_POINTS["jaw"].values())
        all_indices.update(self.FACE_POINTS["cheek"].values())
        all_indices.update(self.FACE_POINTS["nose"].values())

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

        # 2. 用 Numpy 切出所有点位的原始数据
        all_pts = np.array([[lm.x, lm.y, lm.z] for lm in raw_face_landmarks])
        raw_points = all_pts[self.needed_indices]
        
        smooth_mesh_mock = {}
        
        # 用源码里的 __call__ 进行引用
        for i, idx in enumerate(self.needed_indices):
            raw_xyz = raw_points[i] # 拿到当前点的原始 [x, y, z]
            
            # 直接把实例当函数调：对象名(x, y, z, te)
            # 它会直接吐出洗干净的三个轴坐标
            smooth_x, smooth_y, smooth_z = self.filters[idx](raw_xyz[0], raw_xyz[1], raw_xyz[2], te)
            
            # 装进大字典
            smooth_mesh_mock[idx] = [smooth_x, smooth_y, smooth_z]
            
        return smooth_mesh_mock
