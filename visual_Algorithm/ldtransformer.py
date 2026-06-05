import numpy as np
from oefilter import oefilter 
import time
import arkit_solver
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
        # 直接用全部478个点，省得各种KeyError
        self.needed_indices = list(range(478))
        
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

    def get_arkit_blendshapes_response(self, raw_face_landmarks):

        # 1. 滤波清洗
        smooth_mesh = self.get_smooth_mesh(raw_face_landmarks)
        
        if len(smooth_mesh) == 0:
            return {
                "timestamp": int(time.time() * 1000),
                "head": {"rotation": {"x": 0.0, "y": 0.0, "z": 0.0}, "position": [0.0, 0.0, 0.0]},
                "blendshapes": {}
            }
            
        #直接调用 arkit_solver 原本就有的头部姿态函数
        # 它返回的字典包含: "rad_x", "rad_y", "rad_z", "position" (np.array)
        head_pose = arkit_solver.solve_head_rotation(smooth_mesh)
        
        # 3. 直接调用 arkit_solver 原本的核心解算函数，拿到基础系数字典
        raw_bs = arkit_solver.calc_all_arkit_coefficients(smooth_mesh)

        # 4. 提取绝对物理中心位置 (p1, p2, p3, p4 质心)，转换为 list
        # 这样 arkit_solver 内部甚至连 solve_head_rotation 都不用改动
        p1 = np.array(smooth_mesh[21])   
        p2 = np.array(smooth_mesh[251])  
        p3 = np.array(smooth_mesh[397])  
        p4 = np.array(smooth_mesh[172])  
        head_position = ((p1 + p2 + p3 + p4) * 0.25).tolist()

        # 特殊处理：解算器原本的嘴部拉伸叫 "mouthPucker" 或 "mouthFunnel"，
        # 模板中需要 "mouthStretch"，我们在这里取解算器里算好的数值做映射
        # 也可以直接用 mouthPucker 的趋势来当做 Stretch 使用（或者给默认值）
        mouth_stretch_val = raw_bs.get("mouthPucker", 0.0) 

        #完美拼装前端需要的 52 通道大字典
        arkit_blendshapes = {
            "eyeBlinkLeft":     float(raw_bs.get("eyeBlinkLeft", 0.0)),
            "eyeBlinkRight":    float(raw_bs.get("eyeBlinkRight", 0.0)),
            "eyeLookDownLeft":  0.0,     "eyeLookDownRight": 0.0,
            "eyeLookInLeft":    float(raw_bs.get("eyeLookInLeft", 0.0)),
            "eyeLookInRight":   float(raw_bs.get("eyeLookInRight", 0.0)),
            "eyeLookOutLeft":   float(raw_bs.get("eyeLookOutLeft", 0.0)),
            "eyeLookOutRight":  float(raw_bs.get("eyeLookOutRight", 0.0)),
            "eyeLookUpLeft":    float(raw_bs.get("eyeLookUpLeft", 0.0)),
            "eyeLookUpRight":   float(raw_bs.get("eyeLookUpRight", 0.0)),
            "eyeSquintLeft":    float(raw_bs.get("eyeSquintLeft", 0.0)),
            "eyeSquintRight":   float(raw_bs.get("eyeSquintRight", 0.0)),
            "eyeWideLeft":      float(raw_bs.get("eyeWideLeft", 0.0)),
            "eyeWideRight":     float(raw_bs.get("eyeWideRight", 0.0)),
            "browDownLeft":     float(raw_bs.get("browDownLeft", 0.0)),
            "browDownRight":    float(raw_bs.get("browDownRight", 0.0)),
            "browInnerUp":      float(raw_bs.get("browInnerUp", 0.0)),
            "browOuterUpLeft":  float(raw_bs.get("browOuterUpLeft", 0.0)),
            "browOuterUpRight": float(raw_bs.get("browOuterUpRight", 0.0)),
            "jawOpen":          float(raw_bs.get("jawOpen", 0.0)),
            "jawForward":       0.0,
            "jawLeft":          float(raw_bs.get("jawLeft", 0.0)),
            "jawRight":         float(raw_bs.get("jawRight", 0.0)),
            "mouthClose":       float(raw_bs.get("mouthClose", 0.0)),
            "mouthFunnel":      float(raw_bs.get("mouthFunnel", 0.0)),
            "mouthPucker":      float(raw_bs.get("mouthPucker", 0.0)),
            "mouthLeft":        0.0,        "mouthRight":       0.0,
            "mouthSmileLeft":   float(raw_bs.get("mouthSmileLeft", 0.0)),
            "mouthSmileRight":  float(raw_bs.get("mouthSmileRight", 0.0)),
            "mouthFrownLeft":   float(raw_bs.get("mouthFrownLeft", 0.0)),
            "mouthFrownRight":  float(raw_bs.get("mouthFrownRight", 0.0)),
            "mouthStretchLeft": float(mouth_stretch_val),
            "mouthStretchRight":float(mouth_stretch_val),
            "mouthDimpleLeft":  0.0,        "mouthDimpleRight": 0.0,
            "mouthPressLeft":   0.0,        "mouthPressRight":  0.0,
            "mouthRollLower":   0.0,        "mouthRollUpper":   0.0,
            "mouthShrugLower":  0.0,        "mouthShrugUpper":  0.0,
            "mouthUpperUpLeft": 0.0,        "mouthUpperUpRight":0.0,
            "mouthLowerDownLeft":0.0,       "mouthLowerDownRight":0.0,
            "cheekPuff":        0.0,
            "cheekSquintLeft":  float(raw_bs.get("cheekSquintLeft", 0.0)),
            "cheekSquintRight": float(raw_bs.get("cheekSquintRight", 0.0)),
            "noseSneerLeft":    float(raw_bs.get("noseSneerLeft", 0.0)),
            "noseSneerRight":   float(raw_bs.get("noseSneerRight", 0.0)),
            "tongueOut":        0.0
        }

        # 完美交差返回
        return {
            "timestamp": int(time.time() * 1000),
            "head": {
                "rotation": {
                    "x": float(head_pose["rad_x"]), 
                    "y": float(head_pose["rad_y"]), 
                    "z": float(head_pose["rad_z"])
                },
                "position": head_position
            },
            "blendshapes": arkit_blendshapes
        }