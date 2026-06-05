import numpy as np

def remap(val, imin, imax, omin, omax):
    if imax == imin: return omin
    if imin < imax:
        clamped = max(imin, min(imax, val))
    else:
        clamped = max(imax, min(imin, val))
    return omin + (clamped - imin) * (omax - omin) / (imax - imin)

def solve_eyes(lm, eye_dist_base, eye_outer_base, head_y):
    eye_bs = {}
    
    # 这里放 Kalidokit 的 3组点对平均逻辑 (get_eye_lid_ratio)
    # 计算出：eyeBlinkLeft, eyeBlinkRight, eyeWideLeft, eyeWideRight, eyeSquintLeft...
    # 以及解算瞳孔：eyeLookInLeft, eyeLookOutLeft, eyeLookUpLeft...
    
    # 示例一行：
    # eye_bs["eyeBlinkLeft"] = 1.0 - left_open
    return eye_bs


def solve_brows(lm, eye_dist_base):
    brow_bs = {}
    
    # 这里放眉毛抬起、下压、挑眉的几何解算
    # 计算出：browInnerUp, browDownLeft, browDownRight, browOuterUpLeft, browOuterUpRight
    
    return brow_bs


def solve_nose_and_cheeks(lm, eye_dist_base):
    nc_bs = {}
    
    # 计算出：noseSneerLeft, noseSneerRight, cheekPuff, cheekSquintLeft, cheekSquintRight
    
    return nc_bs

def solve_mouth_and_jaw(lm, eye_dist_base, eye_outer_base):
    mouth_bs = {}
    
    # 基础张开
    ratio = np.linalg.norm(np.array(lm[13]) - np.array(lm[14])) / eye_dist_base
    mouth_bs["jawOpen"] = remap(ratio, 0.15, 0.85, 0.0, 1.0)
    
    # 防穿模闭嘴
    lip_dist = np.linalg.norm(np.array(lm[13]) - np.array(lm[14])) / eye_dist_base
    mouth_bs["mouthClose"] = remap(lip_dist, 0.2, 0.05, 0.0, 1.0) if mouth_bs["jawOpen"] > 0.1 else 0.0
    
    # 下巴歪斜与前伸
    jaw_x_offset = (lm[152][0] - lm[6][0]) / eye_dist_base
    mouth_bs["jawLeft"] = remap(jaw_x_offset, 0.0, 0.12, 0.0, 1.0)
    mouth_bs["jawRight"] = remap(jaw_x_offset, -0.12, 0.0, 1.0, 0.0)
    
    
    return mouth_bs

def solve_head_rotation(lm):
   
    # 1:1 提取原版 4 个外围边界骨骼点 (3D 向量)
    p1 = np.array(lm[21])   # 左额头上方骨骼点（左上角）
    p2 = np.array(lm[251])  # 右额头上方骨骼点（右上角）
    p3 = np.array(lm[397])  # 右下颌骨边缘点（右下角）
    p4 = np.array(lm[172])  # 左下颌骨边缘点（左下角）
    
    # 线性插值求下颌左右两点的几何中点（下巴正中心）对应 p3mid = p3.lerp(p4, 0.5)
    p3mid = (p3 + p4) * 0.5
    
    # 此时 plane = [p1, p2, p3mid] 构成了 3D 空间刚体三角形
    
    # 线性代数核心：复刻 Vector.rollPitchYaw 的内部法向量逆推机制
    # 利用三角形的三个顶点建立两个共面向量
    v1 = p2 - p1      # 从左额指向右额的水平向量
    v2 = p3mid - p1   # 从左额指向下巴中心的纵向向量
    
    # 向量外积/叉乘 (Cross Product) 算出绝对垂直于人脸平面的【法向量】
    # 归一化确保它是单位向量
    normal = np.cross(v1, v2)
    normal /= (np.linalg.norm(normal) + 1e-6)
    
    # 根据法向量在标准 3D 坐标轴上的投影，逆推出旋转比例（-1 到 1 之间）
    # 对应原版 JS 内部的未乘以 PI 的 rotate 原始值
    rotate_y = np.arctan2(normal[0], normal[2]) / np.pi  # Yaw
    rotate_x = np.arctan2(-normal[1], normal[2]) / np.pi # Pitch
    
    # 重新修正一个用于算 Roll 的水平基准线倾斜度
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-6)
    rotate_z = np.arctan2(v1_norm[1], v1_norm[0]) / np.pi # Roll
    
    # 1:1 坐标系镜像反转
    # 原版源码：rotate.x *= -1; rotate.z *= -1;
    rotate_x *= -1
    rotate_z *= -1
    
    # 换算成弧度值 (Radians)与角度制 (Degrees)
    rad_x = rotate_x * np.pi
    rad_y = rotate_y * np.pi
    rad_z = rotate_z * np.pi
    
    deg_x = rotate_x * 180.0
    deg_y = rotate_y * 180.0
    deg_z = rotate_z * 180.0
    
    #计算双额线段的中点（额头中心）
    mid_point = (p1 + p2) * 0.5
    
    #计算面部近似的 3D 物理尺寸
    width = np.linalg.norm(p2 - p1)         
    height = np.linalg.norm(p3mid - mid_point) 
    
    # 头部的 3D 空间绝对位置中心点
    position = (mid_point + p3mid) * 0.5
    
    return {
        "x": deg_x, "y": deg_y, "z": deg_z,       
        "rad_x": rad_x, "rad_y": rad_y, "rad_z": rad_z, 
        "width": width,
        "height": height,
        "position": position
    }

def calc_all_arkit_coefficients(smooth_mesh_mock):
    lm = smooth_mesh_mock
    
    # 边界安全检查：如果 MediaPipe 弄丢了人脸，吐出全0，直接返回空的或全0默认值
    if len(lm) == 0 or np.all(np.array(lm[6]) == 0):
        # 返回 52 个全 0 的标准 ARKit 字典保底，防止下游渲染崩掉
        return {k: 0.0 for k in ["jawOpen", "eyeBlinkLeft", "mouthSmileLeft"]} # 示例
        
    computed_bs = {}
    
    # 动捕必须先知道转头角度，因为后面眼睛防抽搐稳定器需要用到 head_y
    head_pose = solve_head_rotation(lm)
    head_y = head_pose["rad_y"]
    
    # 3. 计算全局刚性归一化基准（消减离镜头远近的误差）
    eyeInnerDistance = np.linalg.norm(np.array(lm[133]) - np.array(lm[362]))
    if eyeInnerDistance == 0: eyeInnerDistance = 0.001 # 再次保底
    
    eyeOuterDistance = np.linalg.norm(np.array(lm[130]) - np.array(lm[263]))
    if eyeOuterDistance == 0: eyeOuterDistance = 0.001

    computed_bs.update(solve_eyes(lm, eyeInnerDistance, eyeOuterDistance, head_y))
    computed_bs.update(solve_brows(lm, eyeInnerDistance))
    computed_bs.update(solve_nose_and_cheeks(lm, eyeInnerDistance))
    computed_bs.update(solve_mouth_and_jaw(lm, eyeInnerDistance, eyeOuterDistance))

    # 5. 统一进行最终的边界限幅安全检查 (0.0 到 1.0 之间)
    for k, v in computed_bs.items():
        computed_bs[k] = max(0.0, min(1.0, v))

    return computed_bs