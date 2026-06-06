import numpy as np

def remap(val, imin, imax, omin, omax):
    if imax == imin: return omin
    if imin < imax:
        clamped = max(imin, min(imax, val))
    else:
        clamped = max(imax, min(imin, val))
    return omin + (clamped - imin) * (omax - omin) / (imax - imin)
# 阻止了因为转头而导致的形状扭曲，就是乘以一个逆旋转矩阵
def transform_to_local_space(lm, head_pose):
    origin = head_pose["position"]
    R = head_pose["R"]          # 3×3 旋转矩阵
    
    local_lm = {}
    for idx, pt in lm.items():
        # 平移后乘以旋转矩阵的转置（= 逆矩阵，因为 R 是正交矩阵）
        local_lm[idx] = R.T @ (np.array(pt) - origin)
        
    return local_lm
# 解算出眼部的Arkit系数
def solve_eyes(lm, eye_dist_base, eye_outer_base):

    eye_bs = {}
    # 提取两眼的边界死点 (用于瞳孔追踪基准)
    p_eye_l_inner = np.array(lm[133]) # 左内眼角
    p_eye_l_outer = np.array(lm[130]) # 左外眼角
    # 右眼
    p_eye_r_inner = np.array(lm[362]) # 右内眼角
    p_eye_r_outer = np.array(lm[263]) # 右外眼角

    # 眼睑张开度解算 (垂直 3 组点对距离平均法)
    # 左眼上下眼睑 3 组对称点
    l_top_points = [np.array(lm[159]), np.array(lm[158]), np.array(lm[160])]
    l_bot_points = [np.array(lm[145]), np.array(lm[153]), np.array(lm[144])]
    # 右眼上下眼睑 3 组对称点
    r_top_points = [np.array(lm[386]), np.array(lm[385]), np.array(lm[387])]
    r_bot_points = [np.array(lm[374]), np.array(lm[373]), np.array(lm[380])]

    # 计算 3 组垂直距离的平均值
    l_vert_dist = sum(np.linalg.norm(t - b) for t, b in zip(l_top_points, l_bot_points)) / 3.0
    r_vert_dist = sum(np.linalg.norm(t - b) for t, b in zip(r_top_points, r_bot_points)) / 3.0

    # 归一化：除以外眼角总跨度，消除离镜头远近的干扰
    l_open_ratio = l_vert_dist / (eye_outer_base + 1e-6)
    r_open_ratio = r_vert_dist / (eye_outer_base + 1e-6)


    # 映射到标准 ARKit Blendshapes 权重
    # 正常睁眼 ratio 在 0.18 ~ 0.28 左右，低于 0.13 判定为闭眼，高于 0.29 判定为瞪眼
    eye_bs["eyeBlinkLeft"]  = remap(l_open_ratio, 0.24, 0.12, 0.0, 1.0)
    eye_bs["eyeBlinkRight"] = remap(r_open_ratio, 0.24, 0.12, 0.0, 1.0)
    
    eye_bs["eyeWideLeft"]   = remap(l_open_ratio, 0.26, 0.35, 0.0, 1.0)
    eye_bs["eyeWideRight"]  = remap(r_open_ratio, 0.26, 0.35, 0.0, 1.0)
    
    # Squint (眯眼) 逻辑：当微眯但没完全闭上时触发
    eye_bs["eyeSquintLeft"]  = remap(l_open_ratio, 0.20, 0.14, 0.0, 1.0) if eye_bs["eyeBlinkLeft"] < 0.7 else 0.0
    eye_bs["eyeSquintRight"] = remap(r_open_ratio, 0.20, 0.14, 0.0, 1.0) if eye_bs["eyeBlinkRight"] < 0.7 else 0.0

    # 瞳孔精细解算 (Pupil Tracking)
    # 提取高精度瞳孔中心 3D 点
    p_pupil_l = np.array(lm[468]) # 左瞳孔中心
    p_pupil_r = np.array(lm[473]) # 右瞳孔中心

    # 计算瞳孔中心到内眼角的距离，占“内眼角到外眼角总跨度”的百分比
    l_eye_width = np.linalg.norm(p_eye_l_outer - p_eye_l_inner) + 1e-6
    r_eye_width = np.linalg.norm(p_eye_r_outer - p_eye_r_inner) + 1e-6

    # 左眼水平轴
    l_eye_axis = p_eye_l_outer - p_eye_l_inner
    l_eye_axis /= np.linalg.norm(l_eye_axis) + 1e-6

    # 右眼水平轴
    r_eye_axis = p_eye_r_outer - p_eye_r_inner
    r_eye_axis /= np.linalg.norm(r_eye_axis) + 1e-6

    # 瞳孔相对于内眼角
    l_pupil_vec = p_pupil_l - p_eye_l_inner
    r_pupil_vec = p_pupil_r - p_eye_r_inner

    # 投影到眼轴
    l_pupil_x_ratio = (
        np.dot(l_pupil_vec, l_eye_axis)
        / l_eye_width
    )

    r_pupil_x_ratio = (
        np.dot(r_pupil_vec, r_eye_axis)
        / r_eye_width
    )

    # 判定看左看右：正常直视时 ratio 约为 0.5
    # 左眼（LookIn 是向右看鼻梁，LookOut 是向左看外侧）
    eye_bs["eyeLookInLeft"]   = remap(l_pupil_x_ratio, 0.52, 0.35, 0.0, 1.0) # 偏向内眼角
    eye_bs["eyeLookOutLeft"]  = remap(l_pupil_x_ratio, 0.48, 0.65, 0.0, 1.0) # 偏向外眼角
    
    # 右眼（LookIn 是向左看鼻梁，LookOut 是向右看外侧）
    eye_bs["eyeLookInRight"]  = remap(r_pupil_x_ratio, 0.48, 0.65, 0.0, 1.0) # 偏向内眼角
    eye_bs["eyeLookOutRight"] = remap(r_pupil_x_ratio, 0.52, 0.35, 0.0, 1.0) # 偏向外眼角

    # --- 上下看 (LookUp / LookDown) 解算 ---
    # 计算瞳孔中心点偏离眼角水平线段的垂直绝对距离
    l_line_center = (p_eye_l_inner + p_eye_l_outer) * 0.5
    r_line_center = (p_eye_r_inner + p_eye_r_outer) * 0.5

    # 提取垂直方向(Y轴)位移。由于 MediaPipe Y 轴向下，所以用 (中心 - 瞳孔) 算向上
    face_forward = np.array([0,0,1])

    l_eye_y_axis = np.cross(
        face_forward,
        l_eye_axis
    )

    r_eye_y_axis = np.cross(
        face_forward,
        r_eye_axis
    )

    l_eye_y_axis /= (
        np.linalg.norm(l_eye_y_axis)
        + 1e-6
    )

    r_eye_y_axis /= (
        np.linalg.norm(r_eye_y_axis)
        + 1e-6
    )

    l_pupil_y_offset = (
        np.dot(
            p_pupil_l - l_line_center,
            l_eye_y_axis
        )
        /
        eye_dist_base
    )

    r_pupil_y_offset = (
        np.dot(
            p_pupil_r - r_line_center,
            r_eye_y_axis
        )
        /
        eye_dist_base
    )

    eye_bs["eyeLookUpLeft"]    = remap(l_pupil_y_offset, 0.01, 0.06, 0.0, 1.0)
    eye_bs["eyeLookDownLeft"]  = remap(l_pupil_y_offset, -0.01, -0.06, 0.0, 1.0)
    
    eye_bs["eyeLookUpRight"]   = remap(r_pupil_y_offset, 0.01, 0.06, 0.0, 1.0)
    eye_bs["eyeLookDownRight"] = remap(r_pupil_y_offset, -0.01, -0.06, 0.0, 1.0)

    return eye_bs


def solve_brows(lm, eye_dist_base):

    brow_bs = {}
    
    # 基础基准点
    p_nose_bridge = np.array(lm[6]) # 鼻梁中点作为高度不动点
    
    # 左眉毛核心点：内侧(65), 中间(52), 外侧(70)
    # 右眉毛核心点：内侧(295), 中间(282), 外侧(300)
    
    # 计算眉毛内侧抬起 (browInnerUp) —— 左右眼眉内侧距离鼻梁的垂直距离
    l_inner_dist = np.array(lm[65])[1] - p_nose_bridge[1]
    r_inner_dist = np.array(lm[295])[1] - p_nose_bridge[1]
    # 转换号（MediaPipe Y轴向下，减出来负得越多代表抬得越高）
    brow_inner_up = (-l_inner_dist - r_inner_dist) * 0.5 / eye_dist_base
    brow_bs["browInnerUp"] = remap(brow_inner_up, 0.22, 0.38, 0.0, 1.0)
    
    # 计算下压眉 (browDownLeft / browDownRight) —— 眉头压低
    brow_bs["browDownLeft"]  = remap(-l_inner_dist / eye_dist_base, 0.22, 0.15, 0.0, 1.0)
    brow_bs["browDownRight"] = remap(-r_inner_dist / eye_dist_base, 0.22, 0.15, 0.0, 1.0)
    
    # 计算挑眉/外侧抬起 (browOuterUpLeft / browOuterUpRight)
    l_outer_dist = - (np.array(lm[70])[1] - p_nose_bridge[1]) / eye_dist_base
    r_outer_dist = - (np.array(lm[300])[1] - p_nose_bridge[1]) / eye_dist_base
    
    brow_bs["browOuterUpLeft"]  = remap(l_outer_dist, 0.25, 0.42, 0.0, 1.0)
    brow_bs["browOuterUpRight"] = remap(r_outer_dist, 0.25, 0.42, 0.0, 1.0)
    
    return brow_bs


def solve_nose_and_cheeks(lm, eye_dist_base):

    nc_bs = {}
    
    p_nose_bridge = np.array(lm[6])
    p_nose_tip = np.array(lm[4])
    p_eye_l_inner = np.array(lm[133])
    p_eye_r_inner = np.array(lm[362])
    
    p_sneer_l = np.array(lm[129])
    p_sneer_r = np.array(lm[358])
    
    l_sneer = abs(p_eye_l_inner[1] - p_sneer_l[1]) / eye_dist_base
    r_sneer = abs(p_eye_r_inner[1] - p_sneer_r[1]) / eye_dist_base
    
    nc_bs["noseSneerLeft"]  = remap(l_sneer, 0.10, 0.06, 0.0, 1.0)
    nc_bs["noseSneerRight"] = remap(r_sneer, 0.10, 0.06, 0.0, 1.0)
    
    l_cheek = (np.array(lm[50])[1] - np.array(lm[145])[1]) / eye_dist_base
    r_cheek = (np.array(lm[280])[1] - np.array(lm[374])[1]) / eye_dist_base
    
    nc_bs["cheekSquintLeft"]  = remap(l_cheek, 0.22, 0.12, 0.0, 1.0)
    nc_bs["cheekSquintRight"] = remap(r_cheek, 0.22, 0.12, 0.0, 1.0)
    
    # === 鼓脸检测：脸颊宽度 / 眼距 ===
    cheek_width = np.linalg.norm(
        np.array(lm[50]) - np.array(lm[280])
    )
    puff_ratio = cheek_width / (eye_dist_base + 1e-6)


    nc_bs["cheekPuff"] = remap(puff_ratio, 2.35, 2.50, 0.0, 1.0)    
    return nc_bs

def solve_mouth_and_jaw(lm, eye_dist_base, eye_outer_base):
   
    mouth_bs = {}
    
    # === 基础下巴张开与歪斜 ===
    mouth_height = np.linalg.norm(np.array(lm[13]) - np.array(lm[14]))
    mouth_width_raw = np.linalg.norm(np.array(lm[61]) - np.array(lm[291]))  # 改名，不覆盖
    mar = mouth_height / (mouth_width_raw + 1e-6)

    mouth_bs["jawOpen"] = remap(mar, 0.02, 0.35, 0.0, 1.0)    
    mouth_bs["mouthClose"] = remap(mar, 0.12, 0.01, 0.0, 1.0) if mouth_bs["jawOpen"] > 0.05 else 0.0
    
    # 用面部中轴（两内眼角中点）作为参考
    mid_face_x = (np.array(lm[133])[0] + np.array(lm[362])[0]) * 0.5
    jaw_x_offset = (lm[152][0] - mid_face_x) / eye_dist_base
    mouth_bs["jawLeft"]  = remap(jaw_x_offset, 0.0, 0.15, 0.0, 1.0)
    mouth_bs["jawRight"] = remap(jaw_x_offset, -0.15, 0.0, 1.0, 0.0)
    
    # === 精细嘴型与嘴角弧度控制 ===
    p_mouth_l = np.array(lm[61])
    p_mouth_r = np.array(lm[291])
    p_lip_top = np.array(lm[0])
    p_lip_bot = np.array(lm[17])
    p_mouth_center = (p_lip_top + p_lip_bot) * 0.5

    # 嘴角横向总宽度（归一化）
    mouth_ratio = np.linalg.norm(p_mouth_l - p_mouth_r) / eye_dist_base  # 改名为 mouth_ratio

    # 微笑与撇嘴
    l_corner_y = p_mouth_center[1] - p_mouth_l[1]
    r_corner_y = p_mouth_center[1] - p_mouth_r[1]

    mouth_bs["mouthSmileLeft"]  = remap(l_corner_y / eye_dist_base, 0.02, 0.15, 0.0, 1.0)
    mouth_bs["mouthSmileRight"] = remap(r_corner_y / eye_dist_base, 0.02, 0.15, 0.0, 1.0)
    mouth_bs["mouthFrownLeft"]  = remap(l_corner_y / eye_dist_base, -0.01, -0.12, 0.0, 1.0)
    mouth_bs["mouthFrownRight"] = remap(r_corner_y / eye_dist_base, -0.01, -0.12, 0.0, 1.0)

    # 撅嘴与O动作
    mouth_bs["mouthPucker"] = remap(mouth_ratio, 0.82, 0.58, 0.0, 1.0)
    mouth_bs["mouthFunnel"] = remap(mouth_ratio, 0.82, 0.62, 0.0, 1.0) if mouth_bs["jawOpen"] > 0.15 else 0.0

    return mouth_bs

def solve_head_rotation(lm):
   
    # 1:1 提取原版 4 个外围边界骨骼点 (3D 向量)
    p1 = np.array(lm[21])   # 左额头上方骨骼点（左上角）
    p2 = np.array(lm[251])  # 右额头上方骨骼点（右上角）
    p3 = np.array(lm[397])  # 右下颌骨边缘点（右下角）
    p4 = np.array(lm[172])  # 左下颌骨边缘点（左下角）
    
    # 线性插值求下颌左右两点的几何中点（下巴正中心）
    p3mid = (p3 + p4) * 0.5
    
    # 计算双额线段的中点（额头中心）
    mid_point = (p1 + p2) * 0.5
    
    # === 构造正交旋转矩阵（工业标准做法） ===
    # right: 从左额指向右额
    right = p2 - p1
    right = right / (np.linalg.norm(right) + 1e-6)
    
    # down: 从额头中心指向下巴中心（面部纵向）
    down = p3mid - mid_point
    down = down / (np.linalg.norm(down) + 1e-6)
    
    # forward: right × down → 面部前方（鼻子朝向）
    forward = np.cross(right, down)
    forward = forward / (np.linalg.norm(forward) + 1e-6)
    
    # 重新正交化 down: forward × right
    down = np.cross(forward, right)
    
    # 构造 3×3 旋转矩阵：列向量为 [right, down, forward]
    R = np.column_stack([right, down, forward])
    
    # === 从旋转矩阵提取 Euler 角（用于外部兼容） ===
    # Yaw (绕Y轴): atan2(R[0,2], R[2,2])
    rad_y = np.arctan2(R[0, 2], R[2, 2])
    # Pitch (绕X轴): atan2(-R[1,2], sqrt(R[0,2]^2 + R[2,2]^2))
    rad_x = np.arctan2(-R[1, 2], np.sqrt(R[0, 2]**2 + R[2, 2]**2))
    # Roll (绕Z轴): atan2(R[1,0], R[1,1])
    rad_z = np.arctan2(R[1, 0], R[1, 1])
    
    deg_x = rad_x * 180.0 / np.pi
    deg_y = rad_y * 180.0 / np.pi
    deg_z = rad_z * 180.0 / np.pi
    
    # 计算面部近似的 3D 物理尺寸
    width = np.linalg.norm(p2 - p1)         
    height = np.linalg.norm(p3mid - mid_point) 
    
    # 头部的 3D 空间绝对位置中心点
    position = (mid_point + p3mid) * 0.5
    
    return {
        "x": deg_x, "y": deg_y, "z": deg_z,       
        "rad_x": rad_x, "rad_y": rad_y, "rad_z": rad_z, 
        "width": width,
        "height": height,
        "position": position,
        "R": R          # 新增：3×3 旋转矩阵，给 transform_to_local_space 用
    }
def calc_all_arkit_coefficients(smooth_mesh_mock):
    lm = smooth_mesh_mock
    if len(lm) == 0 or np.all(np.array(lm[6]) == 0):
       # 兜底保障返回，防止下游空值报错崩溃
        return {k: 0.0 for k in ["jawOpen", "eyeBlinkLeft", "mouthSmileLeft"]}   
    computed_bs = {}
    
    #优先提取全局头部的 3D 刚性旋转姿态数据
    head_pose = solve_head_rotation(lm)
    
    # 在这里生成完全剥离了旋转形变的“绝对正脸 3D 点阵”
    local_lm = transform_to_local_space(lm, head_pose)
    
    # 计算全局刚性归一化基准（在无形变的局部空间计算，绝对精确）
    eyeInnerDistance = np.linalg.norm(np.array(local_lm[133]) - np.array(local_lm[362]))
    if eyeInnerDistance == 0: eyeInnerDistance = 0.001 
    
    eyeOuterDistance = np.linalg.norm(np.array(local_lm[130]) - np.array(local_lm[263]))
    if eyeOuterDistance == 0: eyeOuterDistance = 0.001

    #将所有表情解算器全部升级为接收 local_lm 局部空间点阵
    computed_bs.update(solve_eyes(local_lm, eyeInnerDistance, eyeOuterDistance))
    computed_bs.update(solve_brows(local_lm, eyeInnerDistance))
    computed_bs.update(solve_nose_and_cheeks(local_lm, eyeInnerDistance))
    computed_bs.update(solve_mouth_and_jaw(local_lm, eyeInnerDistance, eyeOuterDistance))

    # 5. 统一进行边界安全限制 [0.0, 1.0]
    for k, v in computed_bs.items():
        if not k.startswith("_"):                    
            computed_bs[k] = max(0.0, min(1.0, v))

    # 6. 将解算好的头部欧拉角（角度制）一并注入输出，用于同步驱动虚拟主播的转头
    computed_bs["_head_deg_x"] = head_pose["x"]
    computed_bs["_head_deg_y"] = head_pose["y"]
    computed_bs["_head_deg_z"] = head_pose["z"]

    return computed_bs