import numpy as np

def remap(val, imin, imax, omin, omax):
    if imax == imin: return omin
    if imin < imax:
        clamped = max(imin, min(imax, val))
    else:
        clamped = max(imax, min(imin, val))
    return omin + (clamped - imin) * (omax - omin) / (imax - imin)

def transform_to_local_space(lm, head_pose):
    origin = head_pose["position"]
    R = head_pose["R"]          
    local_lm = {}
    for idx, pt in lm.items():
        local_lm[idx] = R.T @ (np.array(pt) - origin)
    return local_lm

# ==================== 1. 眼部阈值精密校准 ====================
def solve_eyes(lm, eye_dist_base, eye_outer_base):
    eye_bs = {}
    
    # 定义关键点
    p_eye_l_outer = np.array(lm[130]) 
    p_eye_l_inner = np.array(lm[133])
    p_eye_r_inner = np.array(lm[362])
    p_eye_r_outer = np.array(lm[263])

    l_top_points = [np.array(lm[159]), np.array(lm[158]), np.array(lm[160])]
    l_bot_points = [np.array(lm[145]), np.array(lm[153]), np.array(lm[144])]
    r_top_points = [np.array(lm[386]), np.array(lm[385]), np.array(lm[387])]
    r_bot_points = [np.array(lm[374]), np.array(lm[373]), np.array(lm[380])]

    l_vert_dist = sum(abs(t[1] - b[1]) for t, b in zip(l_top_points, l_bot_points)) / 3.0
    r_vert_dist = sum(abs(t[1] - b[1]) for t, b in zip(r_top_points, r_bot_points)) / 3.0

    l_eye_width = np.linalg.norm(p_eye_l_outer[:2] - p_eye_l_inner[:2]) + 1e-6
    r_eye_width = np.linalg.norm(p_eye_r_outer[:2] - p_eye_r_inner[:2]) + 1e-6
    
    l_open_ratio = l_vert_dist / l_eye_width
    r_open_ratio = r_vert_dist / r_eye_width

    # --- 1. 眼睑阈值：优化瞪眼灵敏度 ---
    eye_bs["eyeBlinkLeft"] = remap(l_open_ratio, 0.24, 0.10, 0.0, 1.0)
    eye_bs["eyeBlinkRight"] = remap(r_open_ratio, 0.24, 0.10, 0.0, 1.0)

    # 将 imax 从 0.40 提高到 0.55，瞪眼现在需要更大幅度的张开才能触发
    eye_bs["eyeWideLeft"] = remap(l_open_ratio, 0.32, 0.55, 0.0, 1.0)
    eye_bs["eyeWideRight"] = remap(r_open_ratio, 0.32, 0.55, 0.0, 1.0)

    eye_bs["eyeSquintLeft"] = remap(l_open_ratio, 0.22, 0.12, 0.0, 1.0) * (1.0 - eye_bs["eyeBlinkLeft"])
    eye_bs["eyeSquintRight"] = remap(r_open_ratio, 0.22, 0.12, 0.0, 1.0) * (1.0 - eye_bs["eyeBlinkRight"])

    # --- 2. 瞳孔阈值：压缩区间，大幅提升灵敏度 ---
    p_pupil_l = np.array(lm[468])[:2]
    p_pupil_r = np.array(lm[473])[:2]

    l_pupil_ratio = np.dot(p_pupil_l - p_eye_l_inner[:2], p_eye_l_outer[:2] - p_eye_l_inner[:2]) / (l_eye_width**2)
    r_pupil_ratio = np.dot(p_pupil_r - p_eye_r_inner[:2], p_eye_r_outer[:2] - p_eye_r_inner[:2]) / (r_eye_width**2)

    # 缩短映射区间（如 0.25 -> 0.35），瞳孔转动一点点就能触发完整权重变化
    eye_bs["eyeLookInLeft"]   = remap(l_pupil_ratio, 0.46, 0.35, 0.0, 1.0)
    eye_bs["eyeLookOutLeft"]  = remap(l_pupil_ratio, 0.54, 0.65, 0.0, 1.0)
    eye_bs["eyeLookInRight"]  = remap(r_pupil_ratio, 0.54, 0.65, 0.0, 1.0)
    eye_bs["eyeLookOutRight"] = remap(r_pupil_ratio, 0.46, 0.35, 0.0, 1.0)

    # --- 3. 上下转动：同步提升灵敏度 ---
    l_line_center_y = (p_eye_l_inner[1] + p_eye_l_outer[1]) * 0.5
    r_line_center_y = (p_eye_r_inner[1] + p_eye_r_outer[1]) * 0.5
    
    l_pupil_y_offset = (p_pupil_l[1] - l_line_center_y) / l_eye_width
    r_pupil_y_offset = (p_pupil_r[1] - r_line_center_y) / r_eye_width

    # 将 imax 从 0.15 缩小到 0.08，动作响应会变得非常跟手
    eye_bs["eyeLookUpLeft"]    = remap(l_pupil_y_offset, -0.02, -0.08, 0.0, 1.0)
    eye_bs["eyeLookDownLeft"]  = remap(l_pupil_y_offset, 0.02, 0.08, 0.0, 1.0)
    eye_bs["eyeLookUpRight"]   = remap(r_pupil_y_offset, -0.02, -0.08, 0.0, 1.0)
    eye_bs["eyeLookDownRight"] = remap(r_pupil_y_offset, 0.02, 0.08, 0.0, 1.0)

    return eye_bs

# ==================== 2. 眉毛阈值精密校准 ====================
def solve_brows(lm, eye_dist_base):
    brow_bs = {}
    p_nose_bridge = np.array(lm[6]) 
    
    # Y轴向下，数值减出来越小（负得越多）代表眉毛越高
    l_inner_dist = np.array(lm[65])[1] - p_nose_bridge[1]
    r_inner_dist = np.array(lm[295])[1] - p_nose_bridge[1]
    
    brow_inner_up = (-l_inner_dist - r_inner_dist) * 0.5 / eye_dist_base
    
    # 抬眉与压眉阈值
    brow_bs["browInnerUp"] = remap(brow_inner_up, 0.72, 0.90, 0.0, 1.0)
    brow_bs["browDownLeft"]  = remap(-l_inner_dist / eye_dist_base, 0.72, 0.60, 0.0, 1.0)
    brow_bs["browDownRight"] = remap(-r_inner_dist / eye_dist_base, 0.72, 0.60, 0.0, 1.0)
    
    l_outer_dist = - (np.array(lm[70])[1] - p_nose_bridge[1]) / eye_dist_base
    r_outer_dist = - (np.array(lm[300])[1] - p_nose_bridge[1]) / eye_dist_base
    
    brow_bs["browOuterUpLeft"]  = remap(l_outer_dist, 0.75, 0.95, 0.0, 1.0)
    brow_bs["browOuterUpRight"] = remap(r_outer_dist, 0.75, 0.95, 0.0, 1.0)
    return brow_bs

# ==================== 3. 鼻子与脸颊阈值精密校准 (彻底解决持续鼓嘴) ====================
def solve_nose_and_cheeks(lm, eye_dist_base):
    nc_bs = {}
    p_eye_l_inner = np.array(lm[133])
    p_eye_r_inner = np.array(lm[362])
    
    # 撇嘴/皱鼻
    l_sneer = abs(p_eye_l_inner[1] - lm[129][1]) / eye_dist_base
    r_sneer = abs(p_eye_r_inner[1] - lm[358][1]) / eye_dist_base
    nc_bs["noseSneerLeft"]  = remap(l_sneer, 0.55, 0.42, 0.0, 1.0)
    nc_bs["noseSneerRight"] = remap(r_sneer, 0.55, 0.42, 0.0, 1.0)
    
    # 眯眼抬脸颊
    l_cheek = abs(lm[50][1] - lm[145][1]) / eye_dist_base
    r_cheek = abs(lm[280][1] - lm[374][1]) / eye_dist_base
    nc_bs["cheekSquintLeft"]  = remap(l_cheek, 0.60, 0.45, 0.0, 1.0)
    nc_bs["cheekSquintRight"] = remap(r_cheek, 0.60, 0.45, 0.0, 1.0)
    
    # --- 鼓脸阈值重新对齐 ---
    # 修正：在精确计算的分母下，静态脸宽比值大约在 2.65 ~ 2.70。鼓嘴会超过 2.85
    cheek_width = np.linalg.norm(np.array(lm[50][:2]) - np.array(lm[280][:2]))
    puff_ratio = cheek_width / eye_dist_base
    
# 把鼓嘴的阈值改成匹配您的数据
    nc_bs["cheekPuff"] = remap(puff_ratio, 2.72, 2.95, 0.0, 1.0)
    return nc_bs

# ==================== 4. 嘴部与下巴阈值精密校准 ====================
def solve_mouth_and_jaw(lm, eye_dist_base, eye_outer_base):
    mouth_bs = {}
    
    # 基础嘴部高度与宽度
    mouth_height = np.linalg.norm(np.array(lm[13][:2]) - np.array(lm[14][:2]))
    mouth_width_raw = np.linalg.norm(np.array(lm[61][:2]) - np.array(lm[291][:2]))
    mar = mouth_height / (mouth_width_raw + 1e-6)

    # 嘴巴张开度：调大起始区间，防止微张嘴误触发
    mouth_bs["jawOpen"] = remap(mar, 0.08, 0.65, 0.0, 1.0)    
    mouth_bs["mouthClose"] = remap(mar, 0.12, 0.01, 0.0, 1.0) if mouth_bs["jawOpen"] > 0.05 else 0.0
    
    # 下巴歪斜：增加死区，平滑抖动
    mid_face_x = (lm[133][0] + lm[362][0]) * 0.5
    jaw_x_offset = (lm[152][0] - mid_face_x) / eye_dist_base
    mouth_bs["jawLeft"]  = remap(jaw_x_offset, 0.08, 0.20, 0.0, 1.0)
    mouth_bs["jawRight"] = remap(jaw_x_offset, -0.08, -0.20, 0.0, 1.0)
    
    p_mouth_l = np.array(lm[61])
    p_mouth_r = np.array(lm[291])
    p_mouth_center = (np.array(lm[0]) + np.array(lm[17])) * 0.5
    mouth_ratio = np.linalg.norm(p_mouth_l[:2] - p_mouth_r[:2]) / eye_dist_base

    # 微笑与撇嘴（基于局部 Y 轴高度差）
    l_corner_y = p_mouth_center[1] - p_mouth_l[1]
    r_corner_y = p_mouth_center[1] - p_mouth_r[1]

    # 稍微压低了灵敏度，使表情过渡更自然
    mouth_bs["mouthSmileLeft"]  = remap(l_corner_y / eye_dist_base, 0.04, 0.18, 0.0, 1.0)
    mouth_bs["mouthSmileRight"] = remap(r_corner_y / eye_dist_base, 0.04, 0.18, 0.0, 1.0)

    # 撇嘴（Frown）逻辑修正
    mouth_bs["mouthFrownLeft"]  = remap(-l_corner_y / eye_dist_base, 0.04, 0.18, 0.0, 1.0)
    mouth_bs["mouthFrownRight"] = remap(-r_corner_y / eye_dist_base, 0.04, 0.18, 0.0, 1.0)

    
    # 1. 计算 Funnel (O型嘴)：只有张嘴超过 0.25 时才可能触发
    funnel_raw = remap(mouth_ratio, 1.05, 0.88, 0.0, 1.0)
    mouth_bs["mouthFunnel"] = funnel_raw if mouth_bs["jawOpen"] > 0.25 else 0.0
    
    # 2. 计算 Pucker (撅嘴)：
    # 逻辑：Pucker 和 Funnel 互斥。当 Funnel 权重 > 0.1 时，立即抑制 Pucker
    # 引入一个阈值平滑因子，避免突然跳变
    pucker_raw = remap(mouth_ratio, 1.05, 0.82, 0.0, 1.0)
    
    # 使用 if-else 强制互斥，而不是单纯的乘法，这样更安全
    if mouth_bs["mouthFunnel"] > 0.1:
        mouth_bs["mouthPucker"] = 0.0
    else:
        # 当没有 Funnel 时，允许 Pucker 存在，但限制其最大值，防止嘴唇过度挤压
        mouth_bs["mouthPucker"] = min(pucker_raw, 0.8)

    return mouth_bs

# ==================== 5. 头部刚体解算 (带补全弧度输出) ====================
def solve_head_rotation(lm):
    p1 = np.array(lm[21])   
    p2 = np.array(lm[251])  
    p3 = np.array(lm[397])  
    p4 = np.array(lm[172])  
    
    p3mid = (p3 + p4) * 0.5
    mid_point = (p1 + p2) * 0.5
    
    right = p2 - p1
    right = right / (np.linalg.norm(right) + 1e-6)
    
    down = p3mid - mid_point
    down = down / (np.linalg.norm(down) + 1e-6)
    
    forward = np.cross(right, down)
    forward = forward / (np.linalg.norm(forward) + 1e-6)
    
    down = np.cross(forward, right)
    R = np.column_stack([right, down, forward])
    
    rad_y = np.arctan2(R[0, 2], R[2, 2])
    rad_x = np.arctan2(-R[1, 2], np.sqrt(R[0, 2]**2 + R[2, 2]**2))
    rad_z = np.arctan2(R[1, 0], R[1, 1])
    
    return {
        "x": rad_x * 180.0 / np.pi, "y": rad_y * 180.0 / np.pi, "z": rad_z * 180.0 / np.pi,      
        "rad_x": rad_x, "rad_y": rad_y, "rad_z": rad_z, # 👈 确保满足 ldtransformer.py 的上游调用
        "width": np.linalg.norm(p2 - p1),
        "height": np.linalg.norm(p3mid - mid_point),
        "position": (mid_point + p3mid) * 0.5,
        "R": R          
    }

# ==================== 6. 总控数据分发 ====================
def calc_all_arkit_coefficients(smooth_mesh_mock):
    lm = smooth_mesh_mock
    if len(lm) == 0 or np.all(np.array(lm[6]) == 0):
        return {k: 0.0 for k in ["jawOpen", "eyeBlinkLeft", "mouthSmileLeft"]}   
    computed_bs = {}
    
    head_pose = solve_head_rotation(lm)
    local_lm = transform_to_local_space(lm, head_pose)
    
    # 修复：内眼角对齐为标准的 133 和 362
    eyeInnerDistance = np.linalg.norm(np.array(local_lm[133]) - np.array(local_lm[362]))
    if eyeInnerDistance == 0: eyeInnerDistance = 0.001 
    
    # 外眼角对齐为标准的 130 和 263
    eyeOuterDistance = np.linalg.norm(np.array(local_lm[130]) - np.array(local_lm[263]))
    if eyeOuterDistance == 0: eyeOuterDistance = 0.001

    computed_bs.update(solve_eyes(local_lm, eyeInnerDistance, eyeOuterDistance))
    computed_bs.update(solve_brows(local_lm, eyeInnerDistance))
    computed_bs.update(solve_nose_and_cheeks(local_lm, eyeInnerDistance))
    computed_bs.update(solve_mouth_and_jaw(local_lm, eyeInnerDistance, eyeOuterDistance))

    for k, v in computed_bs.items():
        if not k.startswith("_"):                    
            computed_bs[k] = max(0.0, min(1.0, float(v)))

    computed_bs["_head_deg_x"] = head_pose["x"]
    computed_bs["_head_deg_y"] = head_pose["y"]
    computed_bs["_head_deg_z"] = head_pose["z"]
    print(f"DEBUG: Head Pose -> X:{head_pose['x']:.2f}, Y:{head_pose['y']:.2f}, Z:{head_pose['z']:.2f}")
    return computed_bs