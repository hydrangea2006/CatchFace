import numpy as np
from collections import defaultdict

# ==================== 全局状态 ====================

CALIBRATION = {
    "brow_inner_base": None,
    "sneer_base": None,
    "cheek_base": None,
    "mouth_smile_l_base": None,
    "mouth_smile_r_base": None,
    "jaw_x_base": None,
    "mouth_ratio_base": None,
    "l_eye_base": None,
    "r_eye_base": None,
}

CALIBRATION_BUFFER = defaultdict(list)

PREV_BS = {}
SMOOTH_ALPHA = 0.3
CALIBRATION_FRAME_COUNT = 0
CALIBRATION_FRAMES_NEEDED = 30
IS_CALIBRATED = False


def reset_calibration():
    global CALIBRATION_FRAME_COUNT, IS_CALIBRATED, CALIBRATION_BUFFER, PREV_BS
    for key in CALIBRATION:
        CALIBRATION[key] = None
    CALIBRATION_BUFFER.clear()
    CALIBRATION_FRAME_COUNT = 0
    IS_CALIBRATED = False
    PREV_BS.clear()
    print(">>> 校准已重置！请保持中性平视表情 2 秒...")


def update_calibration_buffer(key, value):
    if not IS_CALIBRATED and CALIBRATION_FRAME_COUNT < CALIBRATION_FRAMES_NEEDED:
        CALIBRATION_BUFFER[key].append(value)


def finalize_calibration():
    global IS_CALIBRATED
    if len(CALIBRATION_BUFFER.get("l_eye_base", [])) < CALIBRATION_FRAMES_NEEDED // 2:
        print(">>> 警告：校准数据不足")
        IS_CALIBRATED = True
        return
    
    for key in CALIBRATION.keys():
        if key in CALIBRATION_BUFFER and CALIBRATION_BUFFER[key]:
            CALIBRATION[key] = np.median(CALIBRATION_BUFFER[key])
    
    IS_CALIBRATED = True
    print(">>> 校准完成！")
    for key, val in CALIBRATION.items():
        if val is not None:
            print(f"    {key:20} = {val:.4f}")


def apply_smoothing(current_bs):
    global PREV_BS
    if not PREV_BS:
        PREV_BS = {k: v for k, v in current_bs.items() 
                   if not k.startswith("_") and not k.startswith("eyeLook")}
        return current_bs
    
    smoothed = {}
    for k, v in current_bs.items():
        if k.startswith("_") or k.startswith("eyeLook"):
            smoothed[k] = v
        else:
            prev_val = PREV_BS.get(k, v)
            smoothed[k] = SMOOTH_ALPHA * v + (1 - SMOOTH_ALPHA) * prev_val
    
    for k, v in smoothed.items():
        if not k.startswith("_") and not k.startswith("eyeLook"):
            PREV_BS[k] = v
    
    return smoothed


def smoothstep(edge0, edge1, x):
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0 + 1e-8)))
    return t * t * (3.0 - 2.0 * t)


def smooth_remap(val, imin, imax, omin=0.0, omax=1.0):
    if imax == imin:
        return omin
    if imin < imax:
        t = (val - imin) / (imax - imin)
    else:
        t = (val - imax) / (imin - imax)
    t = max(0.0, min(1.0, t))
    t_smooth = t * t * (3.0 - 2.0 * t)
    return omin + t_smooth * (omax - omin)


def transform_to_local_space(lm, head_pose):
    origin = head_pose["position"]
    R = head_pose["R"]
    local_lm = {}
    for idx, pt in lm.items():
        local_lm[idx] = R.T @ (np.array(pt) - origin)
    return local_lm


# ==================== 1. 眼部 ====================

def solve_eyes(lm, eye_dist_base, eye_outer_base):
    eye_bs = {}

    p_eye_l_outer = np.array(lm[130])
    p_eye_l_inner = np.array(lm[133])
    p_eye_r_inner = np.array(lm[362])
    p_eye_r_outer = np.array(lm[263])

    # 使用真正的上下眼睑关键点
    l_eye_top = np.array(lm[159])  # 左眼上眼睑
    l_eye_bottom = np.array(lm[145])  # 左眼下眼睑
    r_eye_top = np.array(lm[386])  # 右眼上眼睑
    r_eye_bottom = np.array(lm[374])  # 右眼下眼睑

    l_top_points = [np.array(lm[159]), np.array(lm[158]), np.array(lm[160])]
    l_bot_points = [np.array(lm[145]), np.array(lm[153]), np.array(lm[144])]
    r_top_points = [np.array(lm[386]), np.array(lm[385]), np.array(lm[387])]
    r_bot_points = [np.array(lm[374]), np.array(lm[373]), np.array(lm[380])]

    def ear(top_pts, bot_pts, left_pt, right_pt):
        vert_dists = [np.linalg.norm(t - b) for t, b in zip(top_pts, bot_pts)]
        avg_vert = sum(vert_dists) / len(vert_dists)
        horiz = np.linalg.norm(left_pt[:2] - right_pt[:2]) + 1e-6
        return avg_vert / horiz, horiz

    l_open_ratio, l_eye_width = ear(l_top_points, l_bot_points, p_eye_l_outer, p_eye_l_inner)
    r_open_ratio, r_eye_width = ear(r_top_points, r_bot_points, p_eye_r_outer, p_eye_r_inner)

    update_calibration_buffer("l_eye_base", l_open_ratio)
    update_calibration_buffer("r_eye_base", r_open_ratio)

    if CALIBRATION["l_eye_base"] is not None:
        l_blink_delta = CALIBRATION["l_eye_base"] - l_open_ratio
        r_blink_delta = CALIBRATION["r_eye_base"] - r_open_ratio

        eye_bs["eyeBlinkLeft"]  = smooth_remap(l_blink_delta, 0.03, 0.12, 0.0, 1.0)
        eye_bs["eyeBlinkRight"] = smooth_remap(r_blink_delta, 0.03, 0.12, 0.0, 1.0)

        l_wide_delta = l_open_ratio - CALIBRATION["l_eye_base"]
        r_wide_delta = r_open_ratio - CALIBRATION["r_eye_base"]
        eye_bs["eyeWideLeft"]  = smooth_remap(l_wide_delta, 0.02, 0.08, 0.0, 1.0)
        eye_bs["eyeWideRight"] = smooth_remap(r_wide_delta, 0.02, 0.08, 0.0, 1.0)

        eye_bs["eyeSquintLeft"]  = smooth_remap(l_blink_delta, 0.01, 0.06, 0.0, 1.0) * (1.0 - eye_bs["eyeBlinkLeft"])
        eye_bs["eyeSquintRight"] = smooth_remap(r_blink_delta, 0.01, 0.06, 0.0, 1.0) * (1.0 - eye_bs["eyeBlinkRight"])
    else:
        for k in ["eyeBlinkLeft", "eyeBlinkRight", "eyeWideLeft", "eyeWideRight", 
                  "eyeSquintLeft", "eyeSquintRight"]:
            eye_bs[k] = 0.0

    # 瞳孔转动
    if 468 in lm and 473 in lm:
        p_pupil_l = np.array(lm[468])[:2]
        p_pupil_r = np.array(lm[473])[:2]
    else:
        p_pupil_l = (p_eye_l_inner[:2] + p_eye_l_outer[:2]) * 0.5
        p_pupil_r = (p_eye_r_inner[:2] + p_eye_r_outer[:2]) * 0.5

    # 水平方向
    def get_horizontal_ratio(pupil, inner, outer):
        eye_vec = outer[:2] - inner[:2]
        eye_len = np.linalg.norm(eye_vec)
        if eye_len < 1e-6:
            return 0.5
        pupil_vec = pupil - inner[:2]
        proj = np.dot(pupil_vec, eye_vec) / eye_len
        ratio = proj / eye_len
        return np.clip(ratio, 0.0, 1.0)

    l_hratio = get_horizontal_ratio(p_pupil_l, p_eye_l_inner, p_eye_l_outer)
    r_hratio = get_horizontal_ratio(p_pupil_r, p_eye_r_inner, p_eye_r_outer)

    eye_bs["eyeLookInLeft"] = smooth_remap(l_hratio, 0.1, 0.35, 1.0, 0.0)
    eye_bs["eyeLookOutLeft"] = smooth_remap(l_hratio, 0.65, 0.9, 0.0, 1.0)
    eye_bs["eyeLookInRight"] = smooth_remap(r_hratio, 0.65, 0.9, 0.0, 1.0)
    eye_bs["eyeLookOutRight"] = smooth_remap(r_hratio, 0.1, 0.35, 1.0, 0.0)

    # 在 solve_eyes 函数中，替换垂直方向计算部分（约第220-250行）

    # 垂直方向 - 使用内外眼角中线（不随眨眼漂移）
    def get_vertical_ratio_fixed(pupil, inner, outer, eye_width):
        # 计算内外眼角的 Y 轴中心点（作为固定的眼睛中线）
        center_y = (inner[1] + outer[1]) * 0.5
        
        # 瞳孔偏离中线的绝对距离（MediaPipe中 Y 向下为正）
        # pupil[1] < center_y 代表往上看，pupil[1] > center_y 代表往下看
        y_offset = pupil[1] - center_y
        
        # 用眼角宽度（eye_width）对这个偏移量进行归一化，消除前后距离（近大远小）的影响
        # 正常平视时，ratio 应该在 0 附近；向上看为负，向下看为正
        ratio = y_offset / (eye_width + 1e-6)
        return ratio

    # 调用新函数计算左右眼垂直偏移
    l_vratio = get_vertical_ratio_fixed(p_pupil_l, p_eye_l_inner, p_eye_l_outer, l_eye_width)
    r_vratio = get_vertical_ratio_fixed(p_pupil_r, p_eye_r_inner, p_eye_r_outer, r_eye_width)

    # 映射到 ARKit 权重
    # 向上看：l_vratio 为负值（例如 -0.08），映射到 [1.0, 0.0]
    eye_bs["eyeLookUpLeft"]    = smooth_remap(l_vratio, -0.08, 0.00, 1.0, 0.0)
    eye_bs["eyeLookDownLeft"]  = smooth_remap(l_vratio, 0.00, 0.08, 0.0, 1.0)

    eye_bs["eyeLookUpRight"]   = smooth_remap(r_vratio, -0.08, 0.00, 1.0, 0.0)
    eye_bs["eyeLookDownRight"] = smooth_remap(r_vratio, 0.00, 0.08, 0.0, 1.0)
    return eye_bs

# ==================== 2. 眉毛 ====================

def solve_brows(lm, eye_dist_base):
    brow_bs = {}
    p_nose_bridge = np.array(lm[6])

    l_inner_raw = (p_nose_bridge[1] - np.array(lm[65])[1]) / (eye_dist_base + 1e-6)
    r_inner_raw = (p_nose_bridge[1] - np.array(lm[295])[1]) / (eye_dist_base + 1e-6)
    avg_inner = (l_inner_raw + r_inner_raw) * 0.5

    l_outer_raw = (p_nose_bridge[1] - np.array(lm[70])[1]) / (eye_dist_base + 1e-6)
    r_outer_raw = (p_nose_bridge[1] - np.array(lm[300])[1]) / (eye_dist_base + 1e-6)

    update_calibration_buffer("brow_inner_base", avg_inner)

    if CALIBRATION["brow_inner_base"] is not None:
        inner_delta = avg_inner - CALIBRATION["brow_inner_base"]
        brow_bs["browInnerUp"] = smooth_remap(inner_delta, 0.01, 0.10, 0.0, 1.0)
        brow_bs["browDownLeft"] = smooth_remap(-inner_delta, 0.01, 0.08, 0.0, 1.0)
        brow_bs["browDownRight"] = smooth_remap(-inner_delta, 0.01, 0.08, 0.0, 1.0)

        l_outer_delta = l_outer_raw - CALIBRATION["brow_inner_base"]
        r_outer_delta = r_outer_raw - CALIBRATION["brow_inner_base"]
        brow_bs["browOuterUpLeft"] = smooth_remap(l_outer_delta, 0.01, 0.10, 0.0, 1.0)
        brow_bs["browOuterUpRight"] = smooth_remap(r_outer_delta, 0.01, 0.10, 0.0, 1.0)
    else:
        for k in ["browInnerUp", "browDownLeft", "browDownRight", "browOuterUpLeft", "browOuterUpRight"]:
            brow_bs[k] = 0.0

    return brow_bs


# ==================== 3. 鼻子与脸颊 ====================

def solve_nose_and_cheeks(lm, eye_dist_base):
    nc_bs = {}
    p_eye_l_inner = np.array(lm[133])
    p_eye_r_inner = np.array(lm[362])

    l_sneer_raw = abs(p_eye_l_inner[1] - lm[129][1]) / (eye_dist_base + 1e-6)
    r_sneer_raw = abs(p_eye_r_inner[1] - lm[358][1]) / (eye_dist_base + 1e-6)
    avg_sneer = (l_sneer_raw + r_sneer_raw) * 0.5

    l_cheek_raw = abs(lm[50][1] - lm[145][1]) / (eye_dist_base + 1e-6)
    r_cheek_raw = abs(lm[280][1] - lm[374][1]) / (eye_dist_base + 1e-6)
    avg_cheek = (l_cheek_raw + r_cheek_raw) * 0.5

    update_calibration_buffer("sneer_base", avg_sneer)
    update_calibration_buffer("cheek_base", avg_cheek)

    if CALIBRATION["sneer_base"] is not None:
        sneer_delta = avg_sneer - CALIBRATION["sneer_base"]
        nc_bs["noseSneerLeft"] = smooth_remap(sneer_delta, 0.02, 0.10, 0.0, 1.0)
        nc_bs["noseSneerRight"] = smooth_remap(sneer_delta, 0.02, 0.10, 0.0, 1.0)

        cheek_delta = CALIBRATION["cheek_base"] - avg_cheek
        nc_bs["cheekSquintLeft"] = smooth_remap(cheek_delta, 0.01, 0.08, 0.0, 1.0)
        nc_bs["cheekSquintRight"] = smooth_remap(cheek_delta, 0.01, 0.08, 0.0, 1.0)
    else:
        for k in ["noseSneerLeft", "noseSneerRight", "cheekSquintLeft", "cheekSquintRight"]:
            nc_bs[k] = 0.0

    cheek_width = np.linalg.norm(np.array(lm[50][:2]) - np.array(lm[280][:2]))
    puff_ratio = cheek_width / (eye_dist_base + 1e-6)
    nc_bs["cheekPuff"] = smooth_remap(puff_ratio, 2.7, 2.9, 0.0, 1.0)

    return nc_bs


# ==================== 4. 嘴部与下巴 ====================

def solve_mouth_and_jaw(lm, eye_dist_base, eye_outer_base):
    mouth_bs = {}

    # 1. 基础开口度与横宽
    mouth_height = np.linalg.norm(np.array(lm[13][:2]) - np.array(lm[14][:2]))
    mouth_width_raw = np.linalg.norm(np.array(lm[61][:2]) - np.array(lm[291][:2]))
    mar = mouth_height / (mouth_width_raw + 1e-6)

    mouth_bs["jawOpen"] = smooth_remap(mar, 0.05, 0.5, 0.0, 1.0)
    mouth_bs["mouthClose"] = smooth_remap(mar, 0.1, 0.02, 0.0, 1.0) if mouth_bs["jawOpen"] < 0.1 else 0.0

    # 2. 下巴左右
    mid_face_x = (lm[133][0] + lm[362][0]) * 0.5
    jaw_x_offset = (lm[152][0] - mid_face_x) / (eye_dist_base + 1e-6)
    update_calibration_buffer("jaw_x_base", jaw_x_offset)

    if CALIBRATION["jaw_x_base"] is not None:
        jaw_delta = jaw_x_offset - CALIBRATION["jaw_x_base"]
        mouth_bs["jawLeft"] = smooth_remap(jaw_delta, 0.03, 0.12, 0.0, 1.0)
        mouth_bs["jawRight"] = smooth_remap(-jaw_delta, 0.03, 0.12, 0.0, 1.0)
    else:
        mouth_bs["jawLeft"] = mouth_bs["jawRight"] = 0.0

    # 3. 微笑/皱眉 (移除 abs()，保留方向符号)
    p_mouth_l = np.array(lm[61])
    p_mouth_r = np.array(lm[291])
    p_mouth_center = (np.array(lm[0]) + np.array(lm[17])) * 0.5

    # 显式计算相对 Y 轴位移 (OpenCV/MediaPipe坐标系下，Y向下为正，所以用 center - corner 表示向上扬)
    l_corner_offset = (p_mouth_center[1] - p_mouth_l[1]) / (eye_dist_base + 1e-6)
    r_corner_offset = (p_mouth_center[1] - p_mouth_r[1]) / (eye_dist_base + 1e-6)

    update_calibration_buffer("mouth_smile_l_base", l_corner_offset)
    update_calibration_buffer("mouth_smile_r_base", r_corner_offset)

    if CALIBRATION["mouth_smile_l_base"] is not None:
        l_smile_delta = l_corner_offset - CALIBRATION["mouth_smile_l_base"]
        r_smile_delta = r_corner_offset - CALIBRATION["mouth_smile_r_base"]
        
        # 左嘴角判定
        if l_smile_delta > 0:
            mouth_bs["mouthSmileLeft"] = smooth_remap(l_smile_delta, 0.005, 0.08, 0.0, 1.0)
            mouth_bs["mouthFrownLeft"] = 0.0
        else:
            mouth_bs["mouthSmileLeft"] = 0.0
            mouth_bs["mouthFrownLeft"] = smooth_remap(-l_smile_delta, 0.005, 0.08, 0.0, 1.0)
            
        # 右嘴角判定
        if r_smile_delta > 0:
            mouth_bs["mouthSmileRight"] = smooth_remap(r_smile_delta, 0.005, 0.08, 0.0, 1.0)
            mouth_bs["mouthFrownRight"] = 0.0
        else:
            mouth_bs["mouthSmileRight"] = 0.0
            mouth_bs["mouthFrownRight"] = smooth_remap(-r_smile_delta, 0.005, 0.08, 0.0, 1.0)
    else:
        mouth_bs["mouthSmileLeft"] = mouth_bs["mouthSmileRight"] = 0.0
        mouth_bs["mouthFrownLeft"] = mouth_bs["mouthFrownRight"] = 0.0

    # 4. 漏斗/撅嘴 (引入 MAR 解耦)
    mouth_ratio = np.linalg.norm(p_mouth_l[:2] - p_mouth_r[:2]) / (eye_dist_base + 1e-6)
    update_calibration_buffer("mouth_ratio_base", mouth_ratio)

    if CALIBRATION["mouth_ratio_base"] is not None:
        ratio_delta = CALIBRATION["mouth_ratio_base"] - mouth_ratio
        
        if mar > 0.15: # 开口状态 -> 漏斗嘴 (Oh)
            mouth_bs["mouthFunnel"] = smooth_remap(ratio_delta, 0.02, 0.22, 0.0, 1.0)
            mouth_bs["mouthPucker"] = 0.0
        else:          # 闭口状态 -> 撅嘴 (Woo)
            mouth_bs["mouthFunnel"] = 0.0
            mouth_bs["mouthPucker"] = smooth_remap(ratio_delta, 0.02, 0.22, 0.0, 1.0)
    else:
        mouth_bs["mouthFunnel"] = mouth_bs["mouthPucker"] = 0.0

    return mouth_bs

# ==================== 5. 头部刚体解算 ====================

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
        "x": rad_x * 180.0 / np.pi,
        "y": rad_y * 180.0 / np.pi,
        "z": rad_z * 180.0 / np.pi,
        "rad_x": rad_x,
        "rad_y": rad_y,
        "rad_z": rad_z,
        "width": np.linalg.norm(p2 - p1),
        "height": np.linalg.norm(p3mid - mid_point),
        "position": (mid_point + p3mid) * 0.5,
        "R": R
    }


# ==================== 6. 总控 ====================

def calc_all_arkit_coefficients(smooth_mesh_mock):
    global CALIBRATION_FRAME_COUNT, IS_CALIBRATED
    
    lm = smooth_mesh_mock
    
    if len(lm) == 0 or np.all(np.array(lm[6]) == 0):
        return {k: 0.0 for k in ["jawOpen", "eyeBlinkLeft", "mouthSmileLeft"]}

    computed_bs = {}
    head_pose = solve_head_rotation(lm)
    local_lm = transform_to_local_space(lm, head_pose)

    eyeInnerDistance = np.linalg.norm(np.array(local_lm[133]) - np.array(local_lm[362]))
    if eyeInnerDistance < 1e-6:
        eyeInnerDistance = 0.001

    eyeOuterDistance = np.linalg.norm(np.array(local_lm[130]) - np.array(local_lm[263]))
    if eyeOuterDistance < 1e-6:
        eyeOuterDistance = 0.001

    computed_bs.update(solve_eyes(local_lm, eyeInnerDistance, eyeOuterDistance))
    computed_bs.update(solve_brows(local_lm, eyeInnerDistance))
    computed_bs.update(solve_nose_and_cheeks(local_lm, eyeInnerDistance))
    computed_bs.update(solve_mouth_and_jaw(local_lm, eyeInnerDistance, eyeOuterDistance))

    # 边界截断
    for k, v in computed_bs.items():
        if not k.startswith("_"):
            computed_bs[k] = max(0.0, min(1.0, float(v)))

    computed_bs["_head_deg_x"] = head_pose["x"]
    computed_bs["_head_deg_y"] = head_pose["y"]
    computed_bs["_head_deg_z"] = head_pose["z"]

    # 校准状态机
    if not IS_CALIBRATED:
        if CALIBRATION_FRAME_COUNT == 0:
            print(f">>> 开始校准：前 {CALIBRATION_FRAMES_NEEDED} 帧请保持中性表情...")
        
        CALIBRATION_FRAME_COUNT += 1
        
        if CALIBRATION_FRAME_COUNT >= CALIBRATION_FRAMES_NEEDED:
            finalize_calibration()
            PREV_BS.clear()
            print(">>> 第一帧表情数据:", {k: round(v, 3) for k, v in computed_bs.items() 
                  if not k.startswith("_") and v > 0.05})
        elif CALIBRATION_FRAME_COUNT % 10 == 0:
            print(f"校准中 [{CALIBRATION_FRAME_COUNT}/{CALIBRATION_FRAMES_NEEDED}]...")
    
    # 应用平滑
    computed_bs = apply_smoothing(computed_bs)

    return computed_bs


def get_expression_vector(bs_dict):
    """获取标准ARKit表情向量"""
    standard_keys = [
        "eyeBlinkLeft", "eyeLookDownLeft", "eyeLookInLeft", "eyeLookOutLeft",
        "eyeLookUpLeft", "eyeSquintLeft", "eyeWideLeft", "eyeBlinkRight",
        "eyeLookDownRight", "eyeLookInRight", "eyeLookOutRight", "eyeLookUpRight",
        "eyeSquintRight", "eyeWideRight", "jawForward", "jawLeft",
        "jawRight", "jawOpen", "mouthClose", "mouthFunnel",
        "mouthPucker", "mouthLeft", "mouthRight", "mouthSmileLeft",
        "mouthSmileRight", "mouthFrownLeft", "mouthFrownRight", "mouthDimpleLeft",
        "mouthDimpleRight", "mouthStretchLeft", "mouthStretchRight", "mouthRollLower",
        "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthPressLeft",
        "mouthPressRight", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthUpperUpLeft",
        "mouthUpperUpRight", "browDownLeft", "browDownRight", "browInnerUp",
        "browOuterUpLeft", "browOuterUpRight", "cheekPuff", "cheekSquintLeft",
        "cheekSquintRight", "noseSneerLeft", "noseSneerRight", "tongueOut"
    ]
    return [bs_dict.get(k, 0.0) for k in standard_keys]


def is_calibrated():
    return IS_CALIBRATED