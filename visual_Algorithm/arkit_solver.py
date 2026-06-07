import numpy as np


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

PREV_BS = {}
SMOOTH_ALPHA = 0.3
CALIBRATION_FRAME_COUNT = 0
CALIBRATION_FRAMES_NEEDED = 30


def reset_calibration():
    global CALIBRATION_FRAME_COUNT
    for key in CALIBRATION:
        CALIBRATION[key] = None
    CALIBRATION_FRAME_COUNT = 0
    PREV_BS.clear()
    print(">>> 校准已重置！请保持中性平视表情 2 秒...")


def apply_smoothing(current_bs):
    global PREV_BS
    if not PREV_BS:
        PREV_BS = current_bs
        return current_bs

    smoothed = {}
    for k, v in current_bs.items():
        if k.startswith("_"):
            smoothed[k] = v
        else:
            smoothed[k] = SMOOTH_ALPHA * v + (1 - SMOOTH_ALPHA) * PREV_BS.get(k, v)

    PREV_BS = smoothed
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

    # --- 眼睛自动校准 ---
    if CALIBRATION_FRAME_COUNT < CALIBRATION_FRAMES_NEEDED:
        if CALIBRATION["l_eye_base"] is None:
            CALIBRATION["l_eye_base"] = l_open_ratio
            CALIBRATION["r_eye_base"] = r_open_ratio
            print(f"--- 眼睛基准校准成功! L:{l_open_ratio:.3f}, R:{r_open_ratio:.3f} ---")
        else:
            CALIBRATION["l_eye_base"] = 0.95 * CALIBRATION["l_eye_base"] + 0.05 * l_open_ratio
            CALIBRATION["r_eye_base"] = 0.95 * CALIBRATION["r_eye_base"] + 0.05 * r_open_ratio

    if CALIBRATION["l_eye_base"] is not None:
        l_blink_delta = CALIBRATION["l_eye_base"] - l_open_ratio
        r_blink_delta = CALIBRATION["r_eye_base"] - r_open_ratio

        eye_bs["eyeBlinkLeft"]  = smooth_remap(l_blink_delta, 0.04, 0.15, 0.0, 1.0)
        eye_bs["eyeBlinkRight"] = smooth_remap(r_blink_delta, 0.04, 0.15, 0.0, 1.0)

        l_wide_delta = l_open_ratio - CALIBRATION["l_eye_base"]
        r_wide_delta = r_open_ratio - CALIBRATION["r_eye_base"]
        eye_bs["eyeWideLeft"]  = smooth_remap(l_wide_delta, 0.03, 0.10, 0.0, 1.0)
        eye_bs["eyeWideRight"] = smooth_remap(r_wide_delta, 0.03, 0.10, 0.0, 1.0)

        eye_bs["eyeSquintLeft"]  = smooth_remap(l_blink_delta, 0.02, 0.08, 0.0, 1.0) * (1.0 - eye_bs["eyeBlinkLeft"])
        eye_bs["eyeSquintRight"] = smooth_remap(r_blink_delta, 0.02, 0.08, 0.0, 1.0) * (1.0 - eye_bs["eyeBlinkRight"])
    else:
        eye_bs["eyeBlinkLeft"] = eye_bs["eyeBlinkRight"] = 0.0
        eye_bs["eyeWideLeft"] = eye_bs["eyeWideRight"] = 0.0
        eye_bs["eyeSquintLeft"] = eye_bs["eyeSquintRight"] = 0.0

    # --- 瞳孔水平转动 ---
    p_pupil_l = np.array(lm[468])[:2]
    p_pupil_r = np.array(lm[473])[:2]

    l_pupil_ratio = np.dot(p_pupil_l - p_eye_l_inner[:2], p_eye_l_outer[:2] - p_eye_l_inner[:2]) / (l_eye_width**2)
    r_pupil_ratio = np.dot(p_pupil_r - p_eye_r_inner[:2], p_eye_r_outer[:2] - p_eye_r_inner[:2]) / (r_eye_width**2)

    eye_bs["eyeLookInLeft"]   = smooth_remap(l_pupil_ratio, 0.46, 0.35, 0.0, 1.0)
    eye_bs["eyeLookOutLeft"]  = smooth_remap(l_pupil_ratio, 0.54, 0.65, 0.0, 1.0)
    eye_bs["eyeLookInRight"]  = smooth_remap(r_pupil_ratio, 0.54, 0.65, 0.0, 1.0)
    eye_bs["eyeLookOutRight"] = smooth_remap(r_pupil_ratio, 0.46, 0.35, 0.0, 1.0)

    # --- 瞳孔上下转动 ---
    l_line_center_y = (p_eye_l_inner[1] + p_eye_l_outer[1]) * 0.5
    r_line_center_y = (p_eye_r_inner[1] + p_eye_r_outer[1]) * 0.5

    l_pupil_y_offset = (p_pupil_l[1] - l_line_center_y) / l_eye_width
    r_pupil_y_offset = (p_pupil_r[1] - r_line_center_y) / r_eye_width

    eye_bs["eyeLookUpLeft"]    = smooth_remap(l_pupil_y_offset, -0.02, -0.08, 0.0, 1.0)
    eye_bs["eyeLookDownLeft"]  = smooth_remap(l_pupil_y_offset, 0.02, 0.08, 0.0, 1.0)
    eye_bs["eyeLookUpRight"]   = smooth_remap(r_pupil_y_offset, -0.02, -0.08, 0.0, 1.0)
    eye_bs["eyeLookDownRight"] = smooth_remap(r_pupil_y_offset, 0.02, 0.08, 0.0, 1.0)

    return eye_bs


# ==================== 2. 眉毛 ====================

def solve_brows(lm, eye_dist_base):
    global CALIBRATION_FRAME_COUNT
    brow_bs = {}
    p_nose_bridge = np.array(lm[6])

    l_inner_raw = (p_nose_bridge[1] - np.array(lm[65])[1]) / eye_dist_base
    r_inner_raw = (p_nose_bridge[1] - np.array(lm[295])[1]) / eye_dist_base
    avg_inner = (l_inner_raw + r_inner_raw) * 0.5

    l_outer_raw = (p_nose_bridge[1] - np.array(lm[70])[1]) / eye_dist_base
    r_outer_raw = (p_nose_bridge[1] - np.array(lm[300])[1]) / eye_dist_base

    if CALIBRATION_FRAME_COUNT < CALIBRATION_FRAMES_NEEDED:
        if CALIBRATION["brow_inner_base"] is None:
            CALIBRATION["brow_inner_base"] = avg_inner
        else:
            CALIBRATION["brow_inner_base"] = 0.95 * CALIBRATION["brow_inner_base"] + 0.05 * avg_inner

    if CALIBRATION["brow_inner_base"] is not None:
        inner_delta = avg_inner - CALIBRATION["brow_inner_base"]
        brow_bs["browInnerUp"]     = smooth_remap(inner_delta, 0.02, 0.12, 0.0, 1.0)
        brow_bs["browDownLeft"]    = smooth_remap(-inner_delta, 0.02, 0.10, 0.0, 1.0)
        brow_bs["browDownRight"]   = smooth_remap(-inner_delta, 0.02, 0.10, 0.0, 1.0)

        l_outer_delta = l_outer_raw - CALIBRATION["brow_inner_base"]
        r_outer_delta = r_outer_raw - CALIBRATION["brow_inner_base"]
        brow_bs["browOuterUpLeft"]  = smooth_remap(l_outer_delta, 0.02, 0.12, 0.0, 1.0)
        brow_bs["browOuterUpRight"] = smooth_remap(r_outer_delta, 0.02, 0.12, 0.0, 1.0)
    else:
        for k in ["browInnerUp", "browDownLeft", "browDownRight", "browOuterUpLeft", "browOuterUpRight"]:
            brow_bs[k] = 0.0

    return brow_bs


# ==================== 3. 鼻子与脸颊 ====================

def solve_nose_and_cheeks(lm, eye_dist_base):
    global CALIBRATION_FRAME_COUNT
    nc_bs = {}
    p_eye_l_inner = np.array(lm[133])
    p_eye_r_inner = np.array(lm[362])

    l_sneer_raw = abs(p_eye_l_inner[1] - lm[129][1]) / eye_dist_base
    r_sneer_raw = abs(p_eye_r_inner[1] - lm[358][1]) / eye_dist_base
    avg_sneer = (l_sneer_raw + r_sneer_raw) * 0.5

    l_cheek_raw = abs(lm[50][1] - lm[145][1]) / eye_dist_base
    r_cheek_raw = abs(lm[280][1] - lm[374][1]) / eye_dist_base
    avg_cheek = (l_cheek_raw + r_cheek_raw) * 0.5

    if CALIBRATION_FRAME_COUNT < CALIBRATION_FRAMES_NEEDED:
        if CALIBRATION["sneer_base"] is None:
            CALIBRATION["sneer_base"] = avg_sneer
            CALIBRATION["cheek_base"] = avg_cheek
        else:
            CALIBRATION["sneer_base"] = 0.95 * CALIBRATION["sneer_base"] + 0.05 * avg_sneer
            CALIBRATION["cheek_base"] = 0.95 * CALIBRATION["cheek_base"] + 0.05 * avg_cheek

    if CALIBRATION["sneer_base"] is not None:
        sneer_delta = avg_sneer - CALIBRATION["sneer_base"]
        nc_bs["noseSneerLeft"]  = smooth_remap(sneer_delta, 0.03, 0.12, 0.0, 1.0)
        nc_bs["noseSneerRight"] = smooth_remap(sneer_delta, 0.03, 0.12, 0.0, 1.0)

        cheek_delta = CALIBRATION["cheek_base"] - avg_cheek
        nc_bs["cheekSquintLeft"]  = smooth_remap(cheek_delta, 0.02, 0.10, 0.0, 1.0)
        nc_bs["cheekSquintRight"] = smooth_remap(cheek_delta, 0.02, 0.10, 0.0, 1.0)
    else:
        for k in ["noseSneerLeft", "noseSneerRight", "cheekSquintLeft", "cheekSquintRight"]:
            nc_bs[k] = 0.0

    cheek_width = np.linalg.norm(np.array(lm[50][:2]) - np.array(lm[280][:2]))
    puff_ratio = cheek_width / eye_dist_base
    nc_bs["cheekPuff"] = smooth_remap(puff_ratio, 2.72, 2.95, 0.0, 1.0)

    return nc_bs


# ==================== 4. 嘴部与下巴 ====================

def solve_mouth_and_jaw(lm, eye_dist_base, eye_outer_base):
    global CALIBRATION_FRAME_COUNT
    mouth_bs = {}

    mouth_height = np.linalg.norm(np.array(lm[13][:2]) - np.array(lm[14][:2]))
    mouth_width_raw = np.linalg.norm(np.array(lm[61][:2]) - np.array(lm[291][:2]))
    mar = mouth_height / (mouth_width_raw + 1e-6)

    mouth_bs["jawOpen"]    = smooth_remap(mar, 0.08, 0.65, 0.0, 1.0)
    mouth_bs["mouthClose"] = smooth_remap(mar, 0.12, 0.01, 0.0, 1.0) if mouth_bs["jawOpen"] > 0.05 else 0.0

    mid_face_x = (lm[133][0] + lm[362][0]) * 0.5
    jaw_x_offset = (lm[152][0] - mid_face_x) / eye_dist_base

    if CALIBRATION_FRAME_COUNT < CALIBRATION_FRAMES_NEEDED:
        if CALIBRATION["jaw_x_base"] is None:
            CALIBRATION["jaw_x_base"] = jaw_x_offset
        else:
            CALIBRATION["jaw_x_base"] = 0.95 * CALIBRATION["jaw_x_base"] + 0.05 * jaw_x_offset

    if CALIBRATION["jaw_x_base"] is not None:
        jaw_delta = jaw_x_offset - CALIBRATION["jaw_x_base"]
        mouth_bs["jawLeft"]  = smooth_remap(jaw_delta, 0.04, 0.15, 0.0, 1.0)
        mouth_bs["jawRight"] = smooth_remap(-jaw_delta, 0.04, 0.15, 0.0, 1.0)
    else:
        mouth_bs["jawLeft"] = mouth_bs["jawRight"] = 0.0

    p_mouth_l = np.array(lm[61])
    p_mouth_r = np.array(lm[291])
    p_mouth_center = (np.array(lm[0]) + np.array(lm[17])) * 0.5

    l_corner_y = (p_mouth_center[1] - p_mouth_l[1]) / eye_dist_base
    r_corner_y = (p_mouth_center[1] - p_mouth_r[1]) / eye_dist_base

    if CALIBRATION_FRAME_COUNT < CALIBRATION_FRAMES_NEEDED:
        if CALIBRATION["mouth_smile_l_base"] is None:
            CALIBRATION["mouth_smile_l_base"] = l_corner_y
            CALIBRATION["mouth_smile_r_base"] = r_corner_y
        else:
            CALIBRATION["mouth_smile_l_base"] = 0.95 * CALIBRATION["mouth_smile_l_base"] + 0.05 * l_corner_y
            CALIBRATION["mouth_smile_r_base"] = 0.95 * CALIBRATION["mouth_smile_r_base"] + 0.05 * r_corner_y

    if CALIBRATION["mouth_smile_l_base"] is not None:
        l_smile_delta = l_corner_y - CALIBRATION["mouth_smile_l_base"]
        r_smile_delta = r_corner_y - CALIBRATION["mouth_smile_r_base"]
        mouth_bs["mouthSmileLeft"]  = smooth_remap(l_smile_delta, 0.005, 0.15, 0.0, 1.0)
        mouth_bs["mouthSmileRight"] = smooth_remap(r_smile_delta, 0.005, 0.15, 0.0, 1.0)
        mouth_bs["mouthFrownLeft"]  = smooth_remap(-l_smile_delta, 0.005, 0.15, 0.0, 1.0)
        mouth_bs["mouthFrownRight"] = smooth_remap(-r_smile_delta, 0.005, 0.15, 0.0, 1.0)
    else:
        for k in ["mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft", "mouthFrownRight"]:
            mouth_bs[k] = 0.0

    mouth_ratio = np.linalg.norm(p_mouth_l[:2] - p_mouth_r[:2]) / eye_dist_base

    if CALIBRATION_FRAME_COUNT < CALIBRATION_FRAMES_NEEDED:
        if CALIBRATION["mouth_ratio_base"] is None:
            CALIBRATION["mouth_ratio_base"] = mouth_ratio
        else:
            CALIBRATION["mouth_ratio_base"] = 0.95 * CALIBRATION["mouth_ratio_base"] + 0.05 * mouth_ratio

    if CALIBRATION["mouth_ratio_base"] is not None:
        ratio_delta = CALIBRATION["mouth_ratio_base"] - mouth_ratio
        funnel_raw = smooth_remap(ratio_delta, 0.02, 0.25, 0.0, 1.0)
        mouth_bs["mouthFunnel"] = funnel_raw if mouth_bs["jawOpen"] > 0.25 else 0.0

        pucker_raw = smooth_remap(ratio_delta, 0.02, 0.30, 0.0, 1.0)
        if mouth_bs["mouthFunnel"] > 0.1:
            mouth_bs["mouthPucker"] = 0.0
        else:
            mouth_bs["mouthPucker"] = min(pucker_raw, 0.8)
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
        "x": rad_x * 180.0 / np.pi, "y": rad_y * 180.0 / np.pi, "z": rad_z * 180.0 / np.pi,
        "rad_x": rad_x, "rad_y": rad_y, "rad_z": rad_z,
        "width": np.linalg.norm(p2 - p1),
        "height": np.linalg.norm(p3mid - mid_point),
        "position": (mid_point + p3mid) * 0.5,
        "R": R
    }


# ==================== 6. 总控数据分发 ====================

def calc_all_arkit_coefficients(smooth_mesh_mock):
    global CALIBRATION_FRAME_COUNT
    lm = smooth_mesh_mock
    if len(lm) == 0 or np.all(np.array(lm[6]) == 0):
        return {k: 0.0 for k in ["jawOpen", "eyeBlinkLeft", "mouthSmileLeft"]}

    computed_bs = {}

    head_pose = solve_head_rotation(lm)
    local_lm = transform_to_local_space(lm, head_pose)

    eyeInnerDistance = np.linalg.norm(np.array(local_lm[133]) - np.array(local_lm[362]))
    if eyeInnerDistance == 0: eyeInnerDistance = 0.001

    eyeOuterDistance = np.linalg.norm(np.array(local_lm[130]) - np.array(local_lm[263]))
    if eyeOuterDistance == 0: eyeOuterDistance = 0.001

    computed_bs.update(solve_eyes(local_lm, eyeInnerDistance, eyeOuterDistance))
    computed_bs.update(solve_brows(local_lm, eyeInnerDistance))
    computed_bs.update(solve_nose_and_cheeks(local_lm, eyeInnerDistance))
    computed_bs.update(solve_mouth_and_jaw(local_lm, eyeInnerDistance, eyeOuterDistance))

    # 后处理边界截断
    for k, v in computed_bs.items():
        if not k.startswith("_"):
            computed_bs[k] = max(0.0, min(1.0, float(v)))

    # 头部数据（不平滑）
    computed_bs["_head_deg_x"] = head_pose["x"]
    computed_bs["_head_deg_y"] = head_pose["y"]
    computed_bs["_head_deg_z"] = head_pose["z"]

    # ==================== 校准逻辑 ====================
    if CALIBRATION_FRAME_COUNT == 0:
        print(f">>> 开始校准：前 {CALIBRATION_FRAMES_NEEDED} 帧请保持中性表情...")

    if CALIBRATION_FRAME_COUNT < CALIBRATION_FRAMES_NEEDED:
        CALIBRATION_FRAME_COUNT += 1
        # 强制归零（但保留瞳孔方向，让校准期间也能看到眼睛转动）
        for k in list(computed_bs.keys()):
            if not k.startswith("_") and not k.startswith("eyeLook"):
                computed_bs[k] = 0.0
    elif CALIBRATION_FRAME_COUNT == CALIBRATION_FRAMES_NEEDED:
        CALIBRATION_FRAME_COUNT += 1
        # 重置 EMA 缓存，避免被校准期的全 0 拖慢响应
        PREV_BS.clear()
        print(">>> ----------------------------------------------------")
        print(">>> 【校准完成】动态面部基准已成功锁定！")
        print(f"    brow_inner  = {CALIBRATION['brow_inner_base']:.3f}")
        print(f"    l_eye_base  = {CALIBRATION['l_eye_base']:.3f} | r_eye_base  = {CALIBRATION['r_eye_base']:.3f}")
        print(f"    sneer_base  = {CALIBRATION['sneer_base']:.3f} | cheek_base  = {CALIBRATION['cheek_base']:.3f}")
        print(f"    jaw_x_base  = {CALIBRATION['jaw_x_base']:.3f}")
        print(f"    smile_l     = {CALIBRATION['mouth_smile_l_base']:.3f} | smile_r     = {CALIBRATION['mouth_smile_r_base']:.3f}")
        print(f"    mouth_ratio = {CALIBRATION['mouth_ratio_base']:.3f}")
        print(">>> ----------------------------------------------------")
        print(">>> 驱动正式接入！现在可以正常做表情了。")
        # 打印第一帧非零数据供验证
        print(computed_bs)

    # 应用平滑（瞳孔方向不平滑，保持直接响应）
    for k in list(computed_bs.keys()):
        if k.startswith("eyeLook"):
            continue
        if not k.startswith("_"):
            computed_bs[k] = apply_smoothing_single(k, computed_bs[k])

    # 打印调试（前 20 帧 + 校准完成后额外 5 帧）
    if not hasattr(calc_all_arkit_coefficients, "_print_count"):
        calc_all_arkit_coefficients._print_count = 0
    if not hasattr(calc_all_arkit_coefficients, "_post_calib_count"):
        calc_all_arkit_coefficients._post_calib_count = 0

    if CALIBRATION_FRAME_COUNT <= CALIBRATION_FRAMES_NEEDED:
        if calc_all_arkit_coefficients._print_count < 20:
            print(computed_bs)
            calc_all_arkit_coefficients._print_count += 1
    else:
        if calc_all_arkit_coefficients._post_calib_count < 5:
            print(computed_bs)
            calc_all_arkit_coefficients._post_calib_count += 1

    return computed_bs


# 辅助：单键 EMA，替代 apply_smoothing，方便跳过瞳孔
def apply_smoothing_single(key, value):
    global PREV_BS
    prev = PREV_BS.get(key, value)
    smoothed = SMOOTH_ALPHA * value + (1 - SMOOTH_ALPHA) * prev
    PREV_BS[key] = smoothed
    return smoothed