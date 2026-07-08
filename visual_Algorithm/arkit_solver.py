import numpy as np
from collections import defaultdict

# ======================= 全局配置 =======================
# 动态基准 — 按测量类型分别用不同衰减速度
DYNAMIC_BASELINE = {
    # 眼部：变化慢，稳定
    "l_eye_open":   0.28,
    "r_eye_open":   0.28,
    # 眉毛：用眼角做参考（brow_y - eye_y）/ eye_dist
    # 典型值 0.25~0.45，取中值，前 60 帧快速收敛
    "brow_inner":   0.35,
    "brow_outer_l": 0.30,
    "brow_outer_r": 0.30,
    "brow_down_l":  -0.35,    # brow mid 相对 eye center 的Y偏移，负值=眉在眼上方
    "brow_down_r":  -0.35,
    # 嘴部
    "mouth_width":     0.92,
    "mouth_open":      0.03,
    "mouth_ratio":     0.26,
    "mouth_center_x":  0.0,
    "lip_z_upper":     0.0,
    "lip_z_lower":     0.0,
    "corner_y_l":      0.0,
    "corner_y_r":      0.0,
    # 脸颊/鼻子
    "cheek_squint_l": 0.16,
    "cheek_squint_r": 0.16,
    "l_squint_raw":  0.0,
    "r_squint_raw":  0.0,
    "cheek_puff_z":  0.0,
    "sneer_dist_l":  0.05,
    "sneer_dist_r":  0.05,
    # 下巴
    "jaw_x":         0.0,
    "jaw_forward":   0.0,
    "jaw_open_ratio": 0.06,
}
# 快速收敛计数器：前 60 帧用更快的学习率
_CALIB_FRAME = 0
_CALIB_FAST_FRAMES = 60
_CALIB_FAST_DECAY = 0.85   # 快速阶段 EMA 衰减

# 各测量的 EMA 衰减 (越接近 1 越慢)
BASELINE_DECAY_MAP = {
    "l_eye_open":      0.995,
    "r_eye_open":      0.995,
    "brow_inner":      0.997,
    "brow_outer_l":    0.997,
    "brow_outer_r":    0.997,
    "brow_down_l":     0.997,
    "brow_down_r":     0.997,
    "mouth_width":     0.998,
    "mouth_open":      0.998,
    "mouth_ratio":     0.997,
    "mouth_center_x":  0.997,
    "lip_z_upper":     0.997,
    "lip_z_lower":     0.997,
    "corner_y_l":      0.997,
    "corner_y_r":      0.997,
    "cheek_squint_l":  0.997,
    "cheek_squint_r":  0.997,
    "l_squint_raw":    0.997,
    "r_squint_raw":    0.997,
    "cheek_puff_z":    0.997,
    "sneer_dist_l":    0.997,
    "sneer_dist_r":    0.997,
    "jaw_x":           0.996,
    "jaw_forward":     0.997,
    "jaw_open_ratio":  0.998,
}
DEFAULT_DECAY = 0.997

# Neutral 条件：这些 BS 值必须小于阈值才更新基准
NEUTRAL_CONDITIONS = {
    "jawOpen":          0.06,
    "eyeBlinkLeft":     0.15,
    "eyeBlinkRight":    0.15,
    "mouthSmileLeft":   0.08,
    "mouthSmileRight":  0.08,
    "mouthFunnel":      0.08,
    "mouthPucker":      0.08,
    "browInnerUp":      0.10,
    "cheekPuff":        0.08,
}

# 头部姿态变化阈值：超过此值不更新基准（防止转动时漂移）
HEAD_DELTA_THRESHOLD = 2.5  # 度

# 独立平滑系数 — 不同表情用不同响应速度
SMOOTH_ALPHAS = {
    "default":              0.40,
    # 眨眼要快
    "eyeBlinkLeft":         0.80,
    "eyeBlinkRight":        0.80,
    "eyeSquintLeft":        0.65,
    "eyeSquintRight":       0.65,
    "eyeWideLeft":          0.50,
    "eyeWideRight":         0.50,
    # 眼球要快
    "eyeLookUpLeft":        0.55,
    "eyeLookDownLeft":      0.55,
    "eyeLookInLeft":        0.55,
    "eyeLookOutLeft":       0.55,
    "eyeLookUpRight":       0.55,
    "eyeLookDownRight":     0.55,
    "eyeLookInRight":       0.55,
    "eyeLookOutRight":      0.55,
    # 张嘴要快
    "jawOpen":              0.58,
    # 微笑/撇嘴 稳定
    "mouthSmileLeft":       0.35,
    "mouthSmileRight":      0.35,
    "mouthFrownLeft":       0.35,
    "mouthFrownRight":      0.35,
    # 嘴巴形状
    "mouthFunnel":          0.55,
    "mouthPucker":          0.55,
    "mouthStretchLeft":     0.45,
    "mouthStretchRight":    0.45,
    "mouthDimpleLeft":      0.45,
    "mouthDimpleRight":     0.45,
    # 上/下唇
    "mouthRollLower":       0.45,
    "mouthRollUpper":       0.45,
    "mouthPressLeft":       0.50,
    "mouthPressRight":      0.50,
    "mouthLowerDownLeft":   0.45,
    "mouthLowerDownRight":  0.45,
    "mouthUpperUpLeft":     0.45,
    "mouthUpperUpRight":    0.45,
    # 嘴左右
    "mouthLeft":            0.50,
    "mouthRight":           0.50,
    # 下巴
    "jawLeft":              0.42,
    "jawRight":             0.42,
    "jawForward":           0.42,
    # 眉毛稳定
    "browInnerUp":          0.30,
    "browOuterUpLeft":      0.30,
    "browOuterUpRight":     0.30,
    "browDownLeft":         0.30,
    "browDownRight":        0.30,
    # 脸颊/鼻子
    "cheekPuff":            0.35,
    "cheekSquintLeft":      0.35,
    "cheekSquintRight":     0.35,
    "noseSneerLeft":        0.40,
    "noseSneerRight":       0.40,
    "mouthClose":           0.58,
}

PREV_BS = {}
TEMPLATE_LANDMARKS = None
PREV_HEAD_ANGLES = None   # 上一帧头部角度，用于检测运动

# ======================= 辅助函数 =======================
def to_numpy(points):
    if isinstance(points, np.ndarray):
        return points
    return np.array(points, dtype=np.float32)

def smoothstep01(x):
    """Hermite 平滑插值，在 0 和 1 附近变化平缓"""
    return x * x * (3 - 2 * x)

def remap_clip(x, in_min, in_max, out_min=0.0, out_max=1.0):
    """将 x 从 [in_min, in_max] 映射到 [out_min, out_max]，smoothstep 曲线"""
    if in_max <= in_min:
        return out_min
    t = np.clip((x - in_min) / (in_max - in_min + 1e-8), 0.0, 1.0)
    return out_min + smoothstep01(t) * (out_max - out_min)

def remap_clip_linear(x, in_min, in_max, out_min=0.0, out_max=1.0):
    """线性映射版本，用于需要更灵敏响应的场景"""
    if in_max <= in_min:
        return out_min
    t = np.clip((x - in_min) / (in_max - in_min + 1e-8), 0.0, 1.0)
    return out_min + t * (out_max - out_min)

def update_baseline(key, value, current_bs=None):
    """仅在 neutral 条件下用 EMA 更新基准（前 N 帧快速学习）"""
    global DYNAMIC_BASELINE, PREV_HEAD_ANGLES, _CALIB_FRAME

    # 快速收敛阶段：始终更新，忽略 neutral 检查
    is_fast_phase = _CALIB_FRAME < _CALIB_FAST_FRAMES

    if current_bs is not None and not is_fast_phase:
        # 检查表情中性条件
        is_neutral = True
        for cond_key, threshold in NEUTRAL_CONDITIONS.items():
            if current_bs.get(cond_key, 0.0) > threshold:
                is_neutral = False
                break
        # 检查头部稳定性
        if PREV_HEAD_ANGLES is not None:
            cur_angles = current_bs.get("_head_deg", None)
            if cur_angles is not None:
                d = np.linalg.norm(np.array(cur_angles) - np.array(PREV_HEAD_ANGLES))
                if d > HEAD_DELTA_THRESHOLD:
                    is_neutral = False

        if not is_neutral:
            return

    old = DYNAMIC_BASELINE.get(key, value)
    decay = BASELINE_DECAY_MAP.get(key, DEFAULT_DECAY)

    # 快速阶段用更快的衰减（但渐变回到正常值）
    if is_fast_phase and _CALIB_FAST_FRAMES > 0:
        progress = min(1.0, _CALIB_FRAME / _CALIB_FAST_FRAMES)
        effective_decay = _CALIB_FAST_DECAY + (decay - _CALIB_FAST_DECAY) * progress
    else:
        effective_decay = decay

    if key == "l_eye_open" or key == "r_eye_open":
        new = old * effective_decay + value * (1.0 - effective_decay)
    elif key in ("jaw_x", "jaw_forward", "cheek_puff_z", "sneer_dist_l", "sneer_dist_r",
                 "corner_y_l", "corner_y_r", "lip_z_upper", "lip_z_lower",
                 "mouth_center_x"):
        new = old * effective_decay + value * (1.0 - effective_decay)
        new = new if abs(new) <= abs(value) else value
    elif key in ("mouth_width", "mouth_open", "mouth_ratio",
                 "cheek_squint_l", "cheek_squint_r"):
        new = old * effective_decay + value * (1.0 - effective_decay)
        new = min(new, old)
    else:
        # brow_inner, brow_outer_l/r, brow_down_l/r, etc.
        new = old * effective_decay + value * (1.0 - effective_decay)

    DYNAMIC_BASELINE[key] = new

def get_baseline(key):
    return DYNAMIC_BASELINE.get(key, 0.0)

def apply_smoothing(current_bs):
    global PREV_BS
    if not PREV_BS:
        PREV_BS = {k: v for k, v in current_bs.items() if not k.startswith("_")}
        return current_bs

    smoothed = {}
    for k, v in current_bs.items():
        if k.startswith("_"):
            smoothed[k] = v
        else:
            alpha = SMOOTH_ALPHAS.get(k, SMOOTH_ALPHAS["default"])
            prev = PREV_BS.get(k, v)
            smoothed[k] = alpha * v + (1.0 - alpha) * prev

    for k, v in smoothed.items():
        if not k.startswith("_"):
            PREV_BS[k] = v

    return smoothed

def reset_calibration():
    global PREV_BS, TEMPLATE_LANDMARKS, PREV_HEAD_ANGLES, DYNAMIC_BASELINE, _CALIB_FRAME
    _CALIB_FRAME = 0
    for k in DYNAMIC_BASELINE:
        if "eye_open" in k:
            DYNAMIC_BASELINE[k] = 0.28
        elif k == "brow_inner":
            DYNAMIC_BASELINE[k] = 0.35
        elif k.startswith("brow_outer"):
            DYNAMIC_BASELINE[k] = 0.30
        elif k.startswith("brow_down"):
            DYNAMIC_BASELINE[k] = -0.35
        elif k == "mouth_width":
            DYNAMIC_BASELINE[k] = 0.92
        elif k == "mouth_ratio":
            DYNAMIC_BASELINE[k] = 0.26
        elif k == "mouth_open":
            DYNAMIC_BASELINE[k] = 0.03
        elif "cheek_squint" in k:
            DYNAMIC_BASELINE[k] = 0.16
        elif k == "jaw_open_ratio":
            DYNAMIC_BASELINE[k] = 0.06
        elif k in ("mouth_center_x", "lip_z_upper", "lip_z_lower",
                   "corner_y_l", "corner_y_r"):
            DYNAMIC_BASELINE[k] = 0.0
        else:
            DYNAMIC_BASELINE[k] = 0.0
    PREV_BS.clear()
    TEMPLATE_LANDMARKS = None
    PREV_HEAD_ANGLES = None
    print(">>> 动态基准已重置")

# ======================= 头部旋转 (Procrustes / Umeyama) =======================
def procrustes_rotation(source, target):
    """SVD 求解最优旋转矩阵"""
    source = to_numpy(source)
    target = to_numpy(target)
    source_c = source - np.mean(source, axis=0)
    target_c = target - np.mean(target, axis=0)
    H = source_c.T @ target_c
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return R

def solve_head_rotation(lm):
    """Procrustes 对齐得到头部旋转和位置"""
    global TEMPLATE_LANDMARKS
    # 稳定关键点索引（避开嘴唇和眼睑等变形大的区域）
    template_indices = [1, 4, 5, 6, 8, 9, 10, 33, 133, 263, 362, 152, 168, 193, 411, 427]
    target_pts = np.array([to_numpy(lm[i]) for i in template_indices])

    if TEMPLATE_LANDMARKS is None:
        TEMPLATE_LANDMARKS = target_pts.copy()

    R = procrustes_rotation(TEMPLATE_LANDMARKS, target_pts)

    # 欧拉角 XYZ
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    rad_x = np.arctan2(-R[2, 0], sy)
    rad_y = np.arctan2(R[1, 0], R[0, 0])
    rad_z = np.arctan2(R[2, 1], R[2, 2])

    lm1, lm152, lm168 = to_numpy(lm[1]), to_numpy(lm[152]), to_numpy(lm[168])
    face_center = np.mean([lm1, lm152, lm168], axis=0)
    head_width = np.linalg.norm(to_numpy(lm[21]) - to_numpy(lm[251]))

    return {
        "x": rad_x * 180 / np.pi,
        "y": rad_y * 180 / np.pi,
        "z": rad_z * 180 / np.pi,
        "rad_x": rad_x, "rad_y": rad_y, "rad_z": rad_z,
        "R": R,
        "position": face_center,
        "width": head_width,
    }

def transform_to_local_space(lm, head_pose):
    """将关键点变换到头部局部坐标系"""
    origin = head_pose["position"]
    R = head_pose["R"]
    local = {}
    for idx, pt in lm.items():
        local[idx] = R.T @ (to_numpy(pt) - origin)
    return local

# ======================= 眼部 (Blink + EyeLook) =======================
def solve_eyes(local_lm, head_pose, eye_dist_base, current_bs):
    bs = {}

    # ---- 左右眼关键点 ----
    l_top_pts  = [159, 158, 160]
    l_bot_pts  = [145, 153, 144]
    l_inner, l_outer = 133, 130

    r_top_pts  = [386, 385, 387]
    r_bot_pts  = [374, 373, 380]
    r_inner, r_outer = 362, 263

    def eye_open_ratio(top_idx, bot_idx, inner, outer):
        vert = np.mean([np.linalg.norm(local_lm[t] - local_lm[b])
                       for t, b in zip(top_idx, bot_idx)])
        horiz = np.linalg.norm(local_lm[inner][:2] - local_lm[outer][:2])
        return vert / (horiz + 1e-6)

    l_open = eye_open_ratio(l_top_pts, l_bot_pts, l_inner, l_outer)
    r_open = eye_open_ratio(r_top_pts, r_bot_pts, r_inner, r_outer)

    # ---- 动态基线 ----
    update_baseline("l_eye_open", l_open, current_bs)
    update_baseline("r_eye_open", r_open, current_bs)
    base_l = get_baseline("l_eye_open")
    base_r = get_baseline("r_eye_open")

    # ---- 眨眼：阈值按 baseline 百分比 ----
    l_blink = max(0.0, base_l - l_open)
    r_blink = max(0.0, base_r - r_open)
    bs["eyeBlinkLeft"]  = remap_clip(l_blink, 0, max(0.015, base_l * 0.50), 0, 1)
    bs["eyeBlinkRight"] = remap_clip(r_blink, 0, max(0.015, base_r * 0.50), 0, 1)

    # ---- 瞪眼 ----
    bs["eyeWideLeft"]  = remap_clip(l_open - base_l, 0, max(0.02, base_l * 0.30), 0, 1)
    bs["eyeWideRight"] = remap_clip(r_open - base_r, 0, max(0.02, base_r * 0.30), 0, 1)

    # ---- 挤眼：从下眼睑上升量推导（非 blink 倍率） ----
    def lower_lid_raise(lower_pt, inner, outer):
        """下眼睑点 Y 相对眼角平均 Y 的上升量 / eye_dist"""
        corner_y_avg = (local_lm[inner][1] + local_lm[outer][1]) / 2
        return (local_lm[lower_pt][1] - corner_y_avg) / eye_dist_base

    l_squint_raw = lower_lid_raise(145, l_inner, l_outer)
    r_squint_raw = lower_lid_raise(374, r_inner, r_outer)

    update_baseline("l_squint_raw", l_squint_raw, current_bs)
    update_baseline("r_squint_raw", r_squint_raw, current_bs)
    base_lsq = get_baseline("l_squint_raw")
    base_rsq = get_baseline("r_squint_raw")
    bs["eyeSquintLeft"]  = remap_clip(l_squint_raw - base_lsq, 0,
                                       max(0.005, abs(base_lsq) * 0.40 + 0.003), 0, 1)
    bs["eyeSquintRight"] = remap_clip(r_squint_raw - base_rsq, 0,
                                       max(0.005, abs(base_rsq) * 0.40 + 0.003), 0, 1)

    # ---- 虹膜/瞳孔中心 ----
    pupil_l = local_lm.get(468, (local_lm[l_inner] + local_lm[l_outer]) * 0.5)
    pupil_r = local_lm.get(473, (local_lm[r_inner] + local_lm[r_outer]) * 0.5)

    # ---- 眼窝平面 UV 坐标（4 点平面投影，比简单线段更稳） ----
    def eye_socket_uv(inner, outer, top_pts, bot_pts, pupil):
        """将瞳孔投影到由眼角 + 眼睑中点定义的眼窝平面上，返回 UV"""
        inner_v = local_lm[inner]
        outer_v = local_lm[outer]
        top_mid = np.mean([local_lm[t] for t in top_pts], axis=0)
        bot_mid = np.mean([local_lm[t] for t in bot_pts], axis=0)

        # 水平方向: inner → outer
        h_vec = outer_v - inner_v
        h_len = np.linalg.norm(h_vec)
        if h_len < 1e-6:
            return 0.5, 0.0

        # 垂直方向: bottom → top（相对水平方向正交化）
        v_raw = top_mid - bot_mid
        # 去除水平分量，保证正交
        v_vec = v_raw - np.dot(v_raw, h_vec) / (h_len * h_len) * h_vec
        v_len = np.linalg.norm(v_vec)
        if v_len < 1e-6:
            u = np.dot(pupil - inner_v, h_vec) / (h_len * h_len)
            return np.clip(u, 0, 1), 0.0

        # 眼窝中心
        socket_center = (inner_v + outer_v + top_mid + bot_mid) / 4.0
        # UV：h 范围约 ±0.5h_len，v 范围约 ±0.5v_len
        dp = pupil - socket_center
        u = np.dot(dp, h_vec) / (h_len * h_len) + 0.5
        v = np.dot(dp, v_vec) / (v_len * v_len)
        return np.clip(u, 0.1, 0.9), np.clip(v, -0.4, 0.4)

    l_u, l_v = eye_socket_uv(l_inner, l_outer, l_top_pts, l_bot_pts, pupil_l)
    r_u, r_v = eye_socket_uv(r_inner, r_outer, r_top_pts, r_bot_pts, pupil_r)

    # ---- 水平注视（基于 UV 偏离中心 0.5） ----
    # 动态阈值：中性时瞳孔 UV 约 0.48~0.52，死区 0.08，最大偏移约 0.28
    dead_zone_h = 0.08
    max_offset_h = 0.28
    bs["eyeLookInLeft"]   = remap_clip(0.5 - l_u, dead_zone_h, max_offset_h, 0, 1) if l_u < 0.42 else 0.0
    bs["eyeLookOutLeft"]  = remap_clip(l_u - 0.5, dead_zone_h, max_offset_h, 0, 1) if l_u > 0.58 else 0.0
    bs["eyeLookInRight"]  = remap_clip(0.5 - r_u, dead_zone_h, max_offset_h, 0, 1) if r_u < 0.42 else 0.0
    bs["eyeLookOutRight"] = remap_clip(r_u - 0.5, dead_zone_h, max_offset_h, 0, 1) if r_u > 0.58 else 0.0

    # ---- 垂直注视 ----
    dead_zone_v = 0.04
    max_offset_v = 0.18
    bs["eyeLookUpLeft"]    = remap_clip(-l_v, dead_zone_v, max_offset_v, 0, 1)
    bs["eyeLookDownLeft"]  = remap_clip(l_v,  dead_zone_v, max_offset_v, 0, 1)
    bs["eyeLookUpRight"]   = remap_clip(-r_v, dead_zone_v, max_offset_v, 0, 1)
    bs["eyeLookDownRight"] = remap_clip(r_v,  dead_zone_v, max_offset_v, 0, 1)

    return bs

# ======================= 眉毛 =======================
def solve_brows(local_lm, eye_dist_base, current_bs):
    bs = {}

    # 眼角参考点（比鼻子稳定很多）
    l_eye_inner = local_lm[133]
    r_eye_inner = local_lm[362]
    l_eye_outer = local_lm[130]
    r_eye_outer = local_lm[263]

    # 眉毛关键点
    l_brow_inner = local_lm[65]
    r_brow_inner = local_lm[295]
    l_brow_outer = local_lm[70]
    r_brow_outer = local_lm[300]

    # 眉头中点 + 眼中心（用于 browDown）
    l_brow_mid  = local_lm[105]
    r_brow_mid  = local_lm[334]
    l_eye_center = (l_eye_inner + l_eye_outer) / 2
    r_eye_center = (r_eye_inner + r_eye_outer) / 2

    # ---- browInnerUp：眉头相对内眼角的高度 ----
    inner_h = ((l_brow_inner[1] - l_eye_inner[1]) + (r_brow_inner[1] - r_eye_inner[1])) \
              / (2 * eye_dist_base)
    update_baseline("brow_inner", inner_h, current_bs)
    base_inner = get_baseline("brow_inner")
    range_inner = max(0.04, abs(base_inner) * 0.55 + 0.015)
    bs["browInnerUp"] = remap_clip(inner_h - base_inner, 0, range_inner, 0, 1)

    # ---- browDownLeft/Right：眉腰相对眼中心的降低量 ----
    l_down = (l_eye_center[1] - l_brow_mid[1]) / eye_dist_base
    r_down = (r_eye_center[1] - r_brow_mid[1]) / eye_dist_base
    # l_down > 0 = 眉毛降低到眼睛中心以下
    update_baseline("brow_down_l", l_down, current_bs)
    update_baseline("brow_down_r", r_down, current_bs)
    base_bdl = get_baseline("brow_down_l")
    base_bdr = get_baseline("brow_down_r")
    range_bdl = max(0.02, abs(base_bdl) * 0.45 + 0.01)
    range_bdr = max(0.02, abs(base_bdr) * 0.45 + 0.01)
    bs["browDownLeft"]  = remap_clip(l_down - base_bdl, 0, range_bdl, 0, 1)
    bs["browDownRight"] = remap_clip(r_down - base_bdr, 0, range_bdr, 0, 1)

    # ---- browOuterUp：外眉相对外眼角的高度 ----
    outer_h_l = (l_brow_outer[1] - l_eye_outer[1]) / eye_dist_base
    outer_h_r = (r_brow_outer[1] - r_eye_outer[1]) / eye_dist_base
    update_baseline("brow_outer_l", outer_h_l, current_bs)
    update_baseline("brow_outer_r", outer_h_r, current_bs)
    base_ol = get_baseline("brow_outer_l")
    base_oi = get_baseline("brow_outer_r")
    range_ol = max(0.04, abs(base_ol) * 0.55 + 0.015)
    range_oi = max(0.04, abs(base_oi) * 0.55 + 0.015)
    bs["browOuterUpLeft"]  = remap_clip(outer_h_l - base_ol, 0, range_ol, 0, 1)
    bs["browOuterUpRight"] = remap_clip(outer_h_r - base_oi, 0, range_oi, 0, 1)

    # ---- 皱眉互斥：浓眉降低时抑制 innerUp ----
    brow_down = max(bs["browDownLeft"], bs["browDownRight"])
    if brow_down > 0.2:
        bs["browInnerUp"] *= (1.0 - brow_down * 0.7)

    return bs

# ======================= 鼻子和脸颊 =======================
def solve_nose_cheeks(local_lm, eye_dist_base, mouth_width, current_bs):
    bs = {}

    # ---- 鼻翼皱 (sneer)：鼻子两侧上提，使用鼻翼到内眼角内缘的距离变化 ----
    # 129(左鼻翼下缘) vs 133(左内眼角) 的 Y 距离
    l_sneer = abs(local_lm[133][1] - local_lm[129][1]) / eye_dist_base
    r_sneer = abs(local_lm[362][1] - local_lm[358][1]) / eye_dist_base
    update_baseline("sneer_dist_l", l_sneer, current_bs)
    update_baseline("sneer_dist_r", r_sneer, current_bs)
    base_sl = get_baseline("sneer_dist_l")
    base_sr = get_baseline("sneer_dist_r")
    bs["noseSneerLeft"]  = remap_clip(l_sneer - base_sl, 0, max(0.01, base_sl * 0.5), 0, 1)
    bs["noseSneerRight"] = remap_clip(r_sneer - base_sr, 0, max(0.01, base_sr * 0.5), 0, 1)

    # ---- 面颊挤眼 (cheekSquint) ----
    # 使用下眼睑点(145/374)到脸颊点(50/280)的距离，挤眼/笑时这个距离变大
    l_squint = np.linalg.norm(local_lm[50] - local_lm[145]) / eye_dist_base
    r_squint = np.linalg.norm(local_lm[280] - local_lm[374]) / eye_dist_base
    update_baseline("cheek_squint_l", l_squint, current_bs)
    update_baseline("cheek_squint_r", r_squint, current_bs)
    base_csl = get_baseline("cheek_squint_l")
    base_csr = get_baseline("cheek_squint_r")
    # 挤眼时脸颊上提，距离变大
    bs["cheekSquintLeft"]  = remap_clip(l_squint - base_csl, 0,
                                         max(0.008, base_csl * 0.35), 0, 1)
    bs["cheekSquintRight"] = remap_clip(r_squint - base_csr, 0,
                                         max(0.008, base_csr * 0.35), 0, 1)

    # ---- cheekPuff：脸颊深度 + 嘴宽变化综合 ----
    mouth_width_norm = mouth_width / eye_dist_base
    update_baseline("mouth_width", mouth_width_norm, current_bs)
    base_mw = get_baseline("mouth_width")
    # 鼓嘴时嘴宽内收
    delta_narrow = max(0, base_mw - mouth_width_norm)

    # 脸颊 Z 轴突出：123(左脸颊)/352(右脸颊) 相对鼻尖
    nose_tip = local_lm[1]
    cheek_fwd = ((nose_tip[2] - local_lm[123][2]) + (nose_tip[2] - local_lm[352][2])) / (2 * eye_dist_base + 1e-6)
    update_baseline("cheek_puff_z", cheek_fwd, current_bs)
    base_cz = get_baseline("cheek_puff_z")
    delta_bulge = max(0, cheek_fwd - base_cz)

    puff_narrow = remap_clip(delta_narrow, 0.008, max(0.015, base_mw * 0.15), 0, 1)
    puff_bulge  = remap_clip(delta_bulge, 0.004, max(0.01, abs(base_cz) * 0.5 + 0.005), 0, 1)
    bs["cheekPuff"] = max(puff_narrow, puff_bulge)

    return bs

# ======================= 嘴部和下巴 =======================
def solve_mouth_jaw(local_lm, eye_dist_base, current_bs):
    bs = {}
    eye_dist = eye_dist_base

    # ---- 关键点 ----
    mouth_l   = local_lm[61]    # 左嘴角
    mouth_r   = local_lm[291]   # 右嘴角
    mouth_top = local_lm[13]    # 上唇中点
    mouth_bot = local_lm[14]    # 下唇中点
    chin      = local_lm[152]   # 下巴尖
    nose_tip  = local_lm[1]     # 鼻尖

    mouth_width  = np.linalg.norm(mouth_l - mouth_r)
    mouth_height = np.linalg.norm(mouth_top - mouth_bot)
    w_ratio = mouth_width / (eye_dist + 1e-6)

    # ======== jawOpen：双信号融合 ========
    # 信号1: 嘴高/眼距（说话、打哈欠）
    open_h = mouth_height / (eye_dist + 1e-6)
    # 信号2: 鼻尖到下巴Y距离 × 0.65，弥补闭嘴张嘴的感知
    chin_nose_y = max(0, nose_tip[1] - chin[1]) / (eye_dist + 1e-6)
    jaw_open_raw = max(open_h, chin_nose_y * 0.60)

    update_baseline("mouth_open", jaw_open_raw, current_bs)
    base_open = get_baseline("mouth_open")
    bs["jawOpen"] = remap_clip(jaw_open_raw, base_open + 0.008,
                                max(base_open + 0.55, 0.16), 0, 1)

    # mouthClose: 张嘴极小时触发
    if jaw_open_raw < base_open + 0.05:
        bs["mouthClose"] = remap_clip(base_open + 0.05 - jaw_open_raw, 0, 0.04, 0, 1)
    else:
        bs["mouthClose"] = 0.0

    # ======== 嘴宽基线 ========
    update_baseline("mouth_width", w_ratio, current_bs)
    base_mw = get_baseline("mouth_width")

    # ======== 嘴高宽比（圆唇信号） ========
    ratio = mouth_height / (mouth_width + 1e-6)
    update_baseline("mouth_ratio", ratio, current_bs)
    base_ratio = get_baseline("mouth_ratio")
    delta_ratio = max(0, ratio - base_ratio)   # 嘴巴变圆

    # ======== 微笑 / 撇嘴 ========
    mouth_center = (mouth_top + mouth_bot) / 2
    h_axis = mouth_r - mouth_l
    h_len = np.linalg.norm(h_axis)

    if h_len > 1e-6:
        v_axis = np.array([-h_axis[1], h_axis[0], 0.0])
        v_axis = v_axis / (np.linalg.norm(v_axis) + 1e-6)

        l_proj = np.dot(mouth_l - mouth_center, v_axis) / (eye_dist + 1e-6)
        r_proj = np.dot(mouth_r - mouth_center, v_axis) / (eye_dist + 1e-6)

        update_baseline("corner_y_l", l_proj, current_bs)
        update_baseline("corner_y_r", r_proj, current_bs)
        base_cy_l = get_baseline("corner_y_l")
        base_cy_r = get_baseline("corner_y_r")

        d_smile_l = l_proj - base_cy_l   # >0 上扬 = 微笑
        d_smile_r = r_proj - base_cy_r

        range_l = max(0.012, abs(base_cy_l) * 1.3 + 0.008)
        range_r = max(0.012, abs(base_cy_r) * 1.3 + 0.008)

        bs["mouthSmileLeft"]  = remap_clip(d_smile_l, 0, range_l, 0, 1)
        bs["mouthSmileRight"] = remap_clip(d_smile_r, 0, range_r, 0, 1)
        bs["mouthFrownLeft"]  = remap_clip(-d_smile_l, 0, range_l, 0, 1)
        bs["mouthFrownRight"] = remap_clip(-d_smile_r, 0, range_r, 0, 1)
    else:
        bs["mouthSmileLeft"] = bs["mouthSmileRight"] = 0.0
        bs["mouthFrownLeft"] = bs["mouthFrownRight"] = 0.0

    # ======== Funnel / Pucker：圆唇 + 嘴宽收窄 ========
    # 开口→Funnel, 闭口→Pucker
    open_w = bs.get("jawOpen", 0)

    # 圆唇信号：mouth ratio 增大
    round_raw = remap_clip(delta_ratio, 0, max(0.09, base_ratio * 0.80 + 0.04), 0, 1)

    # 嘴宽收窄信号（辅助）
    delta_narrow = max(0, base_mw - w_ratio)
    narrow_raw = remap_clip(delta_narrow, 0, max(0.025, base_mw * 0.50), 0, 1)

    round_combined = max(round_raw, narrow_raw * 0.65)
    bs["mouthFunnel"] = round_combined * open_w
    bs["mouthPucker"] = round_combined * (1.0 - open_w)

    # ======== Stretch：嘴变宽 ========
    delta_wide = max(0, w_ratio - base_mw)
    bs["mouthStretchLeft"]  = remap_clip(delta_wide, 0, max(0.035, base_mw * 0.60), 0, 1)
    bs["mouthStretchRight"] = bs["mouthStretchLeft"]

    # ======== 酒窝：微笑 × cheekSquint ========
    smile_l = bs.get("mouthSmileLeft", 0)
    smile_r = bs.get("mouthSmileRight", 0)
    squint_l = current_bs.get("cheekSquintLeft", 0) if current_bs else 0
    squint_r = current_bs.get("cheekSquintRight", 0) if current_bs else 0
    bs["mouthDimpleLeft"]  = min(smile_l, max(squint_l, smile_l * 0.55)) * 0.68
    bs["mouthDimpleRight"] = min(smile_r, max(squint_r, smile_r * 0.55)) * 0.68

    # ======== mouthRoll：上下唇 Z 轴外翻 ========
    corners_z_avg = (mouth_l[2] + mouth_r[2]) / 2
    z_upper = mouth_top[2] - corners_z_avg
    z_lower = mouth_bot[2] - corners_z_avg

    update_baseline("lip_z_upper", z_upper, current_bs)
    update_baseline("lip_z_lower", z_lower, current_bs)
    base_zu = get_baseline("lip_z_upper")
    base_zl = get_baseline("lip_z_lower")

    range_zu = max(0.005, eye_dist * 0.018)
    range_zl = max(0.005, eye_dist * 0.018)
    bs["mouthRollUpper"] = remap_clip(z_upper - base_zu, 0, range_zu, 0, 1)
    bs["mouthRollLower"] = remap_clip(z_lower - base_zl, 0, range_zl, 0, 1)

    # ======== mouthPress：抿嘴 ========
    press_h = max(0, base_open - jaw_open_raw)
    bs["mouthPressLeft"]  = remap_clip(press_h, 0.006, max(0.018, base_open * 0.90), 0, 1)
    bs["mouthPressRight"] = bs["mouthPressLeft"]

    # ======== mouthLowerDown：单侧下唇下拉 ========
    l_to_bot = abs(mouth_bot[1] - mouth_l[1]) / (eye_dist + 1e-6)
    r_to_bot = abs(mouth_bot[1] - mouth_r[1]) / (eye_dist + 1e-6)
    avg_to_bot = (l_to_bot + r_to_bot) / 2
    if avg_to_bot > 0.002:
        bs["mouthLowerDownLeft"]  = remap_clip(l_to_bot / avg_to_bot - 1.0, 0, 0.32, 0, 1)
        bs["mouthLowerDownRight"] = remap_clip(r_to_bot / avg_to_bot - 1.0, 0, 0.32, 0, 1)
    else:
        bs["mouthLowerDownLeft"] = bs["mouthLowerDownRight"] = 0.0

    # ======== mouthUpperUp：单侧上唇上提 ========
    l_upper_y = (mouth_top[1] - local_lm[40][1]) / (eye_dist + 1e-6)
    r_upper_y = (mouth_top[1] - local_lm[270][1]) / (eye_dist + 1e-6)
    bs["mouthUpperUpLeft"]  = remap_clip(l_upper_y, 0, max(0.006, eye_dist * 0.022), 0, 1)
    bs["mouthUpperUpRight"] = remap_clip(r_upper_y, 0, max(0.006, eye_dist * 0.022), 0, 1)

    # ======== mouthLeft/Right：嘴中心偏移 ========
    mouth_cx = (mouth_l[0] + mouth_r[0]) / 2
    face_cx = (local_lm[133][0] + local_lm[362][0]) / 2
    mouth_offset_x = (mouth_cx - face_cx) / (eye_dist + 1e-6)
    update_baseline("mouth_center_x", mouth_offset_x, current_bs)
    base_mcx = get_baseline("mouth_center_x")
    delta_mx = mouth_offset_x - base_mcx
    bs["mouthLeft"]  = remap_clip(-delta_mx, 0, max(0.025, abs(base_mcx) * 0.85 + 0.015), 0, 1)
    bs["mouthRight"] = remap_clip(delta_mx,  0, max(0.025, abs(base_mcx) * 0.85 + 0.015), 0, 1)

    # ======== 下巴左右 ========
    jaw_x = (chin[0] - face_cx) / (eye_dist + 1e-6)
    update_baseline("jaw_x", jaw_x, current_bs)
    base_jx = get_baseline("jaw_x")
    delta_jx = jaw_x - base_jx
    bs["jawLeft"]  = remap_clip(delta_jx,  0, max(0.015, abs(base_jx) * 0.5 + 0.01), 0, 1)
    bs["jawRight"] = remap_clip(-delta_jx, 0, max(0.015, abs(base_jx) * 0.5 + 0.01), 0, 1)

    # ======== 下巴前伸 ========
    dz = chin[2] - nose_tip[2]
    update_baseline("jaw_forward", dz, current_bs)
    base_fwd = get_baseline("jaw_forward")
    bs["jawForward"] = remap_clip(dz - base_fwd, 0, max(0.008, abs(base_fwd) * 0.3 + 0.005), 0, 1)

    # ======== 表情互斥 ========
    smile_max = max(bs.get("mouthSmileLeft", 0), bs.get("mouthSmileRight", 0))
    frown_max = max(bs.get("mouthFrownLeft", 0), bs.get("mouthFrownRight", 0))
    funnel_val = bs.get("mouthFunnel", 0)
    pucker_val = bs.get("mouthPucker", 0)
    stretch_val = bs.get("mouthStretchLeft", 0)

    # 微笑强 → 抑制 funnel/pucker/frown
    if smile_max > 0.2:
        decay = max(0, 1.0 - smile_max)
        bs["mouthFunnel"]  *= decay
        bs["mouthPucker"]  *= decay
        bs["mouthFrownLeft"]  *= (1.0 - smile_max * 0.85)
        bs["mouthFrownRight"] *= (1.0 - smile_max * 0.85)

    # funnel/pucker 强 → 抑制 stretch
    fp_max = max(funnel_val, pucker_val)
    if fp_max > 0.2:
        bs["mouthStretchLeft"]  *= (1.0 - fp_max * 0.7)
        bs["mouthStretchRight"] *= (1.0 - fp_max * 0.7)

    # stretch 强 → 抑制 funnel/pucker
    if stretch_val > 0.2:
        bs["mouthFunnel"] *= (1.0 - stretch_val * 0.65)
        bs["mouthPucker"] *= (1.0 - stretch_val * 0.65)

    # 撇嘴强 → 抑制微笑
    if frown_max > 0.2:
        bs["mouthSmileLeft"]  *= (1.0 - frown_max * 0.65)
        bs["mouthSmileRight"] *= (1.0 - frown_max * 0.65)

    return bs

# ======================= 主函数 =======================
def calc_all_arkit_coefficients(landmarks_468):
    """输入: MediaPipe 468 点字典 {index: [x, y, z]} → 输出: ARKit blendshape 字典"""
    global PREV_HEAD_ANGLES, _CALIB_FRAME

    if len(landmarks_468) == 0:
        return {}

    _CALIB_FRAME += 1
    if _CALIB_FRAME == _CALIB_FAST_FRAMES + 1:
        print(">>> 快速校准完成，切换为自适应模式")

    # 1. 头部旋转 + 局部坐标
    head_pose = solve_head_rotation(landmarks_468)
    local_lm = transform_to_local_space(landmarks_468, head_pose)

    # 记录头部角度用于 neutral 检测
    head_angles = np.array([head_pose["x"], head_pose["y"], head_pose["z"]])
    PREV_HEAD_ANGLES = head_angles

    # 眼距基准
    eye_dist = np.linalg.norm(local_lm[133] - local_lm[362])
    if eye_dist < 0.001:
        eye_dist = 0.05

    # 上一帧 blendshape（用于 neutral 条件判断）
    global PREV_BS
    current_bs_ref = PREV_BS if PREV_BS else {}

    # 注入头部角度供 neutral 检测使用
    current_bs_ref["_head_deg"] = head_angles

    # 2. 逐模块求解
    bs = {}
    bs.update(solve_eyes(local_lm, head_pose, eye_dist, current_bs_ref))
    bs.update(solve_brows(local_lm, eye_dist, current_bs_ref))

    mouth_width = np.linalg.norm(local_lm[61] - local_lm[291])
    bs.update(solve_nose_cheeks(local_lm, eye_dist, mouth_width, current_bs_ref))
    bs.update(solve_mouth_jaw(local_lm, eye_dist, current_bs_ref))

    # 3. 未实现的 blendshape 归零
    bs["mouthShrugLower"] = 0.0
    bs["mouthShrugUpper"] = 0.0
    bs["tongueOut"]       = 0.0

    # 4. 头部姿态（调试用，_ 前缀不会被发送）
    bs["_head_deg_x"] = head_pose["x"]
    bs["_head_deg_y"] = head_pose["y"]
    bs["_head_deg_z"] = head_pose["z"]

    # 5. 平滑
    bs = apply_smoothing(bs)

    # 6. 截断到 [0, 1]
    for k, v in bs.items():
        if not k.startswith("_"):
            bs[k] = np.clip(v, 0.0, 1.0)

    return bs

def get_expression_vector(bs_dict):
    """获取标准 ARKit 52 个 blendshape 向量"""
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
    """动态基准始终可用"""
    return True
