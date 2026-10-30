from importlib.metadata import pass_none

import numpy as np

from .config import EAR_BASELINE_DEFAULT, EAR_BASELINE_MIN_SAMPLES, EYE_CLOSURE_RATIO, POSE_VIS_THRESH

# 랜드마크 인덱스
# FaceLandmarker 눈 6점 (p1..p6 순서)
LEFT_EYE_IDXS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDXS = [362, 385, 387, 263, 373, 380]

# BlazePose 인덱스
POSE_IDX = {
    "nose": 0,
    "eye_l": 2, "eye_r": 5,
    "shoulder_l": 11, "shoulder_r": 12,
    "wrist_l": 15, "wrist_r": 16,
    "hip_l": 23, "hip_r": 24,
}

POSE_STRIDE = 4  # x, y, z, visibility


# 기하 / 정규화
def to_score(x, lo, hi, small_is_good=True):
    """원시값 x를 [lo, hi]로 클리핑한 뒤 0~1 점수로 변환."""
    x = np.clip(x, lo, hi)
    if small_is_good:
        return 1.0 - (x - lo) / (hi - lo + 1e-8)
    return (x - lo) / (hi - lo + 1e-8)


def get_pose_xy(pose_vec, idx, w, h, vis_thresh=POSE_VIS_THRESH):
    """idx번 랜드마크의 픽셀 좌표 / 범위 밖이거나 visibility가 낮으면(0.5 아래) None"""
    base = idx * POSE_STRIDE
    if base + 3 >= len(pose_vec):
        return None
    x_n, y_n, vis = pose_vec[base], pose_vec[base + 1], pose_vec[base + 3]
    if vis < vis_thresh:
        return None
    if not (0 <= x_n <= 1 and 0 <= y_n <= 1):
        return None
    return (x_n * w, y_n * h)


def center_of_bbox(b):
    x1, y1, x2, y2 = b
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)


def angle_from_horizontal(p_left, p_right):
    """두 점을 잇는 선과 수평선 사이의 각도(0 ~ π/2) / 좌우 순서와 무관하게 설정"""
    v = np.array(p_right) - np.array(p_left)
    a = abs(np.arctan2(v[1], v[0]))
    return min(a, np.pi - a)


# 눈 / EAR
def eye_aspect_ratio_6pts(eye6):
    """6점 눈 랜드마크(shape (6, 2), 픽셀 좌표)로 EAR 계산."""
    p1, p2, p3, p4, p5, p6 = eye6
    vert = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horiz = 2.0 * np.linalg.norm(p1 - p4) + 1e-8
    return float(vert / horiz)


def compute_EAR_baseline(ear_series, min_samples=EAR_BASELINE_MIN_SAMPLES):
    ear_series = np.array(ear_series)
    ear_series = ear_series[np.isfinite(ear_series)]
    if len(ear_series) < min_samples:
        return np.median(ear_series) if len(ear_series) > 0 else EAR_BASELINE_DEFAULT
    return np.median(ear_series[:min_samples])


def is_eye_closure(ear, ear_baseline, ratio=EYE_CLOSURE_RATIO):
    return ear < (ear_baseline * ratio)

def compute_ear_series(eyes_landmarks):
    """프레임별 양안 평균 EAR / 눈 랜드마크가 없거나 계산 실패 시 NaN"""
    ears = np.full(len(eyes_landmarks), np.nan, dtype=np.float64)
    for t, eyes in enumerate(eyes_landmarks):
        if eyes is None:
            continue
        try:
            eyeL, eyeR = eyes
            ears[t] = 0.5 * (eye_aspect_ratio_6pts(eyeL) + eye_aspect_ratio_6pts(eyeR))
        except Exception:
            pass
    return ears


def weighted_mean(values, weights):
    """weights 기준 가중 평균 적용"""
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("가중치 합이 0 이하")
    return sum(weights[k] * values[k] for k in weights) / total