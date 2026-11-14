import numpy as np
import pandas as pd

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

def gaze_h_series(blendshapes):
    """샘플별 수평 시선. 한쪽 방향 응시 시 양 눈의 In/Out이 반대로 움직이는 것을 합친 값.
    얼굴 미검출 샘플은 NaN."""
    out = np.full(len(blendshapes), np.nan)
    for i, b in enumerate(blendshapes):
        if b is None:
            continue
        out[i] = ((b["eyeLookOutLeft"] + b["eyeLookInRight"]) / 2
                  - (b["eyeLookInLeft"] + b["eyeLookOutRight"]) / 2)
    return out

def rolling_past_median(x, window, min_periods):
    """각 시점 **이전** window 샘플의 중앙값 (현재 샘플 제외 → 실시간 계산 가능). NaN은 건너뜀."""
    return pd.Series(x).rolling(window, min_periods=min_periods).median().shift(1).to_numpy()


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


def matrix_to_euler_deg(m):
    """
    FaceLandmarker 4x4 변환 행렬 → (pitch, yaw, roll) 도(degree)

    회전 순서 R = Rz(roll) @ Ry(yaw) @ Rx(pitch) 기준
    행렬에 스케일이 섞여 있을 수 있어 열 단위로 정규화
    부호 방향(왼쪽/오른쪽, 위/아래)은 실제 영상으로 확인
    """
    R = np.asarray(m, dtype=np.float64)[:3, :3]
    R = R / np.linalg.norm(R, axis=0, keepdims=True)
    pitch = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(-R[2, 0], np.hypot(R[2, 1], R[2, 2]))
    roll = np.arctan2(R[1, 0], R[0, 0])
    return tuple(float(v) for v in np.degrees([pitch, yaw, roll]))


def pose_visible(pose_vec, idx, vis_thresh=POSE_VIS_THRESH):
    """idx번 랜드마크가 보이는지 (좌표 계산 없이 visibility만 확인)."""
    if pose_vec is None:
        return False
    base = idx * POSE_STRIDE
    return base + 3 < len(pose_vec) and pose_vec[base + 3] >= vis_thresh


def shoulder_width(pose_vec, vis_thresh=POSE_VIS_THRESH):
    """양 어깨 사이 가로 거리 (화면 너비 대비 0~1). 어깨가 안 보이면 NaN."""
    if pose_vec is None:
        return float("nan")
    l, r = POSE_IDX["shoulder_l"] * POSE_STRIDE, POSE_IDX["shoulder_r"] * POSE_STRIDE
    if pose_vec[l + 3] < vis_thresh or pose_vec[r + 3] < vis_thresh:
        return float("nan")
    return abs(pose_vec[l] - pose_vec[r])


def blendshape_series(blendshapes, name):
    """샘플별 blendshape 값 배열. 얼굴 미검출 샘플은 NaN."""
    return np.array(
        [b.get(name, np.nan) if b is not None else np.nan for b in blendshapes],
        dtype=float,
    )


def detect_yawn_onsets(jaw, thresh, min_frames, peak_thresh):
    """하품으로 확정되는 샘플에만 True.

    - jaw > thresh가 이어지는 구간에서, 길이 ≥ min_frames 이고 구간 최댓값 ≥ peak_thresh가
      처음 함께 충족되는 순간 1회 (말하기: 짧거나 얕게 열림 / 하품: 길고 크게 열림)
    - 길게 이어져도 구간당 1회만, NaN(얼굴 미검출)은 구간을 끊음
    - 전체 시리즈에서 계산하므로 세그먼트 경계에 걸친 하품도 정확히 1회
    """
    onset = np.zeros(len(jaw), dtype=bool)
    run, peak, counted = 0, 0.0, False
    for i, v in enumerate(jaw):
        if np.isfinite(v) and v > thresh:
            run += 1
            peak = max(peak, v)
            if not counted and run >= min_frames and peak >= peak_thresh:
                onset[i] = True
                counted = True
        else:
            run, peak, counted = 0, 0.0, False
    return onset


def detect_nod_events(pitch, drop_deg, drop_frames, recover_deg, recover_frames):
    """앞으로 급히 숙였다가(pitch +) 곧 다시 드는 동작. 숙인 시점 샘플에 True, 동작당 1회.

    - pitch[i] − pitch[i − drop_frames] > drop_deg 이고
      이후 recover_frames 안에 recover_deg 이상 다시 들면 고개 떨굼
    - 숙인 채 머무는 동작(필기)은 회복 조건에서 걸러짐
    - 회복 확인에 미래 샘플이 필요하므로 실시간에서는 recover_frames만큼 늦게 확정됨
    """
    n = len(pitch)
    out = np.zeros(n, dtype=bool)
    i = drop_frames
    while i < n:
        a, b = pitch[i - drop_frames], pitch[i]
        if np.isfinite(a) and np.isfinite(b) and b - a > drop_deg:
            after = pitch[i + 1:i + 1 + recover_frames]
            if np.isfinite(after).any() and b - np.nanmin(after) > recover_deg:
                out[i] = True
                i += drop_frames + recover_frames   # 같은 동작을 중복으로 세지 않음
                continue
        i += 1
    return out