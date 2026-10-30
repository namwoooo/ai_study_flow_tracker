# 샘플링 / 세그먼트
TARGET_FPS = 5                         # 초당 샘플링 프레임 수
SEGMENT_SEC = 5                        # 세그먼트 길이(초)
SEGMENT_SIZE = TARGET_FPS * SEGMENT_SEC  # 세그먼트당 샘플 수 (= 25)


# MediaPipe 모델 경로
FACE_MODEL_PATH = "face_landmarker.task"
POSE_MODEL_PATH = "pose_landmarker.task"


# 세그먼트 점수 계산 (compute_segment_scores_7features)
# 모듈 전역 dict이므로 직접 수정하지 말고 scoring._resolve_params()를 거칠 것
DEFAULT_SCORING_PARAMS = {
    "ear_thresh": 0.21,
    "time_decay_lambda": 0.08,
    "norm": {
        "head_angle_rad":   (0.0, 0.35),
        "shoulder_tilt":    (0.0, 0.25),
        "hand_face_dist":   (30.0, 200.0),
        "blink_rate_proxy": (0.02, 0.30),
        "perclos":          (0.00, 0.35),
    },
    "w": {
        "phone_use":     0.16,
        "seat_depart":   0.12,
        "head_angle":    0.12,
        "shoulder_tilt": 0.12,
        "hand_face":     0.16,
        "blink":         0.16,
        "perclos":       0.16,
    },
    "seat_departure_penalty": 0.0,
}

PHONE_USE_THRESH = 25.0         # 손-폰 거리 임계값(px)

# 자리 이탈 기준점 자동 설정
REF_WINDOW_MIN = 30             # 기준점 계산에 쓰는 최소 초기 샘플 수
REF_RADIUS_MIN = 60.0           # 기준 반경 하한(px)
REF_RADIUS_DEFAULT = 120.0      # 얼굴 검출이 부족할 때 기본 반경(px)

# EAR / 졸음
EAR_BASELINE_DEFAULT = 0.25     # baseline 계산 전 기본값
EAR_BASELINE_MIN_SAMPLES = 30   # compute_EAR_baseline 최소 샘플 수
EYE_CLOSURE_RATIO = 0.55        # EAR < baseline * ratio 이면 감김
CLOSURE_LONG_SEC = 2.0          # 이 시간 이상 감김 지속 시 졸음
HEAD_NOD_THRESH = 0.15          # head nod 판정 각도 변화(rad)

# 해당 값 미만의 pose 랜드마크 무시
POSE_VIS_THRESH = 0.5

# 개인화
WARMUP_SEGMENTS = 120           # 5초 × 120 = 10분

# 안정 구간 필터 (이 조건에 걸리면 baseline 계산에서 제외)
STABILITY_FILTER = {
    "blink_rate_max":    0.25,
    "hand_face_min":     0.25,
    "phone_use_min":     0.5,
    "head_angle_min":    0.3,
    "shoulder_tilt_min": 0.3,
}

# 안정 구간이 하나도 없을 때 쓰는 기본 평균
BASELINE_DEFAULT_MEANS = {
    "head":     0.7,
    "shoulder": 0.7,
    "hand":     0.5,
    "blink":    0.05,
    "phone":    0.9,
}

STABILITY_WEIGHTS = {
    "head":     0.25,
    "shoulder": 0.25,
    "hand":     0.20,
    "blink":    0.15,
    "phone":    0.15,
}

SEAT_DEPART_PENALTY = 0.3       # 자리 이탈 시 stability에 곱함 (70% 감점)
SLEEPY_PENALTY = 0.5            # 졸음 시 stability에 곱함 (50% 감점)

FINAL_MIX_WEIGHT = 0.6          # personalized = 0.6 * final + 0.4 * stability
STABILITY_MIX_WEIGHT = 0.4