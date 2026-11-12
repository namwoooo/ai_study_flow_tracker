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
        "hand_face_dist":   (0.5, 2.0),
        "blink_rate_proxy": (0.02, 0.30),
        "perclos":          (0.05, 0.25),   # 평상시 -0.05, 졸음 경계 -0.15
        "head_yaw_deg":     (6.0, 15.0),   # focus 세그먼트 p90 ≈ 6°, away_speaking p50 ≈ 10~12°
        "head_up_deg":      (4.0, 20.0),   # focus p90 ≈ 4°, look_up p50 ≈ 23~26°
    },
    # 가중치는 합으로 정규화되므로 합이 1일 필요 없음
    "w": {
        "phone_use":     0.0,
        "seat_depart":   0.0,     # 자리 비움은 status(absent) 처리, 점수에서 제외
        "head_angle":    0.12,
        "shoulder_tilt": 0.12,
        "hand_face":     0.16,
        "blink":         0.16,
        "perclos":       0.16,
        "head_pose":     0.16,
    },
    "seat_departure_penalty": 0.0,
}

# FaceLandmarker blendshapes 중 저장할 항목 (하품, 눈만 돌리는 시선 이탈, 깜빡이 보조)
BLENDSHAPE_KEYS = (
    "jawOpen",
    "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookInLeft", "eyeLookOutLeft", "eyeLookUpLeft", "eyeLookDownLeft",
    "eyeLookInRight", "eyeLookOutRight", "eyeLookUpRight", "eyeLookDownRight",
)

HEAD_REF_WINDOW_SEC = 600   # 기준 자세 = 직전 10분 head pose 중앙값 (과거만 사용 → 실시간 가능)

BODY_MIN_WIDTH_RATIO = 0.6      # 어깨 폭이 평소(얼굴 보일 때)의 이 비율 이상이어야 책상 앞의 몸으로 인정

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
PERCLOS_WINDOW_SEC = 60         # PERCLOS 계산 창(세그먼트 끝 기준 직전 N초)

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
    "hand":     1.0,
    "blink":    0.05,
    "phone":    0.9,
}

# 가중치는 합으로 정규화되므로 합이 1일 필요 없음
STABILITY_WEIGHTS = {
    "head":     0.25,
    "shoulder": 0.25,
    "hand":     0.20,
    "blink":    0.15,
    "phone":    0.0,
}

ABSENT_FACE_MISSING_RATIO = 0.6   # 세그먼트 내 얼굴 미검출 비율이 이 이상이면 자리 비움
BODY_VISIBLE_RATIO = 0.5   # 얼굴이 없어도 양 어깨가 보이는 샘플 비율이 이 이상이면 face_hidden(자리에 있음)
SLEEPY_PENALTY = 0.5            # 졸음 시 stability에 곱함 (50% 감점)

FINAL_MIX_WEIGHT = 0.6          # personalized = 0.6 * final + 0.4 * stability
STABILITY_MIX_WEIGHT = 0.4

HAND_ABSENT_SCORE = 1.0  # 손목이 화면에 없을 때 (책상 위로 간주)

EYE_GATE_DOWN_DEG = 5.0     # 기준 자세보다 이 이상 숙이면 눈 판정 불가 (아래 응시 ≈ 감은 눈으로 보임)
PERCLOS_MIN_SAMPLES = 25    # PERCLOS 창에 판정 가능 샘플이 이보다 적으면 중립(0.5)
BLINK_MIN_SAMPLES = 10      # 세그먼트에 판정 가능 샘플이 이보다 적으면 깜빡임 중립(0.5)

# 개인화: 직전 10분 present 세그먼트 대비 나빠진 방향 변화만 감점
PERSONAL_WINDOW_SEC = 600
PERSONAL_MIN_SEGMENTS = 12   # 기준 구간에 present 세그먼트가 이보다 적으면(시작 후 약 1분) final 그대로 사용

# source: scores(0~1 점수, 낮을수록 나쁨) 또는 raw(원시값)
# floor: 스케일 하한 -> 평소 값이 거의 일정한 피처가 작은 변화에 과민 반응하지 않도록
# 가중치는 합으로 정규화되므로 합이 1일 필요 없음
PERSONAL_FEATURES = {
    "perclos":    {"source": "raw",    "key": "perclos_ratio",    "higher_is_worse": True,  "floor": 0.05, "weight": 0.25},
    "blink":      {"source": "raw",    "key": "blink_rate_proxy", "higher_is_worse": True,  "floor": 0.03, "weight": 0.15},
    "head_pose":  {"source": "scores", "key": "head_pose",        "higher_is_worse": False, "floor": 0.15, "weight": 0.25},
    "head_angle": {"source": "scores", "key": "head_angle",       "higher_is_worse": False, "floor": 0.15, "weight": 0.10},
    "shoulder":   {"source": "scores", "key": "shoulder_tilt",    "higher_is_worse": False, "floor": 0.15, "weight": 0.10},
    "hand":       {"source": "scores", "key": "hand_face",        "higher_is_worse": False, "floor": 0.15, "weight": 0.15},
}