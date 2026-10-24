"""모델명, 생성 파라미터, 임계값 설정"""

LLM_NAME = "Qwen/Qwen2.5-7B-Instruct"
FEATURE_COLUMNS = ["phone_use", "head_angle", "shoulder_tilt",
                   "hand_face", "blink", "perclos"]


# 생성 파라미터
MAX_NEW_TOKENS = 2000
TEMPERATURE = 0.6
TOP_P = 0.9
REPETITION_PENALTY = 1.1

# _compare 임계값
REL = 0.2
EPS = 1e-3