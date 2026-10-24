"""요약 수치"""
from report.config import EPS, REL

# 실제로 측정하는 피처만 항목으로 (시선은 측정하지 않으므로 제외)
# phone_use는 phone_bboxes=None이라 항상 1점 → 변화가 없어 자동으로 빠짐
SECTION_FEATURES = {
    "고개 자세": ["head_angle"],
    "어깨·상체 자세": ["shoulder_tilt"],
    "손·얼굴 움직임": ["hand_face", "phone_use"],
    "눈 깜빡임·졸음": ["blink", "perclos"],
}

# 행동이 늘었을 때 제안할 팁의 '방향' (구체적인 문장은 LLM이 작성)
TIP_DIRECTIONS = {
    "head_angle": "고개 자세를 바로잡는 방법 (모니터·책 높이 조절, 목 스트레칭 등)",
    "shoulder_tilt": "상체가 기울지 않게 하는 방법 (의자·책상 자세, 어깨 스트레칭 등)",
    "hand_face": "손이 얼굴로 가지 않게 하는 방법 (손에 필기구 쥐기 등)",
    "phone_use": "휴대폰을 멀리 두는 방법",
    "blink": "눈의 피로를 줄이는 방법 (화면 거리, 짧은 눈 휴식 등)",
    "perclos": "졸음을 줄이는 방법 (짧은 휴식, 환기, 가벼운 움직임 등)",
}

# CSV 피처는 모두 '높을수록 좋은' 점수
FEATURE_LABELS = {
    "phone_use": "휴대폰을 보는 행동",
    "head_angle": "고개가 기울어지는 정도",
    "shoulder_tilt": "어깨·상체가 기울어지는 정도",
    "hand_face": "손이 얼굴 쪽으로 가는 행동",
    "blink": "눈 깜빡임",
    "perclos": "졸음 신호",
}


# CSV 점수는 높을수록 좋음 → 점수 하락 = 행동 증가
def _describe(col: str, lvl: str) -> str:

    change = "평소보다 늘어남" if lvl == "적음" else "평소보다 줄어듦"
    return f"{FEATURE_LABELS[col]}: {change}"


# 전체 평균 대비 '많음/적음/비슷' 판정
def _compare(value: float, base: float, rel: float = REL, eps: float = EPS) -> str:

    if abs(base) < eps:
        return "비슷" if abs(value) < eps else "많음"
    ratio = (value - base) / abs(base)
    if ratio > rel:
        return "많음"
    if ratio < -rel:
        return "적음"
    return "비슷"

# digest의 수치를 LLM이 그대로 옮겨 쓸 수 있는 사실 문장으로 변환
def build_observations(digest: dict) -> list[str]:

    obs = []
    base = digest["overall_feature_avg"]
    phases = digest["phases"]

    obs.append(f"영상 길이: {digest['total_duration']}")
    obs.append(f"전반부 대비 후반부 집중 추세: {digest['trend']}")

    scores = {k: v["avg_score"] for k, v in phases.items()}
    best, worst = max(scores, key=scores.get), min(scores, key=scores.get)
    if scores[best] - scores[worst] < 0.05:
        obs.append("초반·중반·후반의 집중도 차이는 크지 않음")
    else:
        obs.append(f"집중이 가장 잘 된 구간: {best}({phases[best]['time_range']})")
        obs.append(f"집중이 가장 낮았던 구간: {worst}({phases[worst]['time_range']})")

    for lp in digest["low_focus_periods"]:
        diffs = [
            _describe(c,lvl)
            for c, d in lp["feature_delta"].items()
            if (lvl := _compare(base[c] + d, base[c])) != "비슷"
        ]
        length = "아주 짧은 순간" if lp["duration_sec"] <= 10 else f"약 {lp['duration_sec']}초간"
        obs.append(
            f"집중이 낮았던 순간: {lp['time_range']} ({length}) — "
            + (", ".join(diffs) if diffs else "뚜렷한 행동 변화 없음")
        )

    ev = digest["events"]
    obs.append("자리 이탈: " + (", ".join(ev["seat_departed_times"]) or "없음"))
    obs.append("졸음 신호: " + (", ".join(ev["sleepy_times"]) or "없음"))
    return obs


# 항목별 사실 목록과, 개선이 필요한 피처의 팁 방향 목록을 반환
def build_habit_facts(digest: dict) -> tuple[dict[str, list[str]], list[str]]:

    base = digest["overall_feature_avg"]
    facts, tip_cols = {}, []

    for section, cols in SECTION_FEATURES.items():
        lines = []
        for phase, p in digest["phases"].items():
            for c in cols:
                lvl = _compare(p["features"][c], base[c])
                if lvl == "비슷":
                    continue
                lines.append(f"{phase}에 {_describe(c, lvl)}")
                if lvl == "적음" and c not in tip_cols:
                    tip_cols.append(c)
        facts[section] = lines or ["특별한 변화 없음"]

    tips = [TIP_DIRECTIONS[c] for c in tip_cols]
    return facts, tips