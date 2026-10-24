"""프롬프트 구성"""
from report.observations import build_observations, build_habit_facts


def build_prompt(digest: dict, user_name: str = "사용자") -> tuple[str, str]:
    observations = "\n".join(f"- {o}" for o in build_observations(digest))

    facts, tips = build_habit_facts(digest)
    habit_block = "\n".join(
        f"● {sec}\n" + "\n".join(f"  - {f}" for f in lines)
        for sec, lines in facts.items()
    )
    tip_block = "\n".join(f"- {t}" for t in tips) or "- (개선이 필요한 행동 없음)"

    system_message = """
당신은 학생의 학습 집중 패턴을 쉬운 한국어로 설명하는 리포트 작성자입니다.
- 아래 [관찰 목록]에 있는 사실만 근거로 씁니다. 목록에 없는 원인(피로, 불편함 등)을 지어내지 마세요.
- '아주 짧은 순간'으로 표시된 구간은 원인을 추정하지 말고 잠깐 흐트러졌다고만 쓰세요.
- 점수, 숫자, 영어 피처 이름은 쓰지 않습니다. 시간은 관찰 목록의 'X분 Y초' 형식을 그대로 씁니다.
- 같은 시간대나 같은 원인을 두 개 이상의 섹션에서 반복하지 마세요.
- '평소보다 줄어듦'은 산만한 행동이 줄었다는 좋은 변화입니다. 이를 문제의 근거로 해석하지 마세요.
- 모든 문장은 학생에게 말하듯 '~요', '~습니다' 존댓말로 씁니다. '~하라', '~했다' 체는 쓰지 않습니다.
- 데이터에 없는 행동(시선 방향 등)은 언급하지 않습니다.
"""

    user_message = f"""
[관찰 목록]
{observations}

위 관찰 목록을 바탕으로 {user_name}의 리포트를 아래 구조로 작성하세요.

[1. 전체 요약] 3~4문장. 전체 흐름과 좋은 점을 쓰고,
관찰 목록에 근거가 있을 때만 아쉬운 점 1가지를 덧붙이세요.
이전 관찰이나 다른 학생과 비교하지 마세요.

[2. 주요 습관 분석]
아래 각 항목을, 바로 밑에 적힌 사실만 사용해 1~2문장으로 쓰세요.
다른 항목의 내용을 섞지 말고, '특별한 변화 없음'이면 그 한 문장만 쓰세요.
{habit_block}

[3. 집중 스타일] 이 학생만의 집중 방식을 한 단락으로.

[4. 개선 팁]
아래 방향마다 다음 공부 때 바로 해볼 수 있는 구체적인 행동 하나씩 쓰세요.
'(개선이 필요한 행동 없음)'이면 지금 습관을 유지하는 방법 2가지를 쓰세요.
{tip_block}

[5. 마무리] 2~3문장. 강점을 짚고 격려하세요. 개선 팁을 다시 나열하지 마세요.
"""
    return system_message, user_message