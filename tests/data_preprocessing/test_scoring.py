import numpy as np
from data_preprocessing.preprocessing.scoring import compute_segment_scores_7features


def _eye(x0, d, y=200):
    return np.array([[x0, y], [x0+10, y-d], [x0+20, y-d],
                     [x0+30, y], [x0+20, y+d], [x0+10, y+d]], dtype=np.float32)


def _run(closed_idx, n=75, d_open=4.5, d_closed=0.75):
    eyes = [(_eye(250, d_closed if t in closed_idx else d_open),
             _eye(330, d_closed if t in closed_idx else d_open)) for t in range(n)]
    return compute_segment_scores_7features(
        processed_frames=[True] * n, pose_keypoints=[None] * n, eyes_landmarks=eyes,
        face_bboxes=[(200, 100, 400, 350)] * n, frame_sizes=[(640, 480)] * n,
        segment_size=25, fps=5,
    )


def test_blink_in_first_segment_uses_personal_baseline():
    # 눈이 큰 사람: 뜬 눈 EAR 0.5, 깜빡임 순간 EAR 0.2
    # 기본값 0.25 기준(임계 0.138)으로는 0.2를 감김으로 못 잡고, baseline 0.5 기준(임계 0.275)으로는 잡힘
    res = _run({5, 30}, d_open=7.5, d_closed=3.0)
    assert res[0]["raw"]["blink_rate_proxy"] > 0   # seg0 깜빡임 검출
    assert res[1]["raw"]["blink_rate_proxy"] > 0   # seg1 깜빡임 검출
    assert res[0]["raw"]["EAR_baseline"] == res[1]["raw"]["EAR_baseline"]


def test_closure_long_across_segment_boundary():
    res = _run(set(range(20, 32)))  # 2.4초, seg0→seg1 경계
    assert [r["raw"]["sleepy_flag"] for r in res] == [0, 1, 0]


def test_short_blinks_not_sleepy():
    res = _run({10, 30, 31, 55})    # 0.2~0.4초 깜빡임만
    assert [r["raw"]["sleepy_flag"] for r in res] == [0, 0, 0]


def test_phone_use_ratio_from_scores():
    n = 75
    phone = [None] * n
    for t in range(0, n, 5):        # 1초 간격 검사
        phone[t] = 0.0
    for t in (25, 30, 35):          # seg1에서 3번 검출
        phone[t] = 0.4
    res = compute_segment_scores_7features(
        processed_frames=[True] * n, pose_keypoints=[None] * n,
        eyes_landmarks=[None] * n, face_bboxes=[(200, 100, 400, 350)] * n,
        frame_sizes=[None] * n,     # 얼굴 미검출이어도 폰 판정은 되어야 함
        segment_size=25, fps=5, phone_scores=phone,
    )
    assert [r["raw"]["usage_ratio"] for r in res] == [0.0, 0.6, 0.0]