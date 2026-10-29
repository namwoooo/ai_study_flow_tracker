import numpy as np
from data_preprocessing.preprocessing.scoring import compute_segment_scores_7features


def _eye(x0, d, y=200):
    return np.array([[x0, y], [x0+10, y-d], [x0+20, y-d],
                     [x0+30, y], [x0+20, y+d], [x0+10, y+d]], dtype=np.float32)


def _run(closed_idx, n=75):
    eyes = [(_eye(250, 0.75 if t in closed_idx else 4.5),
             _eye(330, 0.75 if t in closed_idx else 4.5)) for t in range(n)]
    return compute_segment_scores_7features(
        processed_frames=[True] * n, pose_keypoints=[None] * n, eyes_landmarks=eyes,
        face_bboxes=[(200, 100, 400, 350)] * n, frame_sizes=[(640, 480)] * n,
        segment_size=25, fps=5,
    )


def test_closure_long_across_segment_boundary():
    res = _run(set(range(20, 32)))  # 2.4초, seg0→seg1 경계
    assert [r["raw"]["sleepy_flag"] for r in res] == [0, 1, 0]


def test_short_blinks_not_sleepy():
    res = _run({10, 30, 31, 55})    # 0.2~0.4초 깜빡임만
    assert [r["raw"]["sleepy_flag"] for r in res] == [0, 0, 0]