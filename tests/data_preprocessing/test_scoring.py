import numpy as np
import pytest
from data_preprocessing.preprocessing.scoring import compute_segment_scores_7features
from data_preprocessing.preprocessing.features import (POSE_IDX, matrix_to_euler_deg, gaze_h_series,
                                                       rolling_past_median, detect_yawn_onsets)
from data_preprocessing.preprocessing.personalization import apply_personalization

NAN = float("nam")

def _eye(x0, d, y=200):
    return np.array([[x0, y], [x0+10, y-d], [x0+20, y-d],
                     [x0+30, y], [x0+20, y+d], [x0+10, y+d]], dtype=np.float32)


def _run(closed_idx, n=75, d_open=4.5, d_closed=0.75, head_pose=None):
    eyes = [(_eye(250, d_closed if t in closed_idx else d_open),
             _eye(330, d_closed if t in closed_idx else d_open)) for t in range(n)]
    return compute_segment_scores_7features(
        processed_frames=[True] * n, pose_keypoints=[None] * n, eyes_landmarks=eyes,
        face_bboxes=[(200, 100, 400, 350)] * n, frame_sizes=[(640, 480)] * n,
        segment_size=25, fps=5, head_pose=head_pose,
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


def test_perclos_ratio_uses_trailing_window():
    res = _run(set(range(20, 32)))   # 12프레임 감김
    ratios = [r["raw"]["perclos_ratio"] for r in res]
    # seg0: 5/25, seg1: 12/50, seg2: 12/75
    assert ratios == pytest.approx([0.2, 0.24, 0.16], abs=1e-4)


def test_perclos_no_closure_is_full_score():
    res = _run(set())
    assert all(r["scores"]["perclos"] == 1.0 for r in res)

def _pose(wrist_xy=None, w=640, h=480):
    vec = [0.0] * (33 * 4)            # visibility 0 → 모든 랜드마크 무시
    if wrist_xy is not None:
        i = POSE_IDX["wrist_r"] * 4
        vec[i:i + 4] = [wrist_xy[0] / w, wrist_xy[1] / h, 0.0, 1.0]
    return vec


def test_hand_face_absent_and_near():
    n = 50
    # 얼굴 bbox (200,100,400,350) → 중심 (300,225), 높이 250
    # seg1: 손목이 중심에서 100px 아래 → 0.4배 → 0점
    pose = [_pose() for _ in range(25)] + [_pose((300, 325)) for _ in range(25)]
    res = compute_segment_scores_7features(
        processed_frames=[True] * n, pose_keypoints=pose, eyes_landmarks=[None] * n,
        face_bboxes=[(200, 100, 400, 350)] * n, frame_sizes=[(640, 480)] * n,
        segment_size=25, fps=5,
    )
    assert [r["scores"]["hand_face"] for r in res] == [1.0, 0.0]
    assert [r["raw"]["hand_visible_ratio"] for r in res] == [0.0, 1.0]


def _mat(pitch=0.0, yaw=0.0, roll=0.0, scale=1.7):
    p, y, r = np.radians([pitch, yaw, roll])
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    M = np.eye(4)
    M[:3, :3] = (Rz @ Ry @ Rx) * scale
    return M


@pytest.mark.parametrize("pitch,yaw,roll", [(0, 0, 0), (15, 0, 0), (0, -25, 0), (0, 0, 10), (-10, 20, 5)])
def test_matrix_to_euler_deg(pitch, yaw, roll):
    assert matrix_to_euler_deg(_mat(pitch, yaw, roll)) == pytest.approx((pitch, yaw, roll), abs=1e-6)


def test_head_pose_yaw_and_up_penalized_down_tolerated():
    n = 100
    hp = [(-5.0, -2.0, 0.0)] * 50          # seg0, seg1: 기준 자세
    hp += [(-5.0, -22.0, 0.0)] * 25        # seg2: 옆으로 20°
    hp += [(15.0, -2.0, 0.0)] * 25         # seg3: 아래로 20° (필기) → 감점 없음
    res = compute_segment_scores_7features(
        processed_frames=[True] * n, pose_keypoints=[None] * n, eyes_landmarks=[None] * n,
        face_bboxes=[(200, 100, 400, 350)] * n, frame_sizes=[(640, 480)] * n,
        segment_size=25, fps=5, head_pose=hp,
    )
    assert [r["scores"]["head_pose"] for r in res] == [1.0, 1.0, 0.0, 1.0]


import math

def test_absent_segment_excluded():
    n = 75
    sizes = [(640, 480)] * 25 + [None] * 25 + [(640, 480)] * 25   # seg1: 얼굴 없음
    res = compute_segment_scores_7features(
        processed_frames=[True] * n, pose_keypoints=[None] * n, eyes_landmarks=[None] * n,
        face_bboxes=[(200, 100, 400, 350)] * n, frame_sizes=sizes, segment_size=25, fps=5,
    )
    assert [r["flags"]["status"] for r in res] == ["present", "absent", "present"]
    assert math.isnan(res[1]["scores"]["final"])
    assert not math.isnan(res[0]["scores"]["final"])


def _pose_shoulders():
    vec = [0.0] * (33 * 4)
    for k in ("shoulder_l", "shoulder_r"):
        i = POSE_IDX[k] * 4
        vec[i:i + 4] = [0.5, 0.8, 0.0, 1.0]
    return vec


def test_face_hidden_vs_absent():
    n = 75
    sizes = [(640, 480)] * 25 + [None] * 50                           # seg1, seg2: 얼굴 없음
    pose = [None] * 25 + [_pose_shoulders()] * 25 + [None] * 25       # seg1만 어깨 보임
    res = compute_segment_scores_7features(
        processed_frames=[True] * n, pose_keypoints=pose, eyes_landmarks=[None] * n,
        face_bboxes=[(200, 100, 400, 350)] * n, frame_sizes=sizes, segment_size=25, fps=5,
    )
    assert [r["flags"]["status"] for r in res] == ["present", "face_hidden", "absent"]


def test_looking_down_not_counted_as_closure():
    closed = set(range(30, 45))                       # seg1에서 3초간 "감김"처럼 보임
    hp = [(15.0, 0.0, 0.0) if t in closed else (0.0, 0.0, 0.0) for t in range(75)]
    res = _run(closed, head_pose=hp)                  # 그 3초 동안 15° 숙임 → 아래 응시
    assert [r["raw"]["sleepy_flag"] for r in res] == [0, 0, 0]
    assert all(r["raw"]["perclos_ratio"] == 0.0 for r in res)


def _seg(perclos=0.2, head_pose=1.0, status="present", final=0.7, sleepy=0):
    nan, p = float("nan"), status == "present"
    return {
        "scores": {"final": final if p else nan, "head_pose": head_pose if p else nan,
                   "head_angle": 0.5, "shoulder_tilt": 0.7, "hand_face": 1.0},
        "raw": {"perclos_ratio": perclos, "blink_rate_proxy": 0.07, "sleepy_flag": sleepy},
        "flags": {"status": status},
    }


def test_warmup_uses_final():
    out, _ = apply_personalization([_seg() for _ in range(5)])
    assert all(o["scores"]["personalized"] == 0.7 for o in out)


def test_only_worse_direction_penalized():
    segs = [_seg() for _ in range(20)] + [_seg(perclos=0.1), _seg(perclos=0.2), _seg(perclos=0.35)]
    out, summary = apply_personalization(segs)
    st = [o["scores"]["personal_stability"] for o in out[-3:]]
    assert st[0] == pytest.approx(1.0) and st[1] == pytest.approx(1.0)   # 좋아지거나 같으면 감점 없음
    assert st[2] < 0.95                                                   # 평소보다 0.15 악화
    assert summary["features"]["perclos"]["median"] == 0.2


def test_absent_is_nan_and_excluded_from_reference():
    segs = [_seg() for _ in range(15)] + [_seg(status="absent")] * 5 + [_seg()]
    out, _ = apply_personalization(segs)
    assert math.isnan(out[16]["scores"]["personalized"])
    assert out[-1]["scores"]["personal_stability"] == pytest.approx(1.0)

def test_single_strong_deviation_not_diluted():
    base = [_seg() for _ in range(20)]
    hand_drop = _seg()
    hand_drop["scores"]["hand_face"] = 0.2          # 다른 피처는 모두 평소와 같음
    out, _ = apply_personalization(base + [hand_drop])
    assert out[-1]["scores"]["personal_stability"] < 0.6



def _bs(in_l=0.0, out_l=0.0, in_r=0.0, out_r=0.0):
    return {"eyeLookInLeft": in_l, "eyeLookOutLeft": out_l,
            "eyeLookInRight": in_r, "eyeLookOutRight": out_r}


def test_gaze_h_sign_and_missing():
    g = gaze_h_series([_bs(out_l=0.6, in_r=0.6), _bs(in_l=0.6, out_r=0.6), None])
    assert g[0] > 0 > g[1]
    assert np.isnan(g[2])


def test_rolling_past_median_excludes_current():
    x = np.array([0.0] * 10 + [1.0])
    r = rolling_past_median(x, window=10, min_periods=5)
    assert r[-1] == 0.0          # 현재 샘플(1.0)은 기준에 들어가지 않음
    assert np.isnan(r[3])        # min_periods 미만


def test_yawn_across_segment_boundary_counted_once():
    # 세그먼트(25샘플) 경계 20~30에 걸친 2초 하품
    jaw = np.array([0.0] * 20 + [0.5] * 10 + [0.0] * 20)
    onset = detect_yawn_onsets(jaw, 0.25, 5, 0.45)
    assert onset.sum() == 1
    assert onset[24]                        # 연속 5샘플째(첫 세그먼트 마지막 샘플)에서 확정
    assert onset[25:].sum() == 0            # 다음 세그먼트에서 다시 세지 않음


def test_speaking_short_openings_not_yawn():
    jaw = np.array([0.5, 0.5, 0.0, 0.0] * 10)   # 0.4초씩 반복해서 열림
    assert detect_yawn_onsets(jaw, 0.25, 5, 0.45).sum() == 0


def test_face_missing_breaks_run():
    jaw = np.array([0.5] * 3 + [NAN] + [0.5] * 3)
    assert detect_yawn_onsets(jaw, 0.25, 5, 0.45).sum() == 0


def test_long_yawn_counted_once():
    jaw = np.array([0.5] * 30)                   # 6초 동안 계속
    assert detect_yawn_onsets(jaw, 0.25, 5, 0.45).sum() == 1

def test_long_but_shallow_opening_not_yawn():
    jaw = np.array([0.35] * 8)                   # 말하기처럼 1.6초 열렸지만 얕음
    assert detect_yawn_onsets(jaw, 0.25, 5, 0.45).sum() == 0


def test_onset_at_peak_when_peak_comes_late():
    jaw = np.array([0.3] * 6 + [0.6] + [0.3] * 3)   # 길이 조건은 먼저, peak는 나중에 충족
    onset = detect_yawn_onsets(jaw, 0.25, 5, 0.45)
    assert onset.sum() == 1 and onset[6]