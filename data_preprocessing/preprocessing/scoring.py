"""프레임별 랜드마크 → 5초 세그먼트별 7-feature 점수"""

import copy

import numpy as np

from .config import (
    CLOSURE_LONG_SEC,
    DEFAULT_SCORING_PARAMS,
    EYE_CLOSURE_RATIO,
    HEAD_NOD_THRESH,
    PHONE_USE_THRESH,
    REF_RADIUS_DEFAULT,
    REF_RADIUS_MIN,
    REF_WINDOW_MIN,
    TARGET_FPS, PERCLOS_WINDOW_SEC,
    HAND_ABSENT_SCORE, HEAD_REF_WINDOW_SEC,
    ABSENT_FACE_MISSING_RATIO, BODY_VISIBLE_RATIO,
    BODY_MIN_WIDTH_RATIO,
)
from .features import (
    POSE_IDX,
    angle_from_horizontal,
    center_of_bbox,
    compute_EAR_baseline,
    get_pose_xy,
    is_eye_closure,
    to_score,
    compute_ear_series,
    weighted_mean,
    shoulder_width,
)


def _resolve_params(params):
    """기본 파라미터에 사용자 파라미터를 병합. config 원본은 건드리지 않는다."""
    P = copy.deepcopy(DEFAULT_SCORING_PARAMS)
    for k, v in (params or {}).items():
        if k in P and isinstance(v, dict):
            P[k].update(v)
        else:
            P[k] = v
    return P


def _auto_reference(face_bboxes, segment_size):
    """초기 구간 얼굴 bbox로 자리 이탈 판정용 기준점/반경을 설정."""
    window = face_bboxes[:min(len(face_bboxes), max(segment_size, REF_WINDOW_MIN))]
    valid = [b for b in window if b is not None]
    if len(valid) >= 3:
        fcs = [center_of_bbox(b) for b in valid]
        ref_center = tuple(np.median(np.stack(fcs, axis=0), axis=0))
        hs = [(b[3] - b[1]) for b in valid]
        ref_radius = float(max(REF_RADIUS_MIN, 2 * np.median(hs))) if hs else REF_RADIUS_DEFAULT
        return ref_center, ref_radius
    return (0, 0), REF_RADIUS_DEFAULT


def compute_segment_scores_7features(
        processed_frames,
        pose_keypoints,
        eyes_landmarks,
        face_bboxes,
        frame_sizes,
        segment_size,
        phone_bboxes=None,
        ref_center=None,
        ref_radius=None,
        params=None,
        fps=TARGET_FPS,
        head_pose=None,
        debug=False,
):
    T = len(processed_frames)
    num_segments = T // segment_size
    results = []

    P = _resolve_params(params)
    W = P["w"]

    if ref_center is None or ref_radius is None:
        ref_center, ref_radius = _auto_reference(face_bboxes, segment_size)

    # EAR baseline (영상 앞부분의 유효 EAR 30개의 중앙값)
    ears = compute_ear_series(eyes_landmarks)
    ear_baseline = float(compute_EAR_baseline(ears))

    # 프레임별 감김 여부
    ear_valid = np.isfinite(ears)
    closed_series = np.zeros(len(ears), dtype=bool)
    closed_series[ear_valid] = ears[ear_valid] < ear_baseline * EYE_CLOSURE_RATIO
    perclos_window = max(1, int(round(PERCLOS_WINDOW_SEC * fps)))

    closure_long_frames = max(1, int(round(CLOSURE_LONG_SEC * fps)))
    closure_frames = 0  # 세그먼트 경계를 넘어 누적
    prev_closed = None

    # head pose (pitch −: 위, yaw −: 영상 기준 왼쪽)
    if head_pose is not None:
        hp = np.array([x if x is not None else (np.nan,) * 3 for x in head_pose], dtype=float)
        pitch_s, yaw_s = hp[:, 0], hp[:, 1]
    else:
        pitch_s = yaw_s = np.full(T, np.nan)
    head_ref_window = max(1, int(round(HEAD_REF_WINDOW_SEC * fps)))

    # 몸 크기 기준: 얼굴이 보인 샘플의 어깨 폭 (멀리 있는 사람·물체와 책상 앞 사람 구분용)
    sw_s = np.array([shoulder_width(p) for p in pose_keypoints])
    face_ok = np.array([fs is not None for fs in frame_sizes])

    for seg_idx in range(num_segments):
        s = seg_idx * segment_size
        e = s + segment_size

        phone_use_flags = []
        head_angles = []
        shoulder_tilts = []
        hand_face_scores = []
        hand_visible_flags = []
        seat_flags = []
        blink_transitions = 0
        sleepy_flags = []

        for t in range(s, e):
            if frame_sizes[t] is None:
                continue
            w, h = frame_sizes[t]
            fb = face_bboxes[t]
            fc = center_of_bbox(fb) if fb is not None else None

            # ---------------------
            # 1) 폰 사용 (손-폰 거리)
            # ---------------------
            if phone_bboxes is not None and phone_bboxes[t] is not None:
                p = pose_keypoints[t]
                hand_center = None
                if p is not None:
                    wL = get_pose_xy(p, POSE_IDX["wrist_l"], w, h)
                    wR = get_pose_xy(p, POSE_IDX["wrist_r"], w, h)
                    ws = [x for x in (wL, wR) if x is not None]
                    if ws:
                        hand_center = np.mean(np.array(ws), axis=0)

                if hand_center is not None:
                    d_hp = np.linalg.norm(center_of_bbox(phone_bboxes[t]) - hand_center)
                    phone_use_flags.append(d_hp < PHONE_USE_THRESH)
                else:
                    phone_use_flags.append(False)
            else:
                phone_use_flags.append(False)

            # ---------------------
            # 2) 고개 각도
            # ---------------------
            if eyes_landmarks[t] is not None:
                eyeL, eyeR = eyes_landmarks[t]
                cenL = eyeL.mean(axis=0)
                cenR = eyeR.mean(axis=0)
                head_angles.append(angle_from_horizontal(cenL, cenR))

            # ---------------------
            # 3) 어깨 기울기 + 손-얼굴 거리 + 자리 이탈
            # ---------------------
            p = pose_keypoints[t]
            if p is not None:
                sL = get_pose_xy(p, POSE_IDX["shoulder_l"], w, h)
                sR = get_pose_xy(p, POSE_IDX["shoulder_r"], w, h)
                if sL and sR:
                    shoulder_tilts.append(angle_from_horizontal(sL, sR))

                if fb is not None:
                    wL = get_pose_xy(p, POSE_IDX["wrist_l"], w, h)
                    wR = get_pose_xy(p, POSE_IDX["wrist_r"], w, h)
                    ws = [x for x in (wL, wR) if x is not None]
                    hand_visible_flags.append(bool(ws))
                    if ws:
                        face_h = max(1.0, fb[3] - fb[1])
                        d_norm = min(np.linalg.norm(np.array(x) - fc) for x in ws) / face_h
                        hand_face_scores.append(
                            to_score(d_norm, *P["norm"]["hand_face_dist"], small_is_good=False)
                        )
                    else:
                        hand_face_scores.append(HAND_ABSENT_SCORE)

                nose = get_pose_xy(p, POSE_IDX["nose"], w, h)
                if nose is None:
                    hl = get_pose_xy(p, POSE_IDX["hip_l"], w, h)
                    hr = get_pose_xy(p, POSE_IDX["hip_r"], w, h)
                    nose = tuple(np.mean([hl, hr], axis=0)) if (hl and hr) else None

                if nose is not None:
                    dist = np.linalg.norm(np.array(nose) - np.array(ref_center))
                    seat_flags.append(dist > ref_radius)

            # ---------------------
            # 4) EAR 기반 깜빡임 / 졸음
            # ---------------------
            ear = ears[t]
            if np.isfinite(ear):

                # EAR이 baseline의 55% 미만이면 감김
                closed = bool(closed_series[t])
                if debug and seg_idx < 3:
                    print(
                        f"[seg {seg_idx}, t={t}] ear={ear:.4f}, baseline={ear_baseline:.4f}, "
                        f"threshold={ear_baseline * EYE_CLOSURE_RATIO:.4f}, closed={closed}"
                    )

                # 감김 -> 뜸 전환 = 깜빡임 1회
                if prev_closed is not None and prev_closed and not closed:
                    blink_transitions += 1
                prev_closed = closed

                if closed:
                    closure_frames += 1
                else:
                    closure_frames = 0
                closure_long = closure_frames >= closure_long_frames

                head_nod = False
                if len(head_angles) >= 3:
                    head_nod = head_angles[-1] - head_angles[-3] > HEAD_NOD_THRESH

                sleepy_flags.append(closure_long or head_nod)

        # -------------------------------
        # 세그먼트 점수
        # -------------------------------
        usage_ratio = sum(phone_use_flags) / len(phone_use_flags) if phone_use_flags else 0.0
        phone_use_score = 1.0 - usage_ratio

        seat_departed = any(seat_flags)
        seat_depart_score = 0.0 if seat_departed else 1.0

        if head_angles:
            ha = np.array(head_angles)
            head_angle_score = to_score(ha.mean(), *P["norm"]["head_angle_rad"], small_is_good=True)
        else:
            head_angle_score = 0.5

        if shoulder_tilts:
            st = np.array(shoulder_tilts)
            shoulder_tilt_score = to_score(st.mean(), *P["norm"]["shoulder_tilt"], small_is_good=True)
        else:
            shoulder_tilt_score = 0.5

        hand_face_score = np.mean(hand_face_scores) if hand_face_scores else 0.5
        hand_visible_ratio = float(np.mean(hand_visible_flags)) if hand_visible_flags else 0.0

        blink_rate_proxy = blink_transitions / max(1, (e - s))
        blink_score = to_score(blink_rate_proxy, *P["norm"]["blink_rate_proxy"], small_is_good=True)

        segment_sleepy = any(sleepy_flags)

        # head pose: 직전 10분(첫 세그먼트는 자기 자신) 중앙값을 기준 자세로, 위·좌우 이탈만 감점
        r0 = max(0, s - head_ref_window)
        r1 = s if s > 0 else e
        ref_p, ref_y = np.nanmedian(pitch_s[r0:r1]), np.nanmedian(yaw_s[r0:r1])
        dp, dy = pitch_s[s:e] - ref_p, yaw_s[s:e] - ref_y
        if np.isfinite(dp).any() and np.isfinite(ref_p):
            yaw_dev = float(np.nanmedian(np.abs(dy)))
            up_dev = float(np.nanmedian(np.clip(-dp, 0, None)))
            down_dev = float(np.nanmedian(np.clip(dp, 0, None)))
            head_pose_score = min(
                to_score(yaw_dev, *P["norm"]["head_yaw_deg"], small_is_good=True),
                to_score(up_dev, *P["norm"]["head_up_deg"], small_is_good=True),
            )
        else:
            yaw_dev = up_dev = down_dev = float("nan")
            head_pose_score = 0.5

        # PERCLOS: 세그먼트 끝 기준 직전 창의 감김 비율
        w_start = max(0, e - perclos_window)
        win_valid = ear_valid[w_start:e]
        if win_valid.any():
            perclos_ratio = float(closed_series[w_start:e][win_valid].mean())
            perclos_score = to_score(perclos_ratio, *P["norm"]["perclos"], small_is_good=True)
        else:
            perclos_ratio = float("nan")
            perclos_score = 0.5

        feature_scores = {
            "phone_use": phone_use_score,
            "seat_depart": seat_depart_score,
            "head_angle": head_angle_score,
            "shoulder_tilt": shoulder_tilt_score,
            "hand_face": hand_face_score,
            "blink": blink_score,
            "perclos": perclos_score,
            "head_pose": head_pose_score,
        }
        # 상태 판정: 얼굴이 대부분 없으면, 몸(양 어깨)이 보이는지로 자리 비움과 얼굴 가림을 구분
        face_missing_ratio = sum(frame_sizes[t] is None for t in range(s, e)) / (e - s)
        # 기준 어깨 폭: 직전 10분 중 얼굴이 보였던 샘플의 중앙값 (첫 세그먼트는 자기 자신)
        ref_win = sw_s[r0:r1][face_ok[r0:r1]]
        ref_sw = np.nanmedian(ref_win) if np.isfinite(ref_win).any() else np.nan
        seg_sw = sw_s[s:e]
        if np.isfinite(ref_sw):
            body_ok = np.isfinite(seg_sw) & (seg_sw >= BODY_MIN_WIDTH_RATIO * ref_sw)
        else:
            body_ok = np.isfinite(seg_sw)        # 기준이 없으면 보이기만 하면 인정
        body_visible_ratio = float(body_ok.mean())
        body_width_rel = (float(np.nanmedian(seg_sw) / ref_sw)
                          if np.isfinite(seg_sw).any() and np.isfinite(ref_sw) else float("nan"))

        if face_missing_ratio < ABSENT_FACE_MISSING_RATIO:
            status = "present"
        elif body_visible_ratio >= BODY_VISIBLE_RATIO:
            status = "face_hidden"   # 자리에 있지만 얼굴이 가려짐 (고개 숙임, 손으로 가림)
        else:
            status = "absent"

        if status != "present":
            # 얼굴 피처가 없으므로 점수를 비워 둠 (평균에서 자동 제외, 리포트에서 따로 집계)
            feature_scores = {k: float("nan") for k in feature_scores}
            final_score = float("nan")
        else:
            final_score = weighted_mean(feature_scores, W)

        results.append({
            "segment_idx": seg_idx,
            "scores": {
                **{k: round(float(v), 4) for k, v in feature_scores.items()},
                "final": round(final_score, 4),
            },
            "raw": {
                "usage_ratio": round(usage_ratio, 4),
                "sleepy_flag": int(segment_sleepy),
                "blink_rate_proxy": round(blink_rate_proxy, 5),
                "EAR_baseline": round(ear_baseline, 4),
                "perclos_ratio": round(perclos_ratio, 4),
                "hand_visible_ratio": round(hand_visible_ratio, 4),
                "head_yaw_dev": round(yaw_dev, 2),
                "head_up_dev": round(up_dev, 2),
                "head_down_dev": round(down_dev, 2),
                "face_missing_ratio": round(face_missing_ratio, 4),
                "body_visible_ratio": round(body_visible_ratio, 4),
                "body_width_rel": round(body_width_rel, 3),
            },
            "flags": {
                "seat_departed": status == "absent",
                "status": status,
            },
        })

    return results