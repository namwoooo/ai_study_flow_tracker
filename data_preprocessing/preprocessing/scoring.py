"""프레임별 랜드마크 → 5초 세그먼트별 7-feature 점수"""

import copy

import numpy as np

from .config import (
    CLOSURE_LONG_SEC,
    DEFAULT_SCORING_PARAMS,
    EAR_BASELINE_DEFAULT,
    EAR_BASELINE_WARMUP_FRAMES,
    EYE_CLOSURE_RATIO,
    HEAD_NOD_THRESH,
    PHONE_USE_THRESH,
    REF_RADIUS_DEFAULT,
    REF_RADIUS_MIN,
    REF_WINDOW_MIN,
    SCORING_FPS,
)
from .features import (
    POSE_IDX,
    angle_from_horizontal,
    center_of_bbox,
    compute_EAR_baseline,
    eye_aspect_ratio_6pts,
    get_pose_xy,
    is_eye_closure,
    to_score,
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
        debug=False,
):
    T = len(processed_frames)
    num_segments = T // segment_size
    results = []

    P = _resolve_params(params)
    W = P["w"]

    if ref_center is None or ref_radius is None:
        ref_center, ref_radius = _auto_reference(face_bboxes, segment_size)

    # EAR baseline (첫 세그먼트 끝에서 계산, 이후 세그먼트에서 재사용)
    baseline_buffer = []
    ear_baseline_global = EAR_BASELINE_DEFAULT

    for seg_idx in range(num_segments):
        s = seg_idx * segment_size
        e = s + segment_size
        EAR_baseline = ear_baseline_global

        phone_use_flags = []
        head_angles = []
        shoulder_tilts = []
        hand_face_scores = []
        seat_flags = []
        ear_series = []
        blink_transitions = 0
        prev_closed = None

        closure_frames = 0
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

                if fc is not None:
                    wL = get_pose_xy(p, POSE_IDX["wrist_l"], w, h)
                    wR = get_pose_xy(p, POSE_IDX["wrist_r"], w, h)
                    ws = [x for x in (wL, wR) if x is not None]
                    if ws:
                        hand_center = np.mean(np.array(ws), axis=0)
                        d_hf = np.linalg.norm(hand_center - fc)
                        hand_face_scores.append(
                            to_score(d_hf, *P["norm"]["hand_face_dist"], small_is_good=False)
                        )

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
            if eyes_landmarks[t] is not None:
                try:
                    eyeL, eyeR = eyes_landmarks[t]
                    earL = eye_aspect_ratio_6pts(eyeL)
                    earR = eye_aspect_ratio_6pts(eyeR)
                    ear = 0.5 * (earL + earR)
                    ear_series.append(ear)

                    if seg_idx == 0 and t < EAR_BASELINE_WARMUP_FRAMES:
                        baseline_buffer.append(ear)

                    if seg_idx == 0 and t == e - 1:
                        EAR_baseline = compute_EAR_baseline(baseline_buffer)
                        ear_baseline_global = EAR_baseline
                    else:
                        EAR_baseline = ear_baseline_global

                    # EAR이 baseline의 55% 미만이면 감김
                    closed = is_eye_closure(ear, EAR_baseline, ratio=EYE_CLOSURE_RATIO)
                    if debug and seg_idx < 3:
                        print(
                            f"[seg {seg_idx}, t={t}] ear={ear:.4f}, baseline={EAR_baseline:.4f}, "
                            f"threshold={EAR_baseline * EYE_CLOSURE_RATIO:.4f}, closed={closed}"
                        )

                    # 감김 → 뜸 전환 = 깜빡임 1회
                    if prev_closed is not None and prev_closed and not closed:
                        blink_transitions += 1
                    prev_closed = closed

                    if closed:
                        closure_frames += 1
                    else:
                        closure_frames = 0

                    closure_long = closure_frames >= int(CLOSURE_LONG_SEC * SCORING_FPS)

                    head_nod = False
                    if len(head_angles) >= 3:
                        head_nod = head_angles[-1] - head_angles[-3] > HEAD_NOD_THRESH

                    sleepy_flags.append(closure_long or head_nod)

                except Exception:
                    ear_series.append(np.nan)

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

        blink_rate_proxy = blink_transitions / max(1, (e - s))
        blink_score = to_score(blink_rate_proxy, *P["norm"]["blink_rate_proxy"], small_is_good=True)

        segment_sleepy = any(sleepy_flags)
        perclos_score = 1.0 - (1.0 if segment_sleepy else 0.0)

        final_score = (
                W["phone_use"] * phone_use_score
                + W["seat_depart"] * seat_depart_score
                + W["head_angle"] * head_angle_score
                + W["shoulder_tilt"] * shoulder_tilt_score
                + W["hand_face"] * hand_face_score
                + W["blink"] * blink_score
                + W["perclos"] * perclos_score
        )

        results.append({
            "segment_idx": seg_idx,
            "scores": {
                "phone_use": round(phone_use_score, 4),
                "seat_depart": round(seat_depart_score, 4),
                "head_angle": round(head_angle_score, 4),
                "shoulder_tilt": round(shoulder_tilt_score, 4),
                "hand_face": round(hand_face_score, 4),
                "blink": round(blink_score, 4),
                "perclos": round(perclos_score, 4),
                "final": round(final_score, 4),
            },
            "raw": {
                "usage_ratio": round(usage_ratio, 4),
                "sleepy_flag": int(segment_sleepy),
                "blink_rate_proxy": round(blink_rate_proxy, 5),
                "EAR_baseline": round(EAR_baseline, 4),
            },
            "flags": {
                "seat_departed": seat_departed,
            },
        })

    return results