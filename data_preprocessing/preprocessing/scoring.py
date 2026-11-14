"""프레임별 랜드마크 → 5초 세그먼트별 피처 점수"""

import copy

import numpy as np

from .config import (
    ABSENT_FACE_MISSING_RATIO,
    BLINK_MIN_SAMPLES,
    BODY_MIN_WIDTH_RATIO,
    BODY_VISIBLE_RATIO,
    CLOSURE_LONG_SEC,
    DEFAULT_SCORING_PARAMS,
    EYE_CLOSURE_RATIO,
    EYE_GATE_DOWN_DEG,
    HAND_ABSENT_SCORE,
    HEAD_REF_WINDOW_SEC,
    PERCLOS_MIN_SAMPLES,
    PERCLOS_WINDOW_SEC,
    PHONE_USE_THRESH,
    REF_RADIUS_DEFAULT,
    REF_RADIUS_MIN,
    REF_WINDOW_MIN,
    TARGET_FPS,
    YAWN_MIN_SEC,
    YAWN_JAW_THRESH,
    YAWN_PEAK_THRESH,
    NOD_DROP_SEC,
    NOD_DROP_DEG,
    NOD_RECOVER_SEC,
    NOD_RECOVER_DEG,
)
from .features import (
    POSE_IDX,
    angle_from_horizontal,
    center_of_bbox,
    compute_ear_series,
    compute_EAR_baseline,
    get_pose_xy,
    shoulder_width,
    to_score,
    weighted_mean,
    gaze_h_series,
    rolling_past_median,
    blendshape_series,
    detect_yawn_onsets,
    detect_nod_events,
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


def _nanmedian(a):
    """전부 NaN이면 경고 없이 NaN."""
    a = np.asarray(a, dtype=float)
    if not np.isfinite(a).any():
        return float("nan")
    return float(np.nanmedian(a))


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
        blendshapes=None,
        debug=False,
):
    T = len(processed_frames)
    num_segments = T // segment_size
    results = []

    P = _resolve_params(params)
    W = P["w"]

    if ref_center is None or ref_radius is None:
        ref_center, ref_radius = _auto_reference(face_bboxes, segment_size)

    # ------------------------------------------------------------------
    # 루프 전 준비 (모두 과거 데이터만 사용 → 실시간 계산 가능)
    # ------------------------------------------------------------------

    # --- head pose (pitch −: 위, +: 아래 / yaw −: 영상 기준 왼쪽) ---
    if head_pose is not None:
        hp = np.array([x if x is not None else (np.nan,) * 3 for x in head_pose], dtype=float)
        pitch_s, yaw_s = hp[:, 0], hp[:, 1]
    else:
        pitch_s = yaw_s = np.full(T, np.nan)
    head_ref_window = max(1, int(round(HEAD_REF_WINDOW_SEC * fps)))

    # 세그먼트별 기준 자세: 직전 10분 중앙값 (첫 세그먼트는 자기 자신)
    seg_ref = []
    for seg_idx in range(num_segments):
        s = seg_idx * segment_size
        r0, r1 = max(0, s - head_ref_window), (s if s > 0 else s + segment_size)
        seg_ref.append((r0, r1, _nanmedian(pitch_s[r0:r1]), _nanmedian(yaw_s[r0:r1])))

    # 숙인 샘플: 눈꺼풀이 눈을 덮어 감은 눈과 구분 불가 → 눈 판정 제외
    gaze_down = np.zeros(T, dtype=bool)
    for seg_idx, (_, _, ref_p, _) in enumerate(seg_ref):
        s = seg_idx * segment_size
        if np.isfinite(ref_p):
            with np.errstate(invalid="ignore"):
                gaze_down[s:s + segment_size] = (pitch_s[s:s + segment_size] - ref_p) > EYE_GATE_DOWN_DEG

    # --- EAR: 숙이지 않은 샘플만으로 baseline과 감김 판정 ---
    ears = compute_ear_series(eyes_landmarks)
    eye_eval = np.isfinite(ears) & ~gaze_down
    ear_baseline = float(compute_EAR_baseline(np.where(eye_eval, ears, np.nan)))
    closed_series = np.zeros(T, dtype=bool)
    closed_series[eye_eval] = ears[eye_eval] < ear_baseline * EYE_CLOSURE_RATIO
    perclos_window = max(1, int(round(PERCLOS_WINDOW_SEC * fps)))

    closure_long_frames = max(1, int(round(CLOSURE_LONG_SEC * fps)))
    closure_frames = 0  # 세그먼트 경계를 넘어 누적
    prev_closed = None  # 세그먼트 경계를 넘어 유지

    # --- 몸 크기 기준: 얼굴이 보인 샘플의 어깨 폭 (멀리 있는 사람·물체와 책상 앞 사람 구분용) ---
    sw_s = np.array([shoulder_width(p) for p in pose_keypoints], dtype=float)
    face_ok = np.array([fs is not None for fs in frame_sizes])

    # 수평 시선: 직전 10분 중앙값 대비 이탈 (head pose 기준 자세와 같은 창)
    if blendshapes is not None:
        gaze_h = gaze_h_series(blendshapes)
        gaze_ref = rolling_past_median(gaze_h, window=int(HEAD_REF_WINDOW_SEC * fps), min_periods=int(30 * fps))
        gaze_dev = np.abs(gaze_h - gaze_ref)
    else:
        gaze_dev = np.full(len(frame_sizes), np.nan)

    # 하품: jawOpen이 YAWN_MIN_SEC 이상 이어진 순간 (말하기의 짧은 열림과 구분)
    if blendshapes is not None:
        jaw_s = blendshape_series(blendshapes, "jawOpen")
        yawn_onset = detect_yawn_onsets(
            jaw_s, YAWN_JAW_THRESH, max(1, int(round(YAWN_MIN_SEC * fps))), YAWN_PEAK_THRESH,
        )
    else:
        yawn_onset = np.zeros(T, dtype=bool)

    # 고개 떨굼(pitch): 기록만 하고 sleepy에는 넣지 않음 (졸음 영상으로 검증 전)
    nod_onset = detect_nod_events(
        pitch_s, NOD_DROP_DEG, max(1, int(round(NOD_DROP_SEC * fps))),
        NOD_RECOVER_DEG, max(1, int(round(NOD_RECOVER_SEC * fps))),
    )

    # ------------------------------------------------------------------
    # 세그먼트 루프
    # ------------------------------------------------------------------
    for seg_idx in range(num_segments):
        s = seg_idx * segment_size
        e = s + segment_size
        r0, r1, ref_p, ref_y = seg_ref[seg_idx]

        phone_use_flags = []
        head_angles = []
        shoulder_tilts = []
        hand_face_scores = []
        hand_visible_flags = []
        blink_transitions = 0
        sleepy_flags = []
        blink_transitions = 0
        closure_count = 0       # 장시간 감김 횟수 (기록용, 졸음 판정에는 미반영)

        for t in range(s, e):
            if frame_sizes[t] is None:
                closure_frames = 0      # 얼굴 미검출: 감김 연속과 깜빡임 직전 상태를 끊음
                prev_closed = None
                continue
            w, h = frame_sizes[t]
            fb = face_bboxes[t]
            fc = center_of_bbox(fb) if fb is not None else None

            # 1) 폰 사용 (TODO: 폰 검출 미구현, phone_bboxes는 항상 None)
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

            # 2) 고개 기울기 (roll, 눈 두 점 기준)
            if eyes_landmarks[t] is not None:
                eyeL, eyeR = eyes_landmarks[t]
                head_angles.append(angle_from_horizontal(eyeL.mean(axis=0), eyeR.mean(axis=0)))

            # 3) 어깨 기울기 + 손-얼굴 거리
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

                    # 4) EAR 기반 깜빡임 / 장시간 감김 (판정 가능한 샘플만)
                    if eye_eval[t]:
                        ear = ears[t]
                        closed = bool(closed_series[t])
                        if debug and seg_idx < 3:
                            print(
                                f"[seg {seg_idx}, t={t}] ear={ear:.4f}, baseline={ear_baseline:.4f}, "
                                f"threshold={ear_baseline * EYE_CLOSURE_RATIO:.4f}, closed={closed}"
                            )

                        if prev_closed is not None and prev_closed and not closed:
                            blink_transitions += 1
                        prev_closed = closed

                        closure_frames = closure_frames + 1 if closed else 0
                        if closure_frames == closure_long_frames:   # 연속 감김이 기준 길이에 도달한 순간 1회
                            closure_count += 1
                    else:
                        # 판정 불가(숙임·눈 미검출): 눈 상태를 모르므로 연속을 끊음
                        # (끊지 않으면 숙임 사이의 짧은 감김들이 이어 붙어 장시간 감김이 됨)
                        prev_closed = None
                        closure_frames = 0

        # -------------------------------
        # 세그먼트 점수
        # -------------------------------
        usage_ratio = sum(phone_use_flags) / len(phone_use_flags) if phone_use_flags else 0.0
        phone_use_score = 1.0 - usage_ratio

        seat_depart_score = 1.0  # 자리 비움은 status로 처리 (가중치 0, 스키마 유지용)

        if head_angles:
            head_angle_score = to_score(np.mean(head_angles), *P["norm"]["head_angle_rad"], small_is_good=True)
        else:
            head_angle_score = 0.5

        if shoulder_tilts:
            shoulder_tilt_score = to_score(np.mean(shoulder_tilts), *P["norm"]["shoulder_tilt"], small_is_good=True)
        else:
            shoulder_tilt_score = 0.5

        hand_face_score = float(np.mean(hand_face_scores)) if hand_face_scores else 0.5
        hand_visible_ratio = float(np.mean(hand_visible_flags)) if hand_visible_flags else 0.0

        # 깜빡임: 판정 가능한 샘플 수 기준 비율
        eye_eval_count = int(eye_eval[s:e].sum())
        if eye_eval_count >= BLINK_MIN_SAMPLES:
            blink_rate_proxy = blink_transitions / eye_eval_count
            blink_score = to_score(blink_rate_proxy, *P["norm"]["blink_rate_proxy"], small_is_good=True)
        else:
            blink_rate_proxy = float("nan")
            blink_score = 0.5

        yawn_count = int(yawn_onset[s:e].sum())
        nod_count = int(nod_onset[s:e].sum())
        # 졸음 신호: 검증된 하품만. 장시간 감김·고개 떨굼은 정면 구도에서 책 보기와
        # 구분되지 않아 기록만 함 (실제 졸음에 따른 감김 증가는 개인화 PERCLOS가 반영)
        segment_sleepy = yawn_count > 0

        # head pose: 기준 자세(직전 10분 중앙값) 대비 위·좌우 이탈만 감점, 아래(필기)는 제외
        if np.isfinite(ref_p):
            dp, dy = pitch_s[s:e] - ref_p, yaw_s[s:e] - ref_y
        else:
            dp = dy = np.full(e - s, np.nan)
        if np.isfinite(dp).any():
            yaw_dev = _nanmedian(np.abs(dy))
            up_dev = _nanmedian(np.clip(-dp, 0, None))
            down_dev = _nanmedian(np.clip(dp, 0, None))
            head_pose_score = min(
                to_score(yaw_dev, *P["norm"]["head_yaw_deg"], small_is_good=True),
                to_score(up_dev, *P["norm"]["head_up_deg"], small_is_good=True),
            )
        else:
            yaw_dev = up_dev = down_dev = float("nan")
            head_pose_score = 0.5

        # PERCLOS: 세그먼트 끝 기준 직전 창에서, 판정 가능한 샘플 중 감김 비율
        w_start = max(0, e - perclos_window)
        win_valid = eye_eval[w_start:e]
        if win_valid.sum() >= PERCLOS_MIN_SAMPLES:
            perclos_ratio = float(closed_series[w_start:e][win_valid].mean())
            perclos_score = to_score(perclos_ratio, *P["norm"]["perclos"], small_is_good=True)
        else:
            perclos_ratio = float("nan")
            perclos_score = 0.5

        # 시선 이탈 비율: 세그먼트 중 다른 곳을 본 샘플 비율
        gd = gaze_dev[s:e]
        gd = gd[np.isfinite(gd)]
        if len(gd) >= 5:
            gaze_away_ratio = float((gd > P["gaze_dev_thresh"]).mean())
            gaze_score = to_score(gaze_away_ratio, *P["norm"]["gaze_away_ratio"], small_is_good=True)
        else:
            gaze_away_ratio, gaze_score = float("nan"), 0.5   # 샘플 부족 시 중립 (기존 규칙과 동일)

        feature_scores = {
            "phone_use": phone_use_score,
            "seat_depart": seat_depart_score,
            "head_angle": head_angle_score,
            "shoulder_tilt": shoulder_tilt_score,
            "hand_face": hand_face_score,
            "blink": blink_score,
            "perclos": perclos_score,
            "head_pose": head_pose_score,
            "gaze": gaze_score,
        }

        # 상태 판정: 얼굴이 대부분 없으면, 책상 앞 몸(양 어깨, 평소 폭)이 보이는지로 구분
        face_missing_ratio = sum(frame_sizes[t] is None for t in range(s, e)) / (e - s)
        ref_win = sw_s[r0:r1][face_ok[r0:r1]]
        ref_sw = _nanmedian(ref_win)
        seg_sw = sw_s[s:e]
        if np.isfinite(ref_sw):
            with np.errstate(invalid="ignore"):
                body_ok = np.isfinite(seg_sw) & (seg_sw >= BODY_MIN_WIDTH_RATIO * ref_sw)
        else:
            body_ok = np.isfinite(seg_sw)  # 기준이 없으면 보이기만 하면 인정
        body_visible_ratio = float(body_ok.mean())
        seg_sw_med = _nanmedian(seg_sw)
        body_width_rel = seg_sw_med / ref_sw if np.isfinite(seg_sw_med) and np.isfinite(ref_sw) else float("nan")

        # 샘플 단위 재석 여부: 얼굴이 보이거나, 책상 앞 몸이 보이면 자리에 있음
        face_ok_seg = np.array([frame_sizes[t] is not None for t in range(s, e)])
        present_ratio = float((face_ok_seg | body_ok).mean())
        feature_scores["seat_depart"] = present_ratio   # 세그먼트 일부만 비워도 그만큼 감점

        if face_missing_ratio < ABSENT_FACE_MISSING_RATIO:
            status = "present"
        elif body_visible_ratio >= BODY_VISIBLE_RATIO:
            status = "face_hidden"  # 자리에 있지만 얼굴이 가려짐 (고개 숙임, 손으로 가림)
        else:
            status = "absent"

        if status != "present":
            # 얼굴 피처가 없으므로 점수를 비워 둠 (평균에서 자동 제외, 리포트에서 따로 집계)
            feature_scores = {k: float("nan") for k in feature_scores}
            final_score = float("nan")
        else:
            # 세그먼트 일부를 비웠으면 그 비율만큼 감점 (계속 앉아 있으면 1.0이라 영향 없음)
            final_score = weighted_mean(feature_scores, W) * present_ratio

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
                "eye_eval_ratio": round(eye_eval_count / (e - s), 4),
                "hand_visible_ratio": round(hand_visible_ratio, 4),
                "head_yaw_dev": round(yaw_dev, 2),
                "head_up_dev": round(up_dev, 2),
                "head_down_dev": round(down_dev, 2),
                "face_missing_ratio": round(face_missing_ratio, 4),
                "body_visible_ratio": round(body_visible_ratio, 4),
                "body_width_rel": round(body_width_rel, 3),
                "present_ratio": round(present_ratio, 4),
                "gaze_away_ratio": round(gaze_away_ratio, 4),
                "yawn_count": yawn_count,
                "closure_count": int(closure_count),
                "nod_count": int(nod_count),
            },
            "flags": {
                "seat_departed": status == "absent",
                "status": status,
            },
        })

    return results