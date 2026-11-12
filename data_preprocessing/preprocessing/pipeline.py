"""영상 1개 → 세그먼트별 집중도 DataFrame"""

import time

import pandas as pd

from .config import FACE_MODEL_PATH, POSE_MODEL_PATH, SEGMENT_SEC, TARGET_FPS
from .personalization import apply_personalization
from .extraction import extract_landmarks
from .scoring import compute_segment_scores_7features

CSV_COLUMNS = [
    "segment_idx",
    "phone_use", "seat_depart", "head_angle", "shoulder_tilt", "hand_face", "hand_visible_ratio", "blink", "perclos",
    "final", "personalized", "personal_stability",
    "usage_ratio", "sleepy_flag", "blink_rate_proxy", "perclos_ratio", "EAR_baseline",
    "seat_departed", "head_pose", "head_yaw_dev", "head_up_dev", "head_down_dev", "status", "face_missing_ratio",
    "body_visible_ratio", "body_width_rel", "eye_eval_ratio"
]

def score_landmarks(lm, target_fps=TARGET_FPS, segment_sec=SEGMENT_SEC, phone_bboxes=None, debug=False):
    """추출 결과(lm) → (df, baseline, seg_results) / 추출 없이 점수만 다시 계산할 때 사용"""
    seg_results = compute_segment_scores_7features(
        processed_frames=lm["processed_frames"], pose_keypoints=lm["pose_keypoints"],
        eyes_landmarks=lm["eyes_landmarks"], face_bboxes=lm["face_bboxes"],
        frame_sizes=lm["frame_sizes"], segment_size=target_fps * segment_sec,
        head_pose=lm.get("head_pose"),
        phone_bboxes=phone_bboxes, fps=target_fps, debug=debug,
    )   # 기존 그대로
    personalized, baseline = apply_personalization(seg_results, segment_sec)
    df = segments_to_dataframe(personalized)
    return df, baseline, seg_results

def segments_to_dataframe(personalized_segments):
    """개인화된 세그먼트 결과 → 라벨 CSV와 동일한 구조의 DataFrame."""
    rows = []
    for r in personalized_segments:
        s, raw, flg = r["scores"], r["raw"], r["flags"]
        rows.append({
            "segment_idx": r["segment_idx"],
            "phone_use": s["phone_use"],
            "seat_depart": s["seat_depart"],
            "head_angle": s["head_angle"],
            "shoulder_tilt": s["shoulder_tilt"],
            "hand_face": s["hand_face"],
            "hand_visible_ratio": raw["hand_visible_ratio"],
            "blink": s["blink"],
            "perclos": s["perclos"],
            "perclos_ratio": raw["perclos_ratio"],
            "final": s["final"],
            "personalized": s["personalized"],
            "personal_stability": s["personal_stability"],
            "usage_ratio": raw["usage_ratio"],
            "sleepy_flag": raw["sleepy_flag"],
            "blink_rate_proxy": raw["blink_rate_proxy"],
            "EAR_baseline": raw["EAR_baseline"],
            "seat_departed": flg["seat_departed"],
            "head_pose": s["head_pose"],
            "head_yaw_dev": raw["head_yaw_dev"],
            "head_up_dev": raw["head_up_dev"],
            "head_down_dev": raw["head_down_dev"],
            "status": flg["status"],
            "face_missing_ratio": raw["face_missing_ratio"],
            "body_visible_ratio": raw["body_visible_ratio"],
            "body_width_rel": raw["body_width_rel"],
            "eye_eval_ratio": raw["eye_eval_ratio"],
        })
    return pd.DataFrame(rows, columns=CSV_COLUMNS).round(3)


def _episodes(mask):
    """연속된 True 구간의 길이 목록"""
    out, run = [], 0
    for m in list(mask) + [False]:
        if m:
            run += 1
        elif run:
            out.append(run)
            run = 0
    return out


def summarize_session(df, segment_sec=SEGMENT_SEC):
    """LLM 리포트용 세션 요약. 집중도는 얼굴이 보인(present) 세그먼트 기준"""
    to_min = lambda n: round(float(n * segment_sec / 60), 1)
    absent = (df["status"] == "absent").to_numpy()
    hidden = (df["status"] == "face_hidden").to_numpy()
    absent_ep, hidden_ep = _episodes(absent), _episodes(hidden)
    present = df[df["status"] == "present"]
    return {
        "total_min": to_min(len(df)),
        "present_min": to_min(len(present)),
        "absent_min": to_min(absent.sum()),
        "absence_count": int(len(absent_ep)),
        "longest_absence_min": to_min(max(absent_ep, default=0)),
        "face_hidden_min": to_min(hidden.sum()),
        "face_hidden_count": int(len(hidden_ep)),
        "mean_final": round(float(present["final"].mean()), 3),
        "mean_personalized": round(float(present["personalized"].mean()), 3),
    }


def run(
        video_path,
        target_fps=TARGET_FPS,
        segment_sec=SEGMENT_SEC,
        face_model_path=FACE_MODEL_PATH,
        pose_model_path=POSE_MODEL_PATH,
        phone_bboxes=None,
        debug=False,
):
    """
    Returns:
        df (pd.DataFrame): 세그먼트별 점수
        baseline (dict): 개인화 baseline
        info (dict): num_frames, num_segments, native_fps, timing{extract, face, pose, skip, scoring, total}
    """
    t_start = time.time()

    # 1) 랜드마크 추출
    lm = extract_landmarks(video_path, target_fps, face_model_path, pose_model_path)

    # 2) 세그먼트 점수
    t1 = time.time()
    df, baseline, seg_results = score_landmarks(
        lm, target_fps=target_fps, segment_sec=segment_sec,phone_bboxes=phone_bboxes, debug=debug,
    )
    t_scoring = time.time() - t1

    info = {
        "num_frames": lm["num_frames"],
        "num_segments": len(seg_results),
        "native_fps": lm["native_fps"],
        "timing": {
            "extract": lm["timing"]["total"],
            "face": lm["timing"]["face"],
            "pose": lm["timing"]["pose"],
            "skip": lm["timing"]["skip"],
            "scoring": t_scoring,
            "total": time.time() - t_start,
        },
    }
    return df, baseline, info