"""영상 1개 → 세그먼트별 집중도 DataFrame"""

import time

import pandas as pd

from .config import FACE_MODEL_PATH, POSE_MODEL_PATH, SEGMENT_SEC, TARGET_FPS, WARMUP_SEGMENTS
from .extraction import extract_landmarks
from .personalization import (
    apply_personalization_with_delta,
    compute_personal_baseline_stability_filtered,
)
from .scoring import compute_segment_scores_7features

CSV_COLUMNS = [
    "segment_idx",
    "phone_use", "seat_depart", "head_angle", "shoulder_tilt", "hand_face", "blink", "perclos",
    "final", "personalized", "personal_stability",
    "usage_ratio", "sleepy_flag", "blink_rate_proxy", "EAR_baseline",
    "seat_departed",
]


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
            "blink": s["blink"],
            "perclos": s["perclos"],
            "final": s["final"],
            "personalized": s["personalized"],
            "personal_stability": s["personal_stability"],
            "usage_ratio": raw["usage_ratio"],
            "sleepy_flag": raw["sleepy_flag"],
            "blink_rate_proxy": raw["blink_rate_proxy"],
            "EAR_baseline": raw["EAR_baseline"],
            "seat_departed": flg["seat_departed"],
        })
    return pd.DataFrame(rows, columns=CSV_COLUMNS).round(3)


def run(
        video_path,
        target_fps=TARGET_FPS,
        segment_sec=SEGMENT_SEC,
        face_model_path=FACE_MODEL_PATH,
        pose_model_path=POSE_MODEL_PATH,
        warmup_segments=WARMUP_SEGMENTS,
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
    segment_size = target_fps * segment_sec

    # 1) 랜드마크 추출
    lm = extract_landmarks(video_path, target_fps, face_model_path, pose_model_path)

    # 2) 세그먼트 점수
    t1 = time.time()
    seg_results = compute_segment_scores_7features(
        processed_frames=lm["processed_frames"],
        pose_keypoints=lm["pose_keypoints"],
        eyes_landmarks=lm["eyes_landmarks"],
        face_bboxes=lm["face_bboxes"],
        frame_sizes=lm["frame_sizes"],
        segment_size=segment_size,
        phone_bboxes=phone_bboxes,
        debug=debug,
    )
    t_scoring = time.time() - t1

    # 3) 개인화
    baseline = compute_personal_baseline_stability_filtered(seg_results, warmup_segments)
    personalized = apply_personalization_with_delta(seg_results, baseline)

    df = segments_to_dataframe(personalized)

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