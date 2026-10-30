"""안정 구간 기반 개인화 baseline 계산 및 적용"""

import numpy as np

from .config import (
    BASELINE_DEFAULT_MEANS,
    FINAL_MIX_WEIGHT,
    SEAT_DEPART_PENALTY,
    SLEEPY_PENALTY,
    STABILITY_FILTER,
    STABILITY_MIX_WEIGHT,
    STABILITY_WEIGHTS,
    WARMUP_SEGMENTS,
)

from .features import weighted_mean


def _safe_stats(arr, default_mean):
    if len(arr) == 0:
        return default_mean, 1e-4
    arr = np.array(arr, dtype=np.float32)
    return float(arr.mean()), float(arr.std() + 1e-4)


def _stability(x, mu, sigma):
    """z = |x - μ| / σ, stability = exp(-z). baseline에서 멀수록 빠르게 감점."""
    z = abs(x - mu) / sigma
    return float(np.exp(-z))


def _is_stable(seg):
    """baseline 계산에 쓸 안정 구간인지 판정."""
    sc, raw, flg = seg["scores"], seg["raw"], seg["flags"]
    f = STABILITY_FILTER

    if flg["seat_departed"]:                           # 자리 이탈
        return False
    if raw["blink_rate_proxy"] > f["blink_rate_max"]:  # blink spike
        return False
    if raw["sleepy_flag"] == 1:                        # 졸음
        return False
    if sc["hand_face"] < f["hand_face_min"]:           # 손-얼굴 과근접 (폰 사용 의심)
        return False
    if sc["phone_use"] < f["phone_use_min"]:           # 폰 사용
        return False
    if sc["head_angle"] < f["head_angle_min"]:         # 고개 과도 기울어짐
        return False
    if sc["shoulder_tilt"] < f["shoulder_tilt_min"]:   # 어깨 틀어짐
        return False
    return True


def compute_personal_baseline_stability_filtered(segment_results, warmup_segments=WARMUP_SEGMENTS):
    """초기 warmup 구간 중 안정적 집중 상태인 세그먼트만으로 개인 baseline 구축."""
    warm = segment_results[:min(warmup_segments, len(segment_results))]

    head_vals, shoulder_vals, hand_face_vals, blink_vals, phone_vals = [], [], [], [], []

    for seg in warm:
        if not _is_stable(seg):
            continue
        sc, raw = seg["scores"], seg["raw"]
        head_vals.append(sc["head_angle"])
        shoulder_vals.append(sc["shoulder_tilt"])
        hand_face_vals.append(sc["hand_face"])
        blink_vals.append(raw["blink_rate_proxy"])
        phone_vals.append(sc["phone_use"])

    d = BASELINE_DEFAULT_MEANS
    head_mean, head_std = _safe_stats(head_vals, d["head"])
    shoulder_mean, shoulder_std = _safe_stats(shoulder_vals, d["shoulder"])
    hand_mean, hand_std = _safe_stats(hand_face_vals, d["hand"])
    blink_mean, blink_std = _safe_stats(blink_vals, d["blink"])
    phone_mean, phone_std = _safe_stats(phone_vals, d["phone"])

    return {
        "head_mean": head_mean,         "head_std": head_std,
        "shoulder_mean": shoulder_mean, "shoulder_std": shoulder_std,
        "hand_mean": hand_mean,         "hand_std": hand_std,
        "blink_mean": blink_mean,       "blink_std": blink_std,
        "phone_mean": phone_mean,       "phone_std": phone_std,
        "num_stable_segments": len(head_vals),
    }


def apply_personalization_with_delta(segment_results, baseline):
    """baseline 대비 변화량으로 개인화 점수 계산.

    personalized = 0.6 * final + 0.4 * personal_stability
    """
    b = baseline
    sw = STABILITY_WEIGHTS
    personalized = []

    for seg in segment_results:
        sc, raw, flg = seg["scores"], seg["raw"], seg["flags"]

        st_head = _stability(sc["head_angle"], b["head_mean"], b["head_std"])
        st_shoul = _stability(sc["shoulder_tilt"], b["shoulder_mean"], b["shoulder_std"])
        st_hand = _stability(sc["hand_face"], b["hand_mean"], b["hand_std"])
        st_blink = _stability(raw["blink_rate_proxy"], b["blink_mean"], b["blink_std"])
        st_phone = _stability(sc["phone_use"], b["phone_mean"], b["phone_std"])

        personal_stability = weighted_mean(
            {"head": st_head, "shoulder": st_shoul, "hand": st_hand,
             "blink": st_blink, "phone": st_phone},
            STABILITY_WEIGHTS,
        )

        if flg["seat_departed"]:
            personal_stability *= SEAT_DEPART_PENALTY
        if raw["sleepy_flag"] == 1:
            personal_stability *= SLEEPY_PENALTY

        final_personal = FINAL_MIX_WEIGHT * sc["final"] + STABILITY_MIX_WEIGHT * personal_stability

        seg_new = seg.copy()
        seg_new["scores"] = seg["scores"].copy()
        seg_new["scores"]["personalized"] = round(final_personal, 4)
        seg_new["scores"]["personal_stability"] = round(personal_stability, 4)
        personalized.append(seg_new)

    return personalized