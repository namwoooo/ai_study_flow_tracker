"""개인 기준(직전 구간) 대비 변화로 개인화 점수를 계산

- 기준 구간: 현재 세그먼트 **이전**의 최근 PERSONAL_WINDOW_SEC 동안 present 세그먼트
  (과거 데이터만 사용 → 실시간 계산 가능, 영상 길이와 무관하게 동작)
- 피처별 기준값 = 중앙값, 스케일 = max(1.4826 × MAD, 피처별 하한)
  (중앙값·MAD는 기준 구간에 딴짓이 절반 미만 섞여도 흔들리지 않음)
- 평소보다 **나빠진 방향**의 이탈만 감점: z = max(0, 이탈 / 스케일), stability = exp(-가중평균(min(z, Z_MAX)))
- personalized = FINAL_MIX_WEIGHT × final + STABILITY_MIX_WEIGHT × stability
"""

import numpy as np

from .config import (
    FINAL_MIX_WEIGHT,
    PERSONAL_FEATURES,
    PERSONAL_MIN_SEGMENTS,
    PERSONAL_WINDOW_SEC,
    PERSONAL_Z_MAX,
    SEGMENT_SEC,
    SLEEPY_PENALTY,
    STABILITY_MIX_WEIGHT,
)

NAN = float("nan")


def _value(seg, source, key):
    v = seg[source].get(key)
    return float(v) if v is not None else NAN


def _robust_stats(vals, floor):
    """(중앙값, 스케일). 스케일은 1.4826 × MAD(정규분포에서 표준편차와 같아지는 보정)와 하한 중 큰 값."""
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return NAN, NAN
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    return med, max(1.4826 * mad, floor)


def apply_personalization(segment_results, segment_sec=SEGMENT_SEC):
    """세그먼트 결과에 personalized, personal_stability를 추가한다.

    Returns:
        personalized (list[dict]): 입력 세그먼트에 scores.personalized / personal_stability 추가
        summary (dict): 마지막 present 세그먼트 시점의 개인 기준값 (리포트·디버깅용)
    """
    window = max(1, int(round(PERSONAL_WINDOW_SEC / segment_sec)))
    vals = {
        name: np.array([_value(s, cfg["source"], cfg["key"]) for s in segment_results])
        for name, cfg in PERSONAL_FEATURES.items()
    }
    present = np.array([s["flags"]["status"] == "present" for s in segment_results])

    out = []
    summary = {"num_stable_segments": 0, "features": {}}

    for i, seg in enumerate(segment_results):
        scores = dict(seg["scores"])
        personalized, stability = NAN, NAN
        ref_idx = [j for j in range(max(0, i - window), i) if present[j]]   # 과거만

        if not present[i]:
            personalized, stability = NAN, NAN
        elif len(ref_idx) < PERSONAL_MIN_SEGMENTS:
            # 기준이 쌓이는 중(시작 후 약 1분): 개인화 없이 final 사용
            personalized, stability = scores["final"], NAN
        else:
            zs, wts, ref_stats = {}, {}, {}
            for name, cfg in PERSONAL_FEATURES.items():
                x = vals[name][i]
                med, scale = _robust_stats(vals[name][ref_idx], cfg["floor"])
                ref_stats[name] = (med, scale)
                if not (np.isfinite(x) and np.isfinite(med)):
                    continue
                worse = (x - med) if cfg["higher_is_worse"] else (med - x)
                zs[name] = min(max(0.0, worse / scale), PERSONAL_Z_MAX)
                wts[name] = cfg["weight"]

                if wts:
                    # z를 먼저 평균: 한 피처의 큰 이탈이 다른 피처의 정상값에 희석되지 않도록
                    z_mean = sum(wts[n] * zs[n] for n in wts) / sum(wts.values())
                    stability = float(np.exp(-z_mean))
                    if seg["raw"].get("sleepy_flag") == 1:
                        stability *= SLEEPY_PENALTY
                    personalized = FINAL_MIX_WEIGHT * scores["final"] + STABILITY_MIX_WEIGHT * stability
                else:
                    personalized, stability = scores["final"], NAN

            summary = {
                "num_stable_segments": len(ref_idx),
                "features": {n: {"median": round(m, 4), "scale": round(sc, 4)}
                             for n, (m, sc) in ref_stats.items() if np.isfinite(m)},
            }

        scores["personalized"] = round(float(personalized), 4)
        scores["personal_stability"] = round(float(stability), 4)
        out.append({**seg, "scores": scores})

    return out, summary