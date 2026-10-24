"""csv -> 세션 요약 dict"""

import statistics
import pandas as pd
import numpy as np

from report.config import FEATURE_COLUMNS


# 프롬프트로 전달할 data 요약 -> 영상이 길어져도 프롬프트가 크게 늘어나지 않도록
def build_session_digest(csv_path) -> dict:
    """총 segment 수, 총 길이"""
    df = pd.read_csv(csv_path)
    total_seconds = len(df) * 5
    scores = df["personalized"]

    # 전체 통계 (평균/최댓값/최솟값/표준편차)
    overall = {
        "avg": round(statistics.mean(scores), 3),
        "max": round(max(scores), 3),
        "min": round(min(scores), 3),
        "std": round(statistics.stdev(scores), 3) if len(scores) > 1 else 0.0,
    }
    # 각 feature 별 평균
    overall_feature_avg = {col: round(float(df[col].mean()), 3) for col in FEATURE_COLUMNS}

    # 초반, 중반, 후반 집계
    phase_names = ["초반", "중반", "후반"]
    df_sections = np.array_split(df, 3)
    phases = {}

    for name, idx in zip(phase_names, np.array_split(np.arange(len(df)), 3)):
        section = df.iloc[idx]
        if len(section) == 0:
            continue
        start_sec = int(section["segment_idx"].iloc[0]) * 5
        end_sec = (int(section["segment_idx"].iloc[-1]) + 1) * 5
        # 구간별 피처 평균
        phases[name] = {
            "time_range": f"{_format_time(start_sec)}~{_format_time(end_sec)}",
            "avg_score": round(float(section["personalized"].mean()), 3),
            "features": {col: round(float(section[col].mean()), 3) for col in FEATURE_COLUMNS},
        }

    # 집중 저하 구간 top 3
    threshold = overall["avg"] - overall["std"]
    low_mask = scores < threshold
    low_runs = []
    run_start = None

    for seg_idx, is_low in zip(df["segment_idx"], low_mask):
        if is_low and run_start is None:
            run_start = seg_idx
        elif not is_low and run_start is not None:
            low_runs.append((run_start, seg_idx - 1))
            run_start = None

    if run_start is not None:
        low_runs.append((run_start, df["segment_idx"].iloc[-1]))

    low_focus_periods = []
    for start_idx, end_idx in low_runs:
        window = df[(df["segment_idx"] >= start_idx) & (df["segment_idx"] <= end_idx)]
        low_focus_periods.append({
            "time_range": f"{_format_time(start_idx * 5)} ~ {_format_time((end_idx + 1) * 5)}",
            "avg_score": round(float(window["personalized"].mean()), 3),
            "feature_delta": {
                col: round(float(window[col].mean() - overall_feature_avg[col]), 3)
                for col in FEATURE_COLUMNS
            },
            "duration_sec": int((end_idx - start_idx + 1) * 5)
        })

    low_focus_periods.sort(key=lambda p: p["avg_score"])
    low_focus_periods = low_focus_periods[:3]

    # 이벤트 (자리 이탈 / 졸음)
    seat_departed_times = [
        _format_time(seg * 5) for seg in df.loc[df["seat_departed"] == True, "segment_idx"]
    ]
    sleepy_times = [
        _format_time(seg * 5) for seg in df.loc[df["sleepy_flag"] == 1, "segment_idx"]
    ]

    # 전반부 평균, 후반부 평균 비교
    half = len(df) // 2
    first_half_avg = scores.iloc[:half].mean() if half > 0 else scores.mean()
    second_half_avg = scores.iloc[half:].mean()
    diff = second_half_avg - first_half_avg
    if diff > 0.05:
        trend = "상승"
    elif diff < -0.05:
        trend = "하강"
    else:
        trend = "안정"

    return {
        "total_duration": _format_time(total_seconds),
        "total_segments": len(df),
        "overall": overall,
        "overall_feature_avg": overall_feature_avg,
        "phases": phases,
        "low_focus_periods": low_focus_periods,
        "events": {
            "seat_departed_times": seat_departed_times,
            "sleepy_times": sleepy_times,
        },
        "trend": trend,
    }


def _format_time(total_seconds: float) -> str:
    """초 단위 값 -> '1분 5초'와 같은 형식으로 변환"""
    m, s = divmod(int(total_seconds), 60)
    return f"{m}분 {s}초" if m else f"{s}초"