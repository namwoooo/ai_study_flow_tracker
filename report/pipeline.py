"""run(): 전체 흐름 조립 및 저장"""

import os
import time

from report.digest import build_session_digest
from report.llm import load_model
from report.prompt import build_prompt
from report.llm import generate_report


def run(csv_path: str, user_name: str, output_path: str, hf_token: str | None = None) -> str:
    """CSV -> 프롬프트 -> LLaMA 생성 -> 저장까지 한 번에 수행"""
    t0 = time.time()
    tokenizer, llm = load_model(hf_token=hf_token)
    t1 = time.time()
    print(f"[1] 모델 로드: {t1 - t0:.1f}초")

    summary_text = build_session_digest(csv_path)
    t2 = time.time()
    print(f"[2] CSV → 프롬프트 요약 변환: {t2 - t1:.1f}초")

    system_message, user_message = build_prompt(summary_text, user_name)
    t3 = time.time()
    print(f"[3] 프롬프트 구성: {t3 - t2:.1f}초")

    report_text = generate_report(tokenizer, llm, system_message, user_message)
    t4 = time.time()
    print(f"[4] LLM 생성: {t4 - t3:.1f}초")

    save_report(report_text, output_path)
    t5 = time.time()
    print(f"[5] 저장: {t5 - t4:.1f}초")
    print(f"총 소요 시간: {t5 - t0:.1f}초")

    return report_text


def save_report(report_text: str, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8-sig") as f:
        f.write(report_text)