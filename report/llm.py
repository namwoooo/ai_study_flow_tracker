"""LLM 로드·캐시·생성 (GPU 필요)"""
import gc
import os
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from .config import LLM_NAME, MAX_NEW_TOKENS, TEMPERATURE, TOP_P, REPETITION_PENALTY

# 모듈 전역 캐시: 같은 프로세스 안에서 한 번 로드한 모델을 재사용
_tokenizer = None
_llm = None
_loaded_name: str | None = None


def load_model(model_name: str = LLM_NAME, hf_token: str | None = None):
    """모델을 로드한다. 같은 모델이 이미 올라가 있으면 캐시를 반환한다."""
    global _tokenizer, _llm, _loaded_name

    if _llm is not None and _loaded_name == model_name:
        print(f"[load_model] 캐시 사용: {model_name}")
        return _tokenizer, _llm

    if _llm is not None:
        print(f"[load_model] 모델 교체: {_loaded_name} → {model_name}")
        unload_model()

    if torch.cuda.is_available():
        print(f"[load_model] 새로 로드 (현재 GPU 할당 {torch.cuda.memory_allocated() / 1e9:.1f} GB)")

    token = hf_token or os.environ.get("HF_TOKEN")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    _tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    _llm = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        token=token,
    )
    _loaded_name = model_name
    print(f"[load_model] 로드 완료: {_llm.device}")
    return _tokenizer, _llm


def unload_model() -> None:
    """캐시된 모델을 내리고 GPU 메모리를 정리한다 (모델 교체·OOM 복구용)."""
    global _tokenizer, _llm, _loaded_name
    _tokenizer = None
    _llm = None
    _loaded_name = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_loaded() -> bool:
    """모델이 메모리에 올라가 있는지 (FastAPI 헬스체크용)"""
    return _llm is not None


def _eos_token_ids(tokenizer) -> list[int]:
    """모델별 턴 종료 토큰을 자동으로 모은다 (Qwen: <|im_end|>, Llama: <|eot_id|>)."""
    ids = {tokenizer.eos_token_id}
    for tok in ("<|im_end|>", "<|eot_id|>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != tokenizer.unk_token_id:
            ids.add(tid)
    return [i for i in ids if i is not None]


def generate_report(
        tokenizer,
        llm,
        system_message: str,
        user_message: str,
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
) -> str:
    """system/user 메시지로 리포트 텍스트를 생성한다."""
    inputs = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(llm.device)
    input_len = inputs["input_ids"].shape[-1]

    t = time.time()
    output = llm.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        repetition_penalty=REPETITION_PENALTY,
        eos_token_id=_eos_token_ids(tokenizer),
    )
    elapsed = time.time() - t

    n_new = output.shape[-1] - input_len
    print(f"[generate] 입력 {input_len} / 생성 {n_new} 토큰, {n_new / elapsed:.1f} tok/s")

    return tokenizer.decode(
        output[0][input_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )