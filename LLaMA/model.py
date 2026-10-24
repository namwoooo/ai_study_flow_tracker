"""
Converted from a Jupyter Notebook to a Python-style file.

Formatting-only conversion: markdown cells are comments and cell boundaries
are marked with # %% . Code cell contents are preserved as provided.
Notebook-specific commands may need adjustment before running as a .py file.
"""

# %% [markdown]
# #LLaMA 로딩 from hugging face

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
!pip install transformers accelerate sentencepiece bitsandbytes
from huggingface_hub import notebook_login
notebook_login()

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

LLM_NAME = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(LLM_NAME, use_auth_token=True)
llm = AutoModelForCausalLM.from_pretrained(
  LLM_NAME,
  torch_dtype=torch.float16,
  device_map="auto",
  use_auth_token=True
)

# %% [markdown]
# #7개 feature segment 분석 / 프롬프트

# %%
import pandas as pd

def build_focus_summary(csv_path):
  df = pd.read_csv(csv_path)

  lines = []
  for _, row in df.iterrows():
    seg = int(row["segment_idx"])
    final = row["final"]

    phone_face = row["phone_face"]
    head_angle = row["head_angle"]
    shoulder = row["shoulder_tilt"]
    hand_face = row["hand_face"]
    blink = row["blink"]
    perclos = row["perclos"]
    seat = bool(row["seat_departed"])

    status = "자리 이탈 발생" if seat else "정상"

    line = (
      f"[세그먼트 {seg}] (약 {seg*5}~{(seg+1)*5}초)\n"
      f"- final: {final*100:.1f}점\n"
      f"- phone_face: {phone_face:.2f}, head_angle: {head_angle:.2f}, "
      f"shoulder_tilt: {shoulder:.2f}, hand_face: {hand_face:.2f}\n"
      f"- blink: {blink:.2f}, perclos: {perclos:.2f}\n"
      f"- 상태: {status}\n"
    )
    lines.append(line)

  return "\n".join(lines)

summary_text = build_focus_summary("/content/drive/MyDrive/ai_study_flow_tracker/data_preprocessing/labels_revised.csv")
print(summary_text[:2000])

# %%
# 시각화

import re
import matplotlib.pyplot as plt

def parse_focus_segments(summary_text):
  """
  [세그먼트 N] (약 X~Y초) - final: xx.x점 형태의 텍스트를
  [(start_time, final_score)] 리스트로 변환
  """
  pattern = r"\[세그먼트 (\d+)\] \(약 (\d+)~(\d+)초\)\s*- final: ([\d\.]+)점"
  matches = re.findall(pattern, summary_text)
  times_scores = []
  for seg_idx, start, end, score in matches:
    # x축을 초 단위로 - 구간의 시작(평균도 무방)
    times_scores.append((int(start), float(score)))
  return times_scores

def save_and_show_final_score_chart(scores, save_path="/content/drive/MyDrive/ai_study_flow_tracker/LLaMA/results/final_score_chart.png"):
  # 그래프에서는 분으로 표현
  times = [t//60 for t, _ in scores]
  values = [v for _, v in scores]

  plt.figure(figsize=(12, 5))
  plt.plot(times, values, marker='', color='dodgerblue', label='final score')
  plt.xlabel('Time (min)')
  plt.ylabel('Score')
  plt.title('Focus Score through Time')
  plt.grid(True)
  plt.legend()
  if values:
    max_idx = values.index(max(values))
    min_idx = values.index(0)
    plt.legend()
  plt.tight_layout()
  plt.savefig(save_path)
  plt.close()
  print(f"그래프 저장 완료: {save_path}")

  # Jupyter/Colab에서 바로 띄워주기
  try:
    from IPython.display import Image, display
    display(Image(filename=save_path))
  except ImportError:
    pass

# %%
times_scores = parse_focus_segments(summary_text)

# %%
save_and_show_final_score_chart(times_scores, save_path="/content/drive/MyDrive/ai_study_flow_tracker/LLaMA/results/final_score_chart.png")

# %%
# 프롬프트

def build_llama_prompt(segments_json: str, user_name: str = "사용자") -> str:
  """
  segments_json: 세그먼트별 점수 리스트(JSON)
  baseline_json: 개인 baseline(JSON)
  """
  return f"""
[SYSTEM]
당신은 학생들의 학습 집중 패턴을 분석하고,
누구나 쉽게 이해할 수 있는 "친절한 한국어 설명 리포트"를 작성하는 전문가입니다.

목표:
- 어려운 기술 용어나 숫자를 가능한 한 사용하지 않고,
- 학생이 읽으면 “아, 내가 공부할 때 이런 습관이 있구나” 하고 직관적으로 이해할 수 있도록 설명하세요.
- 행동을 중심으로 설명하고, 개선 방향은 실제로 따라 할 수 있는 것만 제시하세요.
- 어려운 용어(EAR, landmark, baseline 등)는 절대 사용하지 않습니다.
- 점수나 수치를 그대로 말하지 말고 “자주 ~했다”, “대체로 안정적이었다”, “중간에 흐트러짐이 있었다”처럼 자연스러운 언어로 설명하세요.

[INPUT DATA]

1) 세그먼트 데이터 (5초 단위)
- segment_idx: 번호
- phone_face, seat_depart, head_angle, shoulder_tilt,
  hand_face, blink, perclos, final, personalized 등은
  "학습 중 어떤 행동이 얼마나 안정적이었는지" 나타내는 값입니다.
- 하지만 리포트에서는 절대 값이나 숫자를 직접 말하지 말고,
  행동/상황을 자연스럽게 해석해서 전달하세요.

세그먼트 데이터:
{segments_json}

[ASSISTANT TASK]
위 데이터를 기반으로 {user_name}의 학습 집중 패턴을
학생이 읽어도 이해될 만큼 쉬운 리포트로 작성해주세요.

반드시 다음 구조를 따르세요:

---

[1. 전체 요약]
- 전체적인 집중 흐름을 간단하게 설명합니다.
- 초반/중반/후반 학습 흐름이 어땠는지 “느낌” 위주로 설명합니다.
- 이 학생이 가진 좋은 점 1~2가지,
  그리고 쉽게 흐트러지는 부분 1~2가지를 부드럽게 언급합니다.

---

[2. 시간 흐름에 따른 집중 패턴]
- 초반, 중반, 후반으로 나누어 어떤 특징이 있었는지 설명합니다.
- 집중이 잘 되었던 구간은 “집중이 잘 잡힌 편”, “습관이 안정적이었다” 등으로 표현합니다.
- 집중이 떨어진 구간은 “잠깐 흐트러짐이 있었다”, “몸이 조금 불편해 보였다” 등 자연어로 표현합니다.
- 자리를 벗어난 구간이 있다면, “중간에 잠깐 자리에서 움직인 것으로 보입니다” 정도로 설명합니다.

---

[3. 주요 습관 분석 (학생이 이해하기 쉽게)]
아래 항목을 각각 4~6문장으로 설명하세요.
절대 기술적 용어/수치 X.

● 시선 습관
- 화면이나 교재를 얼마나 안정적으로 바라보는지
- 흔들렸던 순간이 있었다면 어떤 흐름 때문인지

● 고개·목 자세
- 고개가 자주 움직였는지
- 집중이 잘 될 때의 머리 움직임 특징

● 어깨·상체 자세
- 상체가 안정적이었는지
- 긴장/피로로 인한 변화가 있었는지

● 손·얼굴의 움직임
- 손이 얼굴로 자주 갔는지
- 산만해지는 타이밍과의 관계

● 눈 깜빡임·피곤함 신호
- 중간에 피곤했을 가능성이 있었는지
- 집중 흐름과 피로 신호의 상관성

---

[4. {user_name}의 집중 스타일 해석]
- 이 학생이 어떤 방식으로 집중하는지 1~2단락으로 설명하세요.
- 예: "초반에 빠르게 몰입하지만 일정 시간이 지나면 몸이 조금씩 움직이며 집중력이 떨어지는 편입니다."
- 또는 "시선은 안정적이지만, 손이 얼굴 쪽으로 자주 가면서 흐름이 끊기는 경향이 있습니다." 등.

---

[5. 맞춤형 개선 팁 (실제로 할 수 있는 것만)]
아래 예시처럼 구체적이고 간단한 행동 조언 4~7개를 bullet로 제시하세요.
- “고개가 자주 움직이면 10분에 한 번씩 상체를 한 번 펴 주세요.”
- “손이 얼굴로 자주 간다면 펜을 손에 쥐고 공부하면 도움이 됩니다.”
- “잠깐 피곤해 보였다면 20~30분 간격으로 10초 정도 눈 스트레칭을 해보세요.”

절대 모호한 조언 금지(예: “더 집중하세요”, “자세를 바로 하세요”).

---

[6. 마무리]
- 이 학생이 가진 강점을 다시 한 번 강조하고,
- 앞으로 어떻게 더 좋아질 수 있는지 긍정적인 톤으로 마무리하세요.

"""

# %% [markdown]
# #리포트 생성

# %%
prompt = build_llama_prompt(summary_text, "김이화")

inputs = tokenizer(prompt, return_tensors="pt").to(llm.device)

output = llm.generate(
  **inputs,
  max_new_tokens=1000,
  temperature=0.6,
  top_p=0.9
)

report = tokenizer.decode(output[0], skip_special_tokens=True)
print(report.replace(prompt, ""))

prompt_replaced_report = report.replace(prompt, "")

# %%
with open("/content/drive/MyDrive/ai_study_flow_tracker/LLaMA/reports/report.txt", "w", encoding="utf-8-sig") as f:
  f.write(prompt_replaced_report)
