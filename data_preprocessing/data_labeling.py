# 1) 라이브러리 임포트

import os
import cv2
import numpy as np #파일 경로, 영상처리, 수치연산 사용 import
import mediapipe as mp # 얼굴 눈 몸 랜드마크 검출
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd
import time


# --- EAR에 쓸 MediaPipe FaceMesh 눈 랜드마크 인덱스(6 points) ---
# (자주 쓰이는 조합. 왼/오른쪽 각각 p1..p6 순서)
LEFT_EYE_IDXS  = [33, 160, 158, 133, 153, 144]   # p1,p2,p3,p4,p5,p6
RIGHT_EYE_IDXS = [362, 385, 387, 263, 373, 380]  # p1,p2,p3,p4,p5,p6

def eye_aspect_ratio_6pts(eye6): #입력된 6점으로 ear 계산
  # eye6: np.array shape (6,2) in pixel coords
  p1,p2,p3,p4,p5,p6 = eye6
  vert = np.linalg.norm(p2-p6) + np.linalg.norm(p3-p5)
  horiz = 2.0*np.linalg.norm(p1-p4) + 1e-8
  return float(vert / horiz)

def time_decay_weights(T, lam=0.08):
  # 세그먼트 내부 최근 프레임에 더 큰 가중치를 주는 지수 감쇠 가중치 w 생성, 인덱스가 뒤로 갈수록 (최근으로 갈 수록 값이 큼)
  idx = np.arange(T)
  w = np.exp(-lam * (T - 1 - idx))
  return w / w.sum()

def to_score(x, lo, hi, small_is_good=True): #어떤 원시값 x를 [low, high]으로 클리핑 후 0~1 점수화
  x = np.clip(x, lo, hi)
  if small_is_good:
    return 1.0 - (x - lo) / (hi - lo + 1e-8)
  else:
    return (x - lo) / (hi - lo + 1e-8)

def l2_norm(a, b): #두 벡터 간 L2 (유클리드) 거리
  return np.linalg.norm(a - b, ord=2)

# 논문식 통합 눈감음값→집중도 (세그먼트 단위)
def paper_gaze_score_from_ears(ear_L_series, ear_R_series, ear_thresh=0.21,
                               ec_clip=(1e-3, 0.9), conc_minmax=(1.0, 10.0)):
  # 유효 프레임(눈 랜드마크 성공)만 집계
  L_valid = np.isfinite(ear_L_series)
  R_valid = np.isfinite(ear_R_series)
  c_l = L_valid.mean() if L_valid.size else 0.0 #유효 프레임 (신뢰도)
  c_r = R_valid.mean() if R_valid.size else 0.0 #유효 프레임 (신뢰도)

  if c_l + c_r <= 0:
    return 0.5  # 정보 부족시 중립

  # 닫힘 비율 e_l, e_r
  e_l = (ear_L_series[L_valid] < ear_thresh).mean() if c_l > 0 else np.nan
  e_r = (ear_R_series[R_valid] < ear_thresh).mean() if c_r > 0 else np.nan
  e_l = 0.0 if np.isnan(e_l) else float(e_l)
  e_r = 0.0 if np.isnan(e_r) else float(e_r)

  e_c = (c_l*e_l + c_r*e_r) / (c_l + c_r) #통합 눈 감음 값
  e_c = float(np.clip(e_c, ec_clip[0], ec_clip[1]))  # 폭주 방지
  conc_raw = 1.0 / e_c #집중도 원시값 (닫힘 적을수록 큼)

  mn, mx = conc_minmax #min-max 정규화 후 0~1로 반환
  conc_norm = (conc_raw - mn) / (mx - mn + 1e-8)
  return float(np.clip(conc_norm, 0.0, 1.0))


# 2) 영상 경로 & 파라미터

video_path = "/content/drive/MyDrive/ai_study_flow_tracker/video_namwoo.mp4"
target_fps   = 5            # 샘플링 fps (초당 5 프레임으로 균일 샘플링)
segment_sec  = 5             # 세그먼트 길이(초), 세그먼트 길이를 5초 = 50프레임으로 설정
segment_size = target_fps * segment_sec
model_input_size = (224, 224)  # ViT 입력 크기, 얼굴 crop을 모델 입력 크기로 리사이즈

if not os.path.exists(video_path):
  raise FileNotFoundError(f"{video_path} 존재하지 않음")


# 3) MediaPipe 초기화 (video용)

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
PoseLandmarker = mp.tasks.vision.PoseLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

face_base_options = BaseOptions(model_asset_path="face_landmarker.task")
pose_base_options = BaseOptions(model_asset_path="pose_landmarker.task")

mp_face = FaceLandmarker.create_from_options(FaceLandmarkerOptions(base_options=face_base_options, running_mode=VisionRunningMode.VIDEO, num_faces=1)
                                             )
mp_pose = PoseLandmarker.create_from_options(     #pose도 비디오 모드로 초기화
  PoseLandmarkerOptions(base_options=pose_base_options, running_mode=VisionRunningMode.VIDEO, num_poses=1)
)


# 4) 영상 읽고 프레임 샘플링 + 얼굴 crop + 정규화 + 눈/EAR + Pose

cap = cv2.VideoCapture(video_path)
native_fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = max(1, round(native_fps / target_fps))  # 몇 프레임마다 하나씩 뽑을지

processed_frames = []     # 모델 입력용 face crop (정규화)
face_bboxes = []          # (x1,y1,x2,y2) or None, 얼굴 바운딩 박스
eyes_landmarks = []       # (eyeL(6,2), eyeR(6,2)) or None
pose_keypoints = []       # (33*3,) float list
frame_sizes = []
frame_idx = 0
t_idx = 0

# 전처리 시간 측정
t0 = time.time()
t_face_total = 0.0
t_pose_total = 0.0
t_skip_total = 0.0

# grab()과 retrieve()로 분리하여 실제 디코딩할 프레임만 디코딩
while True:
  if frame_idx % frame_interval == 0:
    ret = cap.grab()
    if not ret:
      break
    ret, frame_bgr = cap.retrieve()  # 실제 디코딩할 프레임
    if not ret:
      break
    frame_idx += 1

    t_sec = (frame_idx - 1) / native_fps
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = frame_bgr.shape

    # numpy array -> mp.Image 타입 전환
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    timestamp_ms = int(t_sec * 1000)

    _tf = time.time()
    res_face = mp_face.detect_for_video(mp_image, timestamp_ms)
    t_face_total += time.time() - _tf

    if not res_face.face_landmarks:
      processed_frames.append(None)
      face_bboxes.append(None)
      frame_sizes.append(None)
      eyes_landmarks.append(None)
      pose_keypoints.append(None)
      continue

    lm = res_face.face_landmarks[0]
    x_coords = np.array([p.x for p in lm]); y_coords = np.array([p.y for p in lm])
    xmin, xmax = int(x_coords.min()*w), int(x_coords.max()*w)
    ymin, ymax = int(y_coords.min()*h), int(y_coords.max()*h)
    xmin, ymin = max(0, xmin), max(0, ymin)
    xmax, ymax = min(w-1, xmax), min(h-1, ymax)
    if xmax <= xmin or ymax <= ymin:
      processed_frames.append(None)
      face_bboxes.append(None)
      frame_sizes.append(None)
      eyes_landmarks.append(None)
      pose_keypoints.append(None)
      continue

    bbox = (xmin, ymin, xmax, ymax)

    face_crop = frame_bgr[ymin:ymax, xmin:xmax]
    if face_crop.size == 0:
      processed_frames.append(None)
      face_bboxes.append(None)
      frame_sizes.append(None)
      eyes_landmarks.append(None)
      pose_keypoints.append(None)
      continue
    face_bboxes.append(bbox)
    frame_sizes.append((w, h))   # 이 프레임의 원본 크기 기록
    processed_frames.append(True)

    lm_np = np.array([[p.x*w, p.y*h] for p in lm], dtype=np.float32)
    try:
      eyeL = lm_np[LEFT_EYE_IDXS, :]
      eyeR = lm_np[RIGHT_EYE_IDXS, :]
      eyes_landmarks.append((eyeL, eyeR))
    except Exception:
      eyes_landmarks.append(None)

    # res_pose = mp_pose.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    _tp = time.time()
    res_pose = mp_pose.detect_for_video(mp_image, timestamp_ms)
    t_pose_total += time.time() - _tp

    if res_pose.pose_landmarks:
      vec = []
      for lm_p in res_pose.pose_landmarks[0]:
        vec.extend([lm_p.x, lm_p.y, lm_p.z])  # MediaPipe Pose는 정규화 좌표
      pose_keypoints.append(vec)
    else:
      pose_keypoints.append([0.0]*(33*3))
    t_idx += 1
  else:
    _ts = time.time()
    ret = cap.grab()  # 디코딩 없이 건너 뛰기
    t_skip_total += time.time() - _ts
    if not ret:
      break
    frame_idx += 1
cap.release()
num_frames = len(processed_frames)  # 루프 후 실제 처리된 샘플 수

print(f"[전처리] 프레임 추출+mediapipe: {time.time() - t0:.1f}초 (총 {num_frames}프레임)")
#%%
# BlazePose index mapping

POSE_IDX = {
  'nose': 0,
  'eye_l': 2, 'eye_r': 5,
  'shoulder_l': 11, 'shoulder_r': 12,
  'wrist_l': 15, 'wrist_r': 16,
  'hip_l': 23, 'hip_r': 24
}

def get_pose_xy(pose_vec, idx, w, h):
  base = idx * 3
  if base+1 >= len(pose_vec):
    return None
  x_n, y_n = pose_vec[base], pose_vec[base+1]
  if not (0 <= x_n <= 1 and 0 <= y_n <= 1):
    return None
  return (x_n * w, y_n * h)

def center_of_bbox(b):
  x1,y1,x2,y2 = b
  return np.array([(x1+x2)/2, (y1+y2)/2], dtype=np.float32)

def angle_from_horizontal(p_left, p_right):
  v = np.array(p_right) - np.array(p_left)
  return abs(np.arctan2(v[1], v[0]))

def perclos(ear_series, ear_thresh=0.21):
  ear_series = np.array(ear_series, dtype=np.float32)
  valid = np.isfinite(ear_series)
  if valid.sum()==0: return np.nan
  return float((ear_series[valid] < ear_thresh).mean())



# 업그레이드 졸음 기능 (Baseline EAR + 70% closure + 5초 지속 + head nod)

def compute_EAR_baseline(ear_series, min_samples=30):
  ear_series = np.array(ear_series)
  ear_series = ear_series[np.isfinite(ear_series)]
  if len(ear_series) < min_samples:
    return np.median(ear_series) if len(ear_series)>0 else 0.25
  return np.median(ear_series[:min_samples])

def is_eye_closure(ear, ear_baseline, ratio=0.55):
  return ear < (ear_baseline * ratio)



# segment scoring

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
        params=None
):

  PHONE_USE_THRESH = 25.0
  T = len(processed_frames)
  num_segments = T // segment_size
  results = []

  # 기본 파라미터
  P = {
    'ear_thresh': 0.21,
    'time_decay_lambda': 0.08,
    'norm': {
      'head_angle_rad':  (0.0, 0.35),
      'shoulder_tilt':   (0.0, 0.25),
      'hand_face_dist':  (30.0, 200.0),
      'blink_rate_proxy':(0.02, 0.30),
      'perclos':         (0.00, 0.35)
    },
    'w': {
      'phone_use':    0.16,
      'seat_depart':  0.12,
      'head_angle':   0.12,
      'shoulder_tilt':0.12,
      'hand_face':    0.16,
      'blink':        0.16,
      'perclos':      0.16
    },
    'seat_departure_penalty': 0.0
  }
  if params:
    for k,v in params.items():
      if k in P and isinstance(v,dict): P[k].update(v)
      else: P[k]=v

  # 기준점 자동 설정
  if ref_center is None or ref_radius is None:
    fcs = [center_of_bbox(b) for b in face_bboxes[:min(len(face_bboxes), max(segment_size, 30))] if b is not None]
    if len(fcs)>=3:
      ref_center = tuple(np.median(np.stack(fcs, axis=0),axis=0))
      hs=[(b[3]-b[1]) for b in face_bboxes[:min(len(face_bboxes),max(segment_size,30))] if b is not None]
      ref_radius = float(max(60.0, 2*np.median(hs))) if hs else 120.0
    else:
      ref_center=(0,0); ref_radius=120

  W = P['w']

  # EAR baseline 저장용
  global EAR_baseline_global
  baseline_buffer = []
  EAR_baseline_global = 0.25

  fps = 30  # 영상 FPS (필요시 수정)

  for seg_idx in range(num_segments):
    s = seg_idx * segment_size
    e = s + segment_size
    EAR_baseline = EAR_baseline_global
    wdec = time_decay_weights(segment_size, lam=P['time_decay_lambda'])

    # 누적
    phone_use_flags=[]
    head_angles=[]
    shoulder_tilts=[]
    hand_face_scores=[]
    seat_flags=[]
    ear_series=[]
    blink_transitions=0
    prev_closed=None

    closure_frames=0
    sleepy_flags=[]

    for t in range(s, e):
      # None일 경우 skip
      if frame_sizes[t] is None:
        continue
      w,h = frame_sizes[t]
      fb = face_bboxes[t]
      fc = center_of_bbox(fb) if fb is not None else None

      # ---------------------
      # 1) 폰 사용 (손-폰 거리)
      # ---------------------
      if phone_bboxes is not None and phone_bboxes[t] is not None:
        p = pose_keypoints[t]
        hand_center=None
        if p is not None:
          wL=get_pose_xy(p,POSE_IDX['wrist_l'],w,h)
          wR=get_pose_xy(p,POSE_IDX['wrist_r'],w,h)
          ws=[x for x in (wL,wR) if x is not None]
          if ws:
            hand_center=np.mean(np.array(ws),axis=0)

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
        eyeL,eyeR = eyes_landmarks[t]
        cenL=eyeL.mean(axis=0); cenR=eyeR.mean(axis=0)
        head_angles.append(angle_from_horizontal(cenL,cenR))

      # ---------------------
      # 3) 어깨 기울기 + 손 얼굴 거리 + 자리 이탈
      # ---------------------
      p = pose_keypoints[t]
      if p is not None:
        sL=get_pose_xy(p,POSE_IDX['shoulder_l'],w,h)
        sR=get_pose_xy(p,POSE_IDX['shoulder_r'],w,h)
        if sL and sR:
          shoulder_tilts.append(angle_from_horizontal(sL,sR))

        if fc is not None:
          wL=get_pose_xy(p,POSE_IDX['wrist_l'],w,h)
          wR=get_pose_xy(p,POSE_IDX['wrist_r'],w,h)
          ws=[x for x in (wL,wR) if x is not None]
          if ws:
            hand_center=np.mean(np.array(ws),axis=0)
            d_hf=np.linalg.norm(hand_center-fc)
            hand_face_scores.append(to_score(d_hf,*P['norm']['hand_face_dist'],small_is_good=False))

        nose=get_pose_xy(p,POSE_IDX['nose'],w,h)
        if nose is None:
          hl=get_pose_xy(p,POSE_IDX['hip_l'],w,h)
          hr=get_pose_xy(p,POSE_IDX['hip_r'],w,h)
          nose = tuple(np.mean([hl,hr],axis=0)) if (hl and hr) else None

        if nose is not None:
          dist=np.linalg.norm(np.array(nose)-np.array(ref_center))
          seat_flags.append(dist>ref_radius)

      # -----------------------------------------
      # 4) 업그레이드된 EAR 기반 졸음 탐지
      # -----------------------------------------
      if eyes_landmarks[t] is not None:
        try:
          eyeL,eyeR = eyes_landmarks[t]
          earL=eye_aspect_ratio_6pts(eyeL)
          earR=eye_aspect_ratio_6pts(eyeR)
          ear=0.5*(earL+earR)
          ear_series.append(ear)

          # baseline 초기 구간 모으기
          if seg_idx==0 and t<40:
            baseline_buffer.append(ear)

          # baseline 계산(첫 세그먼트 끝에서)
          if seg_idx==0 and t==e-1:
            EAR_baseline = compute_EAR_baseline(baseline_buffer)
            EAR_baseline_global = EAR_baseline
          else:
            EAR_baseline = EAR_baseline_global

          # 45% closure
          closed_45 = is_eye_closure(ear, EAR_baseline, ratio=0.55)
          if seg_idx < 3:  # 처음 몇 세그먼트만 확인
            print(f"[seg {seg_idx}, t={t}] ear={ear:.4f}, baseline={EAR_baseline:.4f}, threshold={EAR_baseline*0.55:.4f}, closed={closed_45}")

          if prev_closed is not None and prev_closed and not closed_45:
            blink_transitions += 1
          prev_closed = closed_45

          # closure duration
          if closed_45: closure_frames += 1
          else: closure_frames = 0

          closure_long = (closure_frames >= int(5 * fps))

          # head nod
          head_nod=False
          if len(head_angles)>=3:
            head_nod = (head_angles[-1]-head_angles[-3] > 0.15)

          sleepy_flag = closure_long or head_nod
          sleepy_flags.append(sleepy_flag)

        except:
          ear_series.append(np.nan)

    # -------------------------------
    # 점수 계산
    # -------------------------------
    usage_ratio = sum(phone_use_flags)/len(phone_use_flags) if phone_use_flags else 0.0
    phone_use_score = 1.0 - usage_ratio

    seat_departed = any(seat_flags)
    seat_depart_score = 0.0 if seat_departed else 1.0

    if head_angles:
      ha=np.array(head_angles)
      head_angle_score=to_score(ha.mean(),*P['norm']['head_angle_rad'],small_is_good=True)
    else: head_angle_score=0.5

    if shoulder_tilts:
      st=np.array(shoulder_tilts)
      shoulder_tilt_score=to_score(st.mean(),*P['norm']['shoulder_tilt'],small_is_good=True)
    else: shoulder_tilt_score=0.5

    hand_face_score = np.mean(hand_face_scores) if hand_face_scores else 0.5

    blink_rate_proxy = blink_transitions/max(1,(e-s))
    blink_score=to_score(blink_rate_proxy,*P['norm']['blink_rate_proxy'],small_is_good=True)

    # 업그레이드된 졸음 score
    segment_sleepy = any(sleepy_flags)
    perclos_score = 1.0 - (1.0 if segment_sleepy else 0.0)

    final_score = (
            W['phone_use']*phone_use_score +
            W['seat_depart']*seat_depart_score +
            W['head_angle']*head_angle_score +
            W['shoulder_tilt']*shoulder_tilt_score +
            W['hand_face']*hand_face_score +
            W['blink']*blink_score +
            W['perclos']*perclos_score
    )

    results.append({
      'segment_idx': seg_idx,
      'scores': {
        'phone_use': round(phone_use_score,4),
        'seat_depart': round(seat_depart_score,4),
        'head_angle': round(head_angle_score,4),
        'shoulder_tilt': round(shoulder_tilt_score,4),
        'hand_face': round(hand_face_score,4),
        'blink': round(blink_score,4),
        'perclos': round(perclos_score,4),
        'final': round(final_score,4)
      },
      'raw': {
        'usage_ratio': round(usage_ratio,4),
        'sleepy_flag': int(segment_sleepy),
        'blink_rate_proxy': round(blink_rate_proxy,5),
        'EAR_baseline': round(EAR_baseline,4)
      },
      'flags': {
        'seat_departed': seat_departed
      }
    })

  return results


# 개인화/적용

# ========================================
# 1) 초기 안정 구간 기반 개인화 Baseline 계산
# ========================================

def compute_personal_baseline_stability_filtered(
        segment_results,
        warmup_segments=120   # 5초 세그먼트 기준 약 10분 (5초 × 120 = 600초 = 10분)
):
  """
  본 연구에서는 baseline을 단순히 초반 10분의 평균으로 정의하지 않고,
  **초기 구간에서 안정적 집중 상태(Stability-filtered segment)**만 추출하여
  posture·gaze baseline을 정교하게 구축한다.

  안정 구간 추출 기준:
      - 과도한 시선 흔들림 (gaze jitter)
      - 지나치게 높은 blink spike
      - 과도한 졸음 (sleepy_flag)
      - phone 사용 의심 (phone_use 점수 낮음)
      - seat departure (자리 이탈)
      - 과도한 자세 변화
  """

  warm = segment_results[:min(warmup_segments, len(segment_results))]

  head_vals = []
  shoulder_vals = []
  hand_face_vals = []
  blink_vals = []
  phone_vals = []

  for seg in warm:
    sc  = seg['scores']
    raw = seg['raw']
    flg = seg['flags']

    # ---------- Stability Filter ----------
    # 자리 이탈한 세그먼트 제외
    if flg['seat_departed']:
      continue

    # blink spike (과도한 깜빡임)
    if raw['blink_rate_proxy'] > 0.25:
      continue

    # 졸음 구간 제거 (sleepy_flag == 1)
    if raw['sleepy_flag'] == 1:
      continue

    # 손-얼굴 거리 너무 가까움 (폰 사용 의심)
    if sc['hand_face'] < 0.25:
      continue

    # 폰 사용 점수가 너무 낮음 (폰 사용 중)
    if sc['phone_use'] < 0.5:
      continue

    # 고개가 과도하게 기울어진 구간
    if sc['head_angle'] < 0.3:  # 점수가 낮을수록 기울어진 것
      continue

    # 어깨 틀어짐 과도
    if sc['shoulder_tilt'] < 0.3:
      continue
    # -------------------------------------

    head_vals.append(sc['head_angle'])
    shoulder_vals.append(sc['shoulder_tilt'])
    hand_face_vals.append(sc['hand_face'])
    blink_vals.append(raw['blink_rate_proxy'])
    phone_vals.append(sc['phone_use'])

  def safe_stats(arr, default_mean):
    if len(arr) == 0:
      return default_mean, 1e-4
    arr = np.array(arr, dtype=np.float32)
    return float(arr.mean()), float(arr.std() + 1e-4)

  baseline = {
    'head_mean':     safe_stats(head_vals,     0.7)[0],   # 점수 기준 (높을수록 좋음)
    'head_std':      safe_stats(head_vals,     0.7)[1],
    'shoulder_mean': safe_stats(shoulder_vals, 0.7)[0],
    'shoulder_std':  safe_stats(shoulder_vals, 0.7)[1],
    'hand_mean':     safe_stats(hand_face_vals, 0.5)[0],
    'hand_std':      safe_stats(hand_face_vals, 0.5)[1],
    'blink_mean':    safe_stats(blink_vals,    0.05)[0],
    'blink_std':     safe_stats(blink_vals,    0.05)[1],
    'phone_mean':    safe_stats(phone_vals,    0.9)[0],
    'phone_std':     safe_stats(phone_vals,    0.9)[1],
    'num_stable_segments': len(head_vals)  # 안정 구간 개수 기록
  }

  return baseline


# ========================================
# 2) Baseline 대비 변화량을 이용한 개인화 적용
# ========================================

def apply_personalization_with_delta(segment_results, baseline):
  """
  Baseline 대비 변화량(delta)을 활용해 개인화 집중도를 계산.
  변화량이 작을수록(=baseline과 비슷할수록) 집중도가 높다고 판단.

  z = |x - μ| / σ 기반 안정성 점수:
      stability = exp( - z )

  최종 개인화 점수 = 기존 final * 0.6 + 개인화 안정성 * 0.4
  """

  personalized = []

  for seg in segment_results:
    sc  = seg['scores']
    raw = seg['raw']
    flg = seg['flags']

    def z_score(x, mu, sigma):
      return abs(x - mu) / sigma

    def stability(x, mu, sigma):
      z = z_score(x, mu, sigma)
      return float(np.exp(-z))  # z 클수록 빠르게 감점

    # 각 feature별 안정성 점수 계산
    st_head    = stability(sc['head_angle'], baseline['head_mean'], baseline['head_std'])
    st_shoul   = stability(sc['shoulder_tilt'], baseline['shoulder_mean'], baseline['shoulder_std'])
    st_hand    = stability(sc['hand_face'], baseline['hand_mean'], baseline['hand_std'])
    st_blink   = stability(raw['blink_rate_proxy'], baseline['blink_mean'], baseline['blink_std'])
    st_phone   = stability(sc['phone_use'], baseline['phone_mean'], baseline['phone_std'])

    # 개인화 안정성 종합 점수 (가중 평균)
    personal_stability = (
            0.25 * st_head +
            0.25 * st_shoul +
            0.20 * st_hand +
            0.15 * st_blink +
            0.15 * st_phone
    )

    # 자리 이탈 시 패널티
    if flg['seat_departed']:
      personal_stability *= 0.3  # 70% 감점

    # 졸음 감지 시 패널티
    if raw['sleepy_flag'] == 1:
      personal_stability *= 0.5  # 50% 감점

    # 최종 개인화 점수 (기존 점수와 개인화 안정성 혼합)
    final_personal = 0.6 * sc['final'] + 0.4 * personal_stability

    # 결과 복사 및 추가
    seg_new = seg.copy()
    seg_new['scores'] = seg['scores'].copy()
    seg_new['scores']['personalized'] = round(final_personal, 4)
    seg_new['scores']['personal_stability'] = round(personal_stability, 4)
    personalized.append(seg_new)

  return personalized


# ========================================
# 3) 실행 예시
# ========================================

# Step 1: 7-feature segment 점수 계산
t1 = time.time()
seg_results = compute_segment_scores_7features(
  processed_frames=processed_frames,
  pose_keypoints=pose_keypoints,
  eyes_landmarks=eyes_landmarks,
  face_bboxes=face_bboxes,
  frame_sizes=frame_sizes,
  segment_size=segment_size,
  phone_bboxes=None,  # YOLO 등으로 폰 검출 결과 넣을 수 있음
  params={
    'time_decay_lambda': 0.08,
  }
)
print(f"[점수 계산] {time.time() - t1:.1f}초")

print(f"총 {len(seg_results)}개 세그먼트 계산 완료")

# Step 2: 안정 구간 기반 개인화 baseline 계산
baseline = compute_personal_baseline_stability_filtered(
  segment_results=seg_results,
  warmup_segments=120    # 10분 기반 (120 segments × 5초 = 600초)
)

print("\n=== 개인화 Baseline ===")
for key, val in baseline.items():
  print(f"{key:20s}: {val:.4f}")

print(f"\n안정 구간 개수: {baseline['num_stable_segments']} / {min(120, len(seg_results))}")

# Step 3: Baseline 적용하여 개인화 점수 계산
personalized_segments = apply_personalization_with_delta(
  segment_results=seg_results,
  baseline=baseline
)

# Step 4: 결과 확인
print("\n=== 첫 10개 세그먼트 점수 비교 ===")
print("Seg | Original | Personalized | Stability | Sleepy")
print("-" * 60)
for i, p in enumerate(personalized_segments[:10]):
  print(f"{i:3d} | {p['scores']['final']:8.3f} | "
        f"{p['scores']['personalized']:12.3f} | "
        f"{p['scores']['personal_stability']:9.3f} | "
        f"{p['raw']['sleepy_flag']:6d}")

# Step 5: CSV 저장 (기존 수정 코드와 동일한 라벨 구조)
rows = []
for r in personalized_segments:
  s = r['scores']
  raw = r['raw']
  flg = r['flags']

  # 기존 수정 코드의 CSV 라벨과 동일하게 구성
  row = {
    'segment_idx': r['segment_idx'],
    'phone_use': s['phone_use'],
    'seat_depart': s['seat_depart'],
    'head_angle': s['head_angle'],
    'shoulder_tilt': s['shoulder_tilt'],
    'hand_face': s['hand_face'],
    'blink': s['blink'],
    'perclos': s['perclos'],
    'final': s['final'],
    'personalized': s['personalized'],           # 개인화 점수 추가
    'personal_stability': s['personal_stability'], # 개인화 안정성 점수 추가
    'usage_ratio': raw['usage_ratio'],
    'sleepy_flag': raw['sleepy_flag'],
    'blink_rate_proxy': raw['blink_rate_proxy'],
    'EAR_baseline': raw['EAR_baseline'],
    'seat_departed': flg['seat_departed']
  }
  rows.append(row)

df = pd.DataFrame(rows).round(3)
print(f"\n=== 첫 10개 세그먼트 데이터 ===")
print(df.head(10).to_string(index=False))

print(f"\n=== 전체 통계 ===")
print(df[['final', 'personalized', 'personal_stability']].describe().round(3))

# CSV 저장
output_path = "/content/drive/MyDrive/ai_study_flow_tracker/labels_personalized_video_30.csv"
df.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"\n결과 저장 완료: {output_path}")
print(f"총 전처리 파이프라인: {time.time() - t0:.1f}초")