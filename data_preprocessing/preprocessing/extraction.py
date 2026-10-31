"""영상 → 프레임별 랜드마크 추출"""

import os
import time

import cv2
import mediapipe as mp
import numpy as np

from .config import FACE_MODEL_PATH, POSE_MODEL_PATH, TARGET_FPS, PHONE_DETECT_INTERVAL_SEC, PHONE_MODEL_ID, PHONE_SCORE_THRESH
from .features import LEFT_EYE_IDXS, RIGHT_EYE_IDXS

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
PoseLandmarker = mp.tasks.vision.PoseLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

FRAME_KEYS = ("processed_frames", "face_bboxes", "frame_sizes", "eyes_landmarks", "pose_keypoints")


def _build_options(face_model_path, pose_model_path):
    face_opts = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=face_model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
    )
    pose_opts = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=pose_model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
    )
    return face_opts, pose_opts


def _append_empty(frames):
    """검출 실패 프레임: 모든 리스트에 None을 넣어 타임라인 길이를 맞춘다."""
    for key in FRAME_KEYS:
        frames[key].append(None)


def _process_sampled_frame(frame_bgr, timestamp_ms, mp_face, mp_pose, frames, timing,
                           phone_detector=None, check_phone=False):
    """샘플링된 프레임 1개에 대해 face/pose 랜드마크를 추출해 frames에 append."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = frame_bgr.shape

    # --- Phone (얼굴 검출과 무관하게 먼저 수행) ---
    if phone_detector is not None and check_phone:
        _tph = time.time()
        frames["phone_scores"].append(phone_detector.max_score(frame_rgb))
        timing["phone"] += time.time() - _tph
    else:
        frames["phone_scores"].append(None)   # 검사 안 함

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # --- Face ---
    _tf = time.time()
    res_face = mp_face.detect_for_video(mp_image, timestamp_ms)
    timing["face"] += time.time() - _tf

    if not res_face.face_landmarks:
        _append_empty(frames)
        return

    lm = res_face.face_landmarks[0]
    x_coords = np.array([p.x for p in lm])
    y_coords = np.array([p.y for p in lm])
    xmin, xmax = int(x_coords.min() * w), int(x_coords.max() * w)
    ymin, ymax = int(y_coords.min() * h), int(y_coords.max() * h)
    xmin, ymin = max(0, xmin), max(0, ymin)
    xmax, ymax = min(w - 1, xmax), min(h - 1, ymax)
    if xmax <= xmin or ymax <= ymin:
        _append_empty(frames)
        return

    face_crop = frame_bgr[ymin:ymax, xmin:xmax]
    if face_crop.size == 0:
        _append_empty(frames)
        return

    frames["face_bboxes"].append((xmin, ymin, xmax, ymax))
    frames["frame_sizes"].append((w, h))
    frames["processed_frames"].append(True)

    # --- Eyes ---
    lm_np = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)
    try:
        eyeL = lm_np[LEFT_EYE_IDXS, :]
        eyeR = lm_np[RIGHT_EYE_IDXS, :]
        frames["eyes_landmarks"].append((eyeL, eyeR))
    except Exception:
        frames["eyes_landmarks"].append(None)

    # --- Pose ---
    _tp = time.time()
    res_pose = mp_pose.detect_for_video(mp_image, timestamp_ms)
    timing["pose"] += time.time() - _tp

    if res_pose.pose_landmarks:
        vec = []
        for lm_p in res_pose.pose_landmarks[0]:
            # 모든 랜드마크가 걸러지는 것 방지 위해 1.0 부여
            vis = lm_p.visibility if lm_p.visibility is not None else 1.0
            vec.extend([lm_p.x, lm_p.y, lm_p.z, vis])  # 33 x 4 (x, y, z, visibility)
        frames["pose_keypoints"].append(vec)
    else:
        frames["pose_keypoints"].append(None)


def extract_landmarks(
        video_path,
        target_fps=TARGET_FPS,
        face_model_path=FACE_MODEL_PATH,
        pose_model_path=POSE_MODEL_PATH,
        phone_detector=None,
):
    """영상을 target_fps로 샘플링하며 프레임별 랜드마크를 추출한다.

    반환되는 5개 리스트의 길이는 항상 샘플 수(num_frames)와 같고,
    검출 실패 프레임은 None으로 채워진다.

    Returns:
        dict: processed_frames, face_bboxes, frame_sizes, eyes_landmarks,
              pose_keypoints, num_frames, native_fps,
              timing{total, face, pose, skip}
    """
    for path in (video_path, face_model_path, pose_model_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} 존재하지 않음")

    face_opts, pose_opts = _build_options(face_model_path, pose_model_path)
    frames = {key: [] for key in FRAME_KEYS}
    frames["phone_scores"] = []
    timing = {"total": 0.0, "face": 0.0, "pose": 0.0, "skip": 0.0, "phone": 0.0}
    phone_interval = max(1, round(PHONE_DETECT_INTERVAL_SEC * target_fps))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"영상을 열 수 없음: {video_path}")

    try:
        native_fps = cap.get(cv2.CAP_PROP_FPS)
        if not native_fps or native_fps <= 0:
            raise ValueError(f"FPS를 읽을 수 없음: {video_path}")
        frame_interval = max(1, round(native_fps / target_fps))

        # VIDEO 모드 landmarker는 타임스탬프가 단조 증가해야 하므로 영상마다 새로 생성
        with FaceLandmarker.create_from_options(face_opts) as mp_face, \
                PoseLandmarker.create_from_options(pose_opts) as mp_pose:

            t0 = time.time()
            frame_idx = 0

            # grab()/retrieve() 분리: 분석할 프레임만 디코딩
            while True:
                if frame_idx % frame_interval == 0:
                    if not cap.grab():
                        break
                    ret, frame_bgr = cap.retrieve()
                    if not ret:
                        break
                    frame_idx += 1

                    t_sec = (frame_idx - 1) / native_fps
                    timestamp_ms = int(t_sec * 1000)
                    sample_idx = len(frames["processed_frames"])
                    _process_sampled_frame(
                        frame_bgr, timestamp_ms, mp_face, mp_pose, frames, timing,
                        phone_detector=phone_detector,
                        check_phone=(sample_idx % phone_interval == 0),
                    )
                else:
                    _ts = time.time()
                    ret = cap.grab()
                    timing["skip"] += time.time() - _ts
                    if not ret:
                        break
                    frame_idx += 1

            timing["total"] = time.time() - t0
    finally:
        cap.release()

    return {
        **frames,
        "num_frames": len(frames["processed_frames"]),
        "native_fps": native_fps,
        "timing": timing,
    }


class PhoneDetector:
    """RT-DETR 기반 폰 검출기"""
    def __init__(self, model_id=PHONE_MODEL_ID, device=None, score_thresh=PHONE_SCORE_THRESH):
        import torch
        from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.score_thresh = score_thresh
        self.proc = RTDetrImageProcessor.from_pretrained(model_id)
        self.model = RTDetrForObjectDetection.from_pretrained(model_id).to(self.device).eval()
        self.phone_ids = {i for i, n in self.model.config.id2label.items() if n == "cell phone"}
        if not self.phone_ids:
            raise ValueError(f"{model_id}에 'cell phone' 클래스가 없음")

    def max_score(self, frame_rgb):
        """프레임 내 폰 최고 점수. 임계값 미만이면 0.0"""
        torch = self._torch
        h, w = frame_rgb.shape[:2]
        inputs = self.proc(images=frame_rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
        res = self.proc.post_process_object_detection(
            out, target_sizes=torch.tensor([[h, w]]), threshold=self.score_thresh)[0]
        scores = [s.item() for s, l in zip(res["scores"], res["labels"]) if l.item() in self.phone_ids]
        return max(scores, default=0.0)