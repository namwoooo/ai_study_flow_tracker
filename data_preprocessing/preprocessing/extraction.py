"""영상 → 프레임별 랜드마크 추출"""

import os
import time

import cv2
import mediapipe as mp
import numpy as np

from .config import FACE_MODEL_PATH, POSE_MODEL_PATH, TARGET_FPS
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


def _process_sampled_frame(frame_bgr, timestamp_ms, mp_face, mp_pose, frames, timing):
    """샘플링된 프레임 1개에 대해 face/pose 랜드마크를 추출해 frames에 append."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = frame_bgr.shape
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
):
    """영상을 target_fps 간격의 시간 슬롯(기본 0.2초)마다 첫 프레임 하나씩 샘플링해 랜드마크를 추출한다.

    샘플 i는 항상 i / target_fps 초 슬롯에 대응한다. (가변 프레임레이트 영상 대응)
    프레임이 없는 슬롯과 검출 실패 샘플은 None으로 채운다.

    Returns:
        dict: processed_frames, face_bboxes, frame_sizes, eyes_landmarks, pose_keypoints,
              t_ms(샘플별 실제 프레임 시각, 빈 슬롯은 None), num_frames, native_fps,
              timing{total, face, pose, skip}
    """
    for path in (video_path, face_model_path, pose_model_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} 존재하지 않음")

    face_opts, pose_opts = _build_options(face_model_path, pose_model_path)
    frames = {key: [] for key in FRAME_KEYS}
    frames["t_ms"] = []  # 샘플별 실제 프레임 시각 (빈 슬롯은 None)
    timing = {"total": 0.0, "face": 0.0, "pose": 0.0, "skip": 0.0}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"영상을 열 수 없음: {video_path}")

    try:
        native_fps = cap.get(cv2.CAP_PROP_FPS)   # 기록용

        step_ms = 1000.0 / target_fps

        # VIDEO 모드 landmarker는 타임스탬프가 단조 증가해야 하므로 영상마다 새로 생성
        with FaceLandmarker.create_from_options(face_opts) as mp_face, \
            PoseLandmarker.create_from_options(pose_opts) as mp_pose:

            t0 = time.time()
            while True:
                _ts = time.time()
                if not cap.grab():
                    break
                t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                slot = int(t_ms // step_ms)

                # 이 0.2초 칸은 이미 채움 → 디코딩 없이 다음 프레임
                if slot < len(frames["processed_frames"]):
                    timing["skip"] += time.time() - _ts
                    continue

                # 프레임이 하나도 없던 칸은 빈 샘플로 채워 번호 = 시간 유지
                while len(frames["processed_frames"]) < slot:
                    _append_empty(frames)
                    frames["t_ms"].append(None)

                ret, frame_bgr = cap.retrieve()
                if not ret:
                    break
                frames["t_ms"].append(t_ms)
                _process_sampled_frame(frame_bgr, int(t_ms), mp_face, mp_pose, frames, timing)

            timing["total"] = time.time() - t0
    finally:
        cap.release()

    return {
        **frames,
        "num_frames": len(frames["processed_frames"]),
        "native_fps": native_fps,
        "timing": timing,
    }