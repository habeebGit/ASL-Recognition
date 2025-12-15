#!/usr/bin/env python3
"""
Extract pose + hand keypoints from video files using MediaPipe Holistic and
save per-video NumPy sequences (.npy). Each file becomes an array of shape
(T, F) where T is frames and F is flattened keypoint vector.

Usage:
    python3 preprocess_keypoints.py --input video.mp4 --out-dir keypoints/

Dependencies:
    pip install mediapipe opencv-python numpy

This script is intended for prototyping. For production pipelines, handle
frame-rate normalization, smoothing, and missing-landmark handling more
carefully.
"""
import os
import argparse
import numpy as np

try:
    import cv2
    import mediapipe as mp
except Exception as e:
    raise ImportError("Missing dependency: install 'mediapipe' and 'opencv-python'.") from e

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def extract_keypoints_from_frame(results):
    """Return a 1D numpy vector of selected landmarks for one frame.
    We collect pose (33), left_hand (21), right_hand (21).
    Each landmark contributes (x, y, z) when available; missing -> 0.
    Final vector length = (33 + 21 + 21) * 3 = 225
    """
    pose = results.pose_landmarks.landmark if results.pose_landmarks else []
    lh = results.left_hand_landmarks.landmark if results.left_hand_landmarks else []
    rh = results.right_hand_landmarks.landmark if results.right_hand_landmarks else []

    def lm_to_list(landmarks, count):
        out = []
        for i in range(count):
            if i < len(landmarks):
                l = landmarks[i]
                out.extend([l.x, l.y, l.z])
            else:
                out.extend([0.0, 0.0, 0.0])
        return out

    vec = []
    vec.extend(lm_to_list(pose, 33))
    vec.extend(lm_to_list(lh, 21))
    vec.extend(lm_to_list(rh, 21))
    return np.array(vec, dtype=np.float32)


def normalize_sequence(seq):
    """Basic normalization: subtract median torso (pose landmark 0..11 contains upper body)
    and scale by shoulder distance when available.
    seq: (T, F)
    """
    if seq.size == 0:
        return seq
    T, F = seq.shape
    reshaped = seq.reshape(T, -1, 3)  # (T, landmarks, 3)
    # Use pose landmark 11 (right_shoulder) and 12 (left_shoulder) if present
    shoulders = reshaped[:, :33, :2][:, [11, 12], :]
    # compute center as mean of available shoulders per frame
    centers = np.nanmean(shoulders.reshape(T, -1, 2), axis=1)
    # fallback if NaN
    centers = np.nan_to_num(centers)
    # subtract center from x,y; keep z unchanged
    for t in range(T):
        for i in range(reshaped.shape[1]):
            reshaped[t, i, 0:2] -= centers[t]
    # scale by shoulder distance mean
    try:
        p11 = reshaped[:, 11, :2]
        p12 = reshaped[:, 12, :2]
        d = np.linalg.norm(p11 - p12, axis=1)
        scale = np.mean(d[d > 0.0])
        if not np.isfinite(scale) or scale == 0.0:
            scale = 1.0
    except Exception:
        scale = 1.0
    reshaped[:, :, 0:2] /= scale
    return reshaped.reshape(T, F)


def process_video(input_path, out_dir, max_frames=None, show=False):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {input_path}")

    seq = []
    with mp_holistic.Holistic(static_image_mode=False, model_complexity=1) as holistic:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            vec = extract_keypoints_from_frame(results)
            seq.append(vec)
            if show:
                annotated = frame.copy()
                mp_drawing.draw_landmarks(
                    annotated, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                mp_drawing.draw_landmarks(
                    annotated, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(
                    annotated, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                cv2.imshow('preview', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if max_frames and frame_idx >= max_frames:
                break
    cap.release()
    if show:
        cv2.destroyAllWindows()
    if len(seq) == 0:
        return None
    arr = np.stack(seq, axis=0)
    arr = normalize_sequence(arr)
    base = os.path.splitext(os.path.basename(input_path))[0]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, base + '.npy')
    np.save(out_path, arr)
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', required=True, help='Input video file or directory')
    p.add_argument('--out-dir', '-o', default='keypoints', help='Output directory for .npy sequences')
    p.add_argument('--max-frames', type=int, default=None)
    p.add_argument('--show', action='store_true')
    args = p.parse_args()

    inputs = []
    if os.path.isdir(args.input):
        for fn in sorted(os.listdir(args.input)):
            if fn.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                inputs.append(os.path.join(args.input, fn))
    else:
        inputs = [args.input]

    for vid in inputs:
        print(f"Processing {vid} ...")
        out = process_video(vid, args.out_dir, max_frames=args.max_frames, show=args.show)
        if out:
            print(f"Wrote {out}")
        else:
            print(f"No landmarks found for {vid}")


if __name__ == '__main__':
    main()
