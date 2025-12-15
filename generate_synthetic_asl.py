#!/usr/bin/env python3
"""
Generate a tiny synthetic ASL keypoint dataset (npy files) and labels.csv.

Creates:
  dataset/
    train/
      sample_000.npy ...
    val/
      sample_000.npy ...
    labels.csv
    vocab.txt

Run:
  python3 generate_synthetic_asl.py
"""
import os
import csv
import numpy as np

OUT_DIR = "dataset"
TRAIN_DIR = os.path.join(OUT_DIR, "train")
VAL_DIR = os.path.join(OUT_DIR, "val")
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

# params
n_train = 20
n_val = 5
feature_dim = 225   # match asl_model default
min_frames = 30
max_frames = 120

# small gloss vocabulary (word-level labels for demo)
glosses = ["hello", "yes", "no", "thanks", "please", "name", "sorry", "love", "help", "stop"]

def synth_keypoint_sequence(n_frames, dim):
    # smooth random walk to mimic motion in keypoint space
    steps = np.random.normal(scale=0.5, size=(n_frames, dim)).astype(np.float32)
    seq = np.cumsum(steps, axis=0)
    # simple smoothing
    kernel = np.ones(5) / 5.0
    for d in range(dim):
        seq[:, d] = np.convolve(seq[:, d], kernel, mode="same")
    # normalize per-sequence
    seq = (seq - seq.mean(axis=0)) / (seq.std(axis=0) + 1e-6)
    return seq

def write_split(n, out_dir, prefix):
    rows = []
    for i in range(n):
        n_frames = np.random.randint(min_frames, max_frames + 1)
        seq = synth_keypoint_sequence(n_frames, feature_dim)
        fname = f"{prefix}_{i:03d}.npy"
        path = os.path.join(out_dir, fname)
        np.save(path, seq)
        transcription = np.random.choice(glosses)
        rows.append((os.path.join(os.path.basename(out_dir), fname), transcription))
    return rows

if __name__ == "__main__":
    rows = []
    rows += write_split(n_train, TRAIN_DIR, "sample")
    rows += write_split(n_val, VAL_DIR, "sample")
    # write labels.csv (filename relative to dataset folder)
    labels_path = os.path.join(OUT_DIR, "labels.csv")
    with open(labels_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename", "transcription"])
        for r in rows:
            writer.writerow(r)
    # write a simple vocab (unique tokens from glosses)
    vocab_path = os.path.join(OUT_DIR, "vocab.txt")
    with open(vocab_path, "w") as fh:
        for w in sorted(set(" ".join(glosses).split())):
            fh.write(w + "\n")
    print(f"Generated dataset: {OUT_DIR} (train={n_train}, val={n_val})")
    print(f"Labels written to: {labels_path}")