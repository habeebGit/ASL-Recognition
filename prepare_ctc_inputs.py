#!/usr/bin/env python3
"""
Prepare CTC-ready NPZ files and vocab.json from dataset/labels.csv and
dataset/{train,val} .npy keypoint files.

Usage:
  python3 prepare_ctc_inputs.py --dataset-root dataset
"""
import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

def read_labels(labels_csv):
    rows = []
    with open(labels_csv, newline='') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append((r['filename'], r['transcription']))
    return rows

def build_vocab(transcriptions):
    chars = set()
    for t in transcriptions:
        chars.update(list(t))
    chars = sorted(chars)
    # indices start at 1, 0 reserved for padding
    char_to_idx = {c: i+1 for i, c in enumerate(chars)}
    return char_to_idx

def pad_features(seqs, max_len):
    feat_dim = seqs[0].shape[1] if seqs else 0
    out = np.zeros((len(seqs), max_len, feat_dim), dtype=np.float32)
    lengths = np.zeros((len(seqs),), dtype=np.int32)
    for i, s in enumerate(seqs):
        L = s.shape[0]
        lengths[i] = L
        out[i, :L] = s
    return out, lengths

def pad_labels(encoded, max_lab_len):
    out = np.zeros((len(encoded), max_lab_len), dtype=np.int32)  # padding=0
    lengths = np.zeros((len(encoded),), dtype=np.int32)
    for i, seq in enumerate(encoded):
        L = len(seq)
        lengths[i] = L
        out[i, :L] = seq
    return out, lengths

def process_split(dataset_root, split, char_to_idx):
    split_dir = Path(dataset_root) / split
    if not split_dir.exists():
        print(f"Split dir missing, skipping: {split_dir}")
        return None
    labels = read_labels(Path(dataset_root) / "labels.csv")
    rows_split = [(r, t) for r, t in labels if str(r).startswith(f"{split}/")]
    seqs = []
    labels_enc = []
    for rel_path, transcription in rows_split:
        p = Path(dataset_root) / rel_path
        if not p.exists():
            print("Missing file, skipping:", p)
            continue
        seq = np.load(p)
        seqs.append(seq.astype(np.float32))
        enc = [char_to_idx.get(c, 0) for c in transcription]  # unknown -> 0
        # remove zeros at encoding time? keep as zeros (padding)
        labels_enc.append([x for x in enc if x != 0])
    if not seqs:
        return None
    max_t = max(s.shape[0] for s in seqs)
    max_lab = max(len(l) for l in labels_enc) if labels_enc else 0
    feats_padded, input_lengths = pad_features(seqs, max_t)
    labels_padded, label_lengths = pad_labels(labels_enc, max_lab)
    return {
        "features": feats_padded,
        "input_lengths": input_lengths,
        "labels": labels_padded,
        "label_lengths": label_lengths
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", default="dataset")
    args = p.parse_args()

    ds = Path(args.dataset_root)
    labels_csv = ds / "labels.csv"
    if not labels_csv.exists():
        raise SystemExit("labels.csv not found in dataset root. Create dataset first.")

    rows = read_labels(labels_csv)
    trans = [t for _, t in rows]
    char_to_idx = build_vocab(trans)
    vocab_path = ds / "vocab.json"
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    with open(vocab_path, "w") as fh:
        json.dump(char_to_idx, fh, indent=2)
    print("Wrote vocab.json (chars -> indices), size:", len(char_to_idx))

    for split in ("train", "val"):
        out = process_split(args.dataset_root, split, char_to_idx)
        if out is None:
            print(f"No samples for split {split}, skipping.")
            continue
        out_path = ds / f"{split}_ctc.npz"
        np.savez_compressed(out_path,
                            features=out["features"],
                            input_lengths=out["input_lengths"],
                            labels=out["labels"],
                            label_lengths=out["label_lengths"])
        print(f"Wrote {out_path}: features={out['features'].shape}, labels={out['labels'].shape}")

    print("Done. CTC NPZ files and vocab.json are ready.")

if __name__ == "__main__":
    main()