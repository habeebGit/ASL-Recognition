#!/usr/bin/env python3
"""
Evaluate a CTC model on a prepared val .npz file (from prepare_ctc_inputs.py).

Produces average CER and WER and prints per-sample comparisons.

Usage:
  python3 evaluate_ctc.py --val dataset/val_ctc.npz --vocab dataset/vocab.json --model trained_model --batch 8

If --model points to a SavedModel directory, it will be loaded. If omitted,
a fresh asl_model.ASLModel will be constructed (random weights) to allow
pipeline testing.
"""
import argparse
import json
import numpy as np
from pathlib import Path

# TensorFlow import may crash on macOS native TF builds; use container if needed.
import tensorflow as tf

# simple Levenshtein distance
def levenshtein(a, b):
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur[j] = min(ins, delete, sub)
        prev = cur
    return prev[-1]

def cer(ref, hyp):
    r = list(ref)
    h = list(hyp)
    d = levenshtein(r, h)
    return d / max(1, len(r))

def wer(ref, hyp):
    r = ref.split()
    h = hyp.split()
    d = levenshtein(r, h)
    return d / max(1, len(r))

def load_npz(path):
    d = np.load(path)
    return d["features"], d["input_lengths"], d["labels"], d["label_lengths"]

def indices_to_text(idxs, idx_to_char):
    # idx_to_char: mapping of int -> char (keys may be ints or strings)
    out = []
    for i in idxs:
        if i is None:
            continue
        # try direct
        if int(i) in idx_to_char:
            out.append(idx_to_char[int(i)])
            continue
        # try 1-based to 0-based shift
        if int(i) - 1 in idx_to_char:
            out.append(idx_to_char[int(i) - 1])
            continue
        # skip unknown / blank
    return "".join(out)

def decode_batch_greedy(logits, input_lengths, idx_to_char):
    # logits: [B, T, C] softmax/logits
    # ctc_decode expects predictions shape (batch, time, classes)
    # we use keras ctc_decode which returns a list of decoded tensors
    batch_size = logits.shape[0]
    # ensure float32
    preds = tf.convert_to_tensor(logits, dtype=tf.float32)
    seq_len = tf.convert_to_tensor(input_lengths, dtype=tf.int32)
    decoded, _ = tf.keras.backend.ctc_decode(preds, seq_len, greedy=True)
    dec = decoded[0]  # [B, D]
    dec_np = tf.keras.backend.get_value(dec)
    texts = []
    for i in range(batch_size):
        seq = [int(x) for x in dec_np[i] if x != -1 and x != 0]  # filter padding (-1) and zeros (if padding)
        # try to map sequence to chars (handle 1-based or 0-based)
        # prefer mapping where keys match
        if len(idx_to_char) == 0:
            texts.append("")
            continue
        # build flexible mapping function:
        s = []
        for idv in seq:
            if idv in idx_to_char:
                s.append(idx_to_char[idv])
            elif (idv - 1) in idx_to_char:
                s.append(idx_to_char[idv - 1])
            # else skip
        texts.append("".join(s))
    return texts

def pad_to_max(arr, max_t):
    out = np.zeros((arr.shape[0], max_t, arr.shape[2]), dtype=np.float32)
    out[:, :arr.shape[1], :] = arr
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--val", default="dataset/val_ctc.npz", help="Path to val ctc npz")
    p.add_argument("--vocab", default="dataset/vocab.json", help="Path to vocab.json (char->idx)")
    p.add_argument("--model", default="", help="Path to SavedModel dir (optional)")
    p.add_argument("--batch", type=int, default=8)
    args = p.parse_args()

    val_path = Path(args.val)
    if not val_path.exists():
        raise SystemExit(f"Val npz not found: {val_path}")

    # load vocab
    vocab_map = {}
    if Path(args.vocab).exists():
        with open(args.vocab) as fh:
            char_to_idx = json.load(fh)
        # invert mapping (note: char_to_idx uses strings as keys in JSON)
        idx_to_char = {}
        for ch, idx in char_to_idx.items():
            idx_to_char[int(idx)] = ch
    else:
        idx_to_char = {}

    feats, input_lengths, labels, label_lengths = load_npz(val_path)
    B, T, F = feats.shape
    print(f"Loaded val: samples={B}, max_time={T}, feat_dim={F}")

    # load model
    model = None
    if args.model:
        model_path = Path(args.model)
        if model_path.exists():
            try:
                model = tf.keras.models.load_model(str(model_path))
                print("Loaded SavedModel from", model_path)
            except Exception as e:
                print("Failed to load SavedModel:", e)
    if model is None:
        # fallback: try to import ASLModel and build model with matching feature dim and vocab size
        try:
            from asl_model import ASLModel
            vocab_size = max(idx_to_char.keys()) if idx_to_char else 28
            m = ASLModel(num_features=F)
            m.build_model(vocab_size=vocab_size)
            model = m.model
            print("Built ASLModel (random weights) as fallback")
        except Exception as e:
            raise SystemExit("No model available and failed to build fallback ASLModel: " + str(e))

    # evaluation loop (batch)
    tot_cer = 0.0
    tot_wer = 0.0
    n = 0
    ds = []
    # create batches from arrays to avoid tf.data overhead here
    for i in range(0, B, args.batch):
        batch_feats = feats[i:i+args.batch]
        batch_input_lengths = input_lengths[i:i+args.batch]
        batch_labels = labels[i:i+args.batch]
        batch_label_lengths = label_lengths[i:i+args.batch]

        # model.predict expects shape [B, T, F]
        preds = model.predict(batch_feats, verbose=0)
        # if preds are logits, apply softmax
        if preds.dtype != np.float32:
            preds = preds.astype(np.float32)
        # ensure shape matches [B, T, C]
        if preds.shape[1] != batch_feats.shape[1]:
            # if model downsampled time, try to expand by repeating last frame
            if preds.shape[1] < batch_feats.shape[1]:
                pad_t = batch_feats.shape[1] - preds.shape[1]
                last = np.repeat(preds[:, -1:, :], pad_t, axis=1)
                preds = np.concatenate([preds, last], axis=1)
            else:
                preds = preds[:, :batch_feats.shape[1], :]

        # decode greedy
        decoded_texts = decode_batch_greedy(preds, batch_input_lengths, idx_to_char)

        # convert labels to text
        for j in range(batch_feats.shape[0]):
            lab_len = int(batch_label_lengths[j])
            lab_seq = [int(x) for x in batch_labels[j][:lab_len] if int(x) != 0]
            # map lab_seq to chars
            lab_chars = []
            for idv in lab_seq:
                if idv in idx_to_char:
                    lab_chars.append(idx_to_char[idv])
                elif (idv - 1) in idx_to_char:
                    lab_chars.append(idx_to_char[idv - 1])
            ref = "".join(lab_chars)
            hyp = decoded_texts[j]
            sample_cer = cer(ref, hyp)
            sample_wer = wer(ref, hyp)
            tot_cer += sample_cer
            tot_wer += sample_wer
            n += 1
            print(f"REF: '{ref}'")
            print(f"HYP: '{hyp}'")
            print(f"CER={sample_cer:.3f} WER={sample_wer:.3f}")
            print("-" * 30)

    if n == 0:
        print("No samples evaluated.")
        return
    print(f"Avg CER: {tot_cer / n:.4f}, Avg WER: {tot_wer / n:.4f} over {n} samples")

if __name__ == "__main__":
    main()