#!/usr/bin/env python3
"""
Quick training job for ASLModel on the prepared synthetic CTC .npz files.

Usage (from project root):
  python3 train_asl_on_synthetic.py --train dataset/train_ctc.npz --val dataset/val_ctc.npz --vocab dataset/vocab.json --out trained_asl_model --epochs 5 --batch 4

This script is intentionally small / demo-only: trains a few epochs and saves a SavedModel.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from asl_model import ASLModel

def load_npz(path):
    d = np.load(path)
    return d["features"].astype(np.float32), d["labels"].astype(np.int32), d["input_lengths"].astype(np.int32), d["label_lengths"].astype(np.int32)

def make_tf_dataset(features, labels, batch_size=8, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((features, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=max(16, len(features)))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="dataset/train_ctc.npz")
    p.add_argument("--val", default="dataset/val_ctc.npz")
    p.add_argument("--vocab", default="dataset/vocab.json")
    p.add_argument("--out", default="trained_asl_model")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=4)
    args = p.parse_args()

    train_p = Path(args.train)
    if not train_p.exists():
        raise SystemExit(f"Train NPZ not found: {train_p}. Run prepare_ctc_inputs.py first.")

    feats_tr, labels_tr, in_l_tr, lab_l_tr = load_npz(str(train_p))
    print("Train:", feats_tr.shape, labels_tr.shape)

    val_exists = False
    if Path(args.val).exists():
        feats_val, labels_val, in_l_val, lab_l_val = load_npz(str(args.val))
        val_exists = True
        print("Val:", feats_val.shape, labels_val.shape)
    else:
        feats_val, labels_val = None, None
        print("Val NPZ not found, training without validation.")

    # load vocab size
    if Path(args.vocab).exists():
        with open(args.vocab) as fh:
            char_to_idx = json.load(fh)
        vocab_size = len(char_to_idx)
    else:
        # fallback: infer max id in labels + 1
        vocab_size = int(np.max(labels_tr)) if labels_tr.size else 28
    print("Vocab size:", vocab_size)

    # build model
    feature_dim = feats_tr.shape[2]
    model_wrapper = ASLModel(num_features=feature_dim)
    model = model_wrapper.build_model(vocab_size=vocab_size)
    model_wrapper.compile_model(lr=1e-3)

    # prepare datasets
    train_ds = make_tf_dataset(feats_tr, labels_tr, batch_size=args.batch, shuffle=True)
    val_ds = make_tf_dataset(feats_val, labels_val, batch_size=args.batch, shuffle=False) if val_exists else None

    # fit
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(args.out + "/ckpt", save_weights_only=False, save_best_only=False),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks, verbose=2)

    # save final model
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir))
    print("Saved trained model to", out_dir)

if __name__ == "__main__":
    main()
```# filepath: /Users/habeebmohammed/software/model-training/voice2text/train_asl_on_synthetic.py
#!/usr/bin/env python3
"""
Quick training job for ASLModel on the prepared synthetic CTC .npz files.

Usage (from project root):
  python3 train_asl_on_synthetic.py --train dataset/train_ctc.npz --val dataset/val_ctc.npz --vocab dataset/vocab.json --out trained_asl_model --epochs 5 --batch 4

This script is intentionally small / demo-only: trains a few epochs and saves a SavedModel.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from asl_model import ASLModel

def load_npz(path):
    d = np.load(path)
    return d["features"].astype(np.float32), d["labels"].astype(np.int32), d["input_lengths"].astype(np.int32), d["label_lengths"].astype(np.int32)

def make_tf_dataset(features, labels, batch_size=8, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((features, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=max(16, len(features)))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="dataset/train_ctc.npz")
    p.add_argument("--val", default="dataset/val_ctc.npz")
    p.add_argument("--vocab", default="dataset/vocab.json")
    p.add_argument("--out", default="trained_asl_model")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=4)
    args = p.parse_args()

    train_p = Path(args.train)
    if not train_p.exists():
        raise SystemExit(f"Train NPZ not found: {train_p}. Run prepare_ctc_inputs.py first.")

    feats_tr, labels_tr, in_l_tr, lab_l_tr = load_npz(str(train_p))
    print("Train:", feats_tr.shape, labels_tr.shape)

    val_exists = False
    if Path(args.val).exists():
        feats_val, labels_val, in_l_val, lab_l_val = load_npz(str(args.val))
        val_exists = True
        print("Val:", feats_val.shape, labels_val.shape)
    else:
        feats_val, labels_val = None, None
        print("Val NPZ not found, training without validation.")

    # load vocab size
    if Path(args.vocab).exists():
        with open(args.vocab) as fh:
            char_to_idx = json.load(fh)
        vocab_size = len(char_to_idx)
    else:
        # fallback: infer max id in labels + 1
        vocab_size = int(np.max(labels_tr)) if labels_tr.size else 28
    print("Vocab size:", vocab_size)

    # build model
    feature_dim = feats_tr.shape[2]
    model_wrapper = ASLModel(num_features=feature_dim)
    model = model_wrapper.build_model(vocab_size=vocab_size)
    model_wrapper.compile_model(lr=1e-3)

    # prepare datasets
    train_ds = make_tf_dataset(feats_tr, labels_tr, batch_size=args.batch, shuffle=True)
    val_ds = make_tf_dataset(feats_val, labels_val, batch_size=args.batch, shuffle=False) if val_exists else None

    # fit
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(args.out + "/ckpt", save_weights_only=False, save_best_only=False),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks, verbose=2)

    # save final model
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir))
    print("Saved trained model to", out_dir)

if __name__ == "__main__":