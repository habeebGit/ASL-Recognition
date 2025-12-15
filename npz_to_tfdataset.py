#!/usr/bin/env python3
"""
Create tf.data.Dataset objects from the prepared CTC .npz files produced by
prepare_ctc_inputs.py.

Usage:
  python3 npz_to_tfdataset.py --train dataset/train_ctc.npz --val dataset/val_ctc.npz --batch 8

Produces batched tf.data.Dataset that yields dictionaries:
  {"features": float32 [B, T, F], "input_lengths": int32 [B]},
  {"labels": int32 [B, L], "label_lengths": int32 [B]}

These datasets can be adapted to your training loop or Keras model with a
custom training step that consumes lengths for CTC loss.
"""
from pathlib import Path
import argparse
import numpy as np
import tensorflow as tf

def load_ctc_npz(path):
    d = np.load(path)
    return {
        "features": d["features"].astype(np.float32),
        "input_lengths": d["input_lengths"].astype(np.int32),
        "labels": d["labels"].astype(np.int32),
        "label_lengths": d["label_lengths"].astype(np.int32),
    }

def _to_tf_tensors(sample):
    features, labels, input_lengths, label_lengths = sample
    return (
        {"features": tf.convert_to_tensor(features, dtype=tf.float32),
         "input_lengths": tf.convert_to_tensor(input_lengths, dtype=tf.int32)},
        {"labels": tf.convert_to_tensor(labels, dtype=tf.int32),
         "label_lengths": tf.convert_to_tensor(label_lengths, dtype=tf.int32)}
    )

def get_ctc_dataset(npz_path, batch_size=8, shuffle=True, shuffle_buffer=1024):
    data = load_ctc_npz(npz_path)
    features = data["features"]
    labels = data["labels"]
    input_lengths = data["input_lengths"]
    label_lengths = data["label_lengths"]

    ds = tf.data.Dataset.from_tensor_slices((features, labels, input_lengths, label_lengths))
    if shuffle:
        ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
    ds = ds.map(_to_tf_tensors, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def make_train_val_datasets(train_npz, val_npz=None, batch_size=8):
    train_ds = get_ctc_dataset(train_npz, batch_size=batch_size, shuffle=True)
    val_ds = None
    if val_npz and Path(val_npz).exists():
        val_ds = get_ctc_dataset(val_npz, batch_size=batch_size, shuffle=False)
    return train_ds, val_ds

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, help="Path to train_ctc.npz")
    p.add_argument("--val", default="", help="Path to val_ctc.npz (optional)")
    p.add_argument("--batch", type=int, default=8)
    args = p.parse_args()

    train_ds, val_ds = make_train_val_datasets(args.train, args.val or None, batch_size=args.batch)
    # print a single batch summary
    for batch in train_ds.take(1):
        x, y = batch
        print("features shape:", x["features"].shape)
        print("input_lengths shape:", x["input_lengths"].shape)
        print("labels shape:", y["labels"].shape)
        print("label_lengths shape:", y["label_lengths"].shape)
    if val_ds:
        for batch in val_ds.take(1):
            x, y = batch
            print("VAL features shape:", x["features"].shape)
    print("Done.")
```# filepath: /Users/habeebmohammed/software/model-training/voice2text/npz_to_tfdataset.py
#!/usr/bin/env python3
"""
Create tf.data.Dataset objects from the prepared CTC .npz files produced by
prepare_ctc_inputs.py.

Usage:
  python3 npz_to_tfdataset.py --train dataset/train_ctc.npz --val dataset/val_ctc.npz --batch 8

Produces batched tf.data.Dataset that yields dictionaries:
  {"features": float32 [B, T, F], "input_lengths": int32 [B]},
  {"labels": int32 [B, L], "label_lengths": int32 [B]}

These datasets can be adapted to your training loop or Keras model with a
custom training step that consumes lengths for CTC loss.
"""
from pathlib import Path
import argparse
import numpy as np
import tensorflow as tf

def load_ctc_npz(path):
    d = np.load(path)
    return {
        "features": d["features"].astype(np.float32),
        "input_lengths": d["input_lengths"].astype(np.int32),
        "labels": d["labels"].astype(np.int32),
        "label_lengths": d["label_lengths"].astype(np.int32),
    }

def _to_tf_tensors(sample):
    features, labels, input_lengths, label_lengths = sample
    return (
        {"features": tf.convert_to_tensor(features, dtype=tf.float32),
         "input_lengths": tf.convert_to_tensor(input_lengths, dtype=tf.int32)},
        {"labels": tf.convert_to_tensor(labels, dtype=tf.int32),
         "label_lengths": tf.convert_to_tensor(label_lengths, dtype=tf.int32)}
    )

def get_ctc_dataset(npz_path, batch_size=8, shuffle=True, shuffle_buffer=1024):
    data = load_ctc_npz(npz_path)
    features = data["features"]
    labels = data["labels"]
    input_lengths = data["input_lengths"]
    label_lengths = data["label_lengths"]

    ds = tf.data.Dataset.from_tensor_slices((features, labels, input_lengths, label_lengths))
    if shuffle:
        ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
    ds = ds.map(_to_tf_tensors, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def make_train_val_datasets(train_npz, val_npz=None, batch_size=8):
    train_ds = get_ctc_dataset(train_npz, batch_size=batch_size, shuffle=True)
    val_ds = None
    if val_npz and Path(val_npz).exists():
        val_ds = get_ctc_dataset(val_npz, batch_size=batch_size, shuffle=False)
    return train_ds, val_ds

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, help="Path to train_ctc.npz")
    p.add_argument("--val", default="", help="Path to val_ctc.npz (optional)")
    p.add_argument("--batch", type=int, default=8)
    args = p.parse_args()

    train_ds, val_ds = make_train_val_datasets(args.train, args.val or None, batch_size=args.batch)
    # print a single batch summary
    for batch in train_ds.take(1):
        x, y = batch
        print("features shape:", x["features"].shape)
        print("input_lengths shape:", x["input_lengths"].shape)
        print("labels shape:", y["labels"].shape)
        print("label_lengths shape:", y["label_lengths"].shape)
    if val_ds:
        for batch in val_ds.take(1):
            x, y = batch
            print("VAL features