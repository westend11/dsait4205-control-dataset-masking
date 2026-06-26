"""
A model with the property "attend to task-relevant
pixels, ignore the rest" should score as follows:

  Invariance  Delta = d(f(x), f(x'))   on the non_occluding split.
              A robust model -> Delta ~ 0, and Delta stays flat as the
              distractor intensity rises. On the occluding split Delta should
              rise (the negative control: the model is allowed to react there).

  Mask align  IoU(s_hat, m), where s_hat is the model's saliency/mask.
              A masking model (MaDi; Grooten et al., 2024) should cover the distractor:
              IoU -> high. A robust model should *exclude* it from its saliency:
              IoU(saliency, m) ~ 0.

Run:
    python metrics.py --data dataset
"""

import argparse
import json
import os
from collections import defaultdict
import numpy as np
from PIL import Image


# Core metrics (model-agnostic)
def invariance_delta(fx, fxp, ord=2):
    """Distance between feature vectors of x and x'. Lower = more invariant."""
    fx = np.asarray(fx, dtype=np.float64).ravel()
    fxp = np.asarray(fxp, dtype=np.float64).ravel()
    if ord == 1:
        return float(np.abs(fx - fxp).mean())
    return float(np.sqrt(((fx - fxp) ** 2).mean()))


def iou(mask_a, mask_b):
    """Intersection-over-union of two binary masks. 0 if both empty."""
    a = np.asarray(mask_a) > 0
    b = np.asarray(mask_b) > 0
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum()) / float(union)


# Trivial stand-in model (replace with the model under test).
def pixel_feature(img):
    """A stand-in feature map f: raw normalised pixels. 
    This is the non-invariant baselines that the robus model 
    should flatten.
    """
    return np.asarray(img, dtype=np.float64).ravel() / 255.0


def diff_saliency(x, x_prime, thresh=10):
    """A stand-in saliency s_hat: where x' differs from x. Used only to show the
    IoU computation; a real model exposes its own mask/saliency."""
    d = np.abs(np.asarray(x_prime, dtype=np.int16) - np.asarray(x, dtype=np.int16)).sum(axis=2)
    return (d > thresh).astype(np.uint8)


# Dataset loading + aggregate evaluation.
def load_rows(data_dir):
    with open(os.path.join(data_dir, "metadata.jsonl")) as f:
        return [json.loads(line) for line in f]


def _load(data_dir, rel):
    return np.asarray(Image.open(os.path.join(data_dir, rel)))


def evaluate(data_dir, feature_fn=pixel_feature):
    """Aggregate Delta (per split x type x intensity) and IoU(s_hat, m).
    """
    rows = load_rows(data_dir)
    # delta[split][type][level] -> list of deltas; iou likewise.
    delta = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    iou_acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    scene_cache = {}
    for row in rows:
        xp = row["x"]
        if xp not in scene_cache:
            scene_cache[xp] = _load(data_dir, xp)
        x = scene_cache[xp]
        x_prime = _load(data_dir, row["x_prime"])
        m = _load(data_dir, row["m"]) > 0

        d = invariance_delta(feature_fn(x), feature_fn(x_prime))
        s_hat = diff_saliency(x, x_prime)

        split, t, lvl = row["split"], row["type"], row["intensity_level"]
        delta[split][t][lvl].append(d)
        iou_acc[split][t][lvl].append(iou(s_hat, m))

    def _mean_tree(tree):
        return {s: {t: {lvl: float(np.mean(v)) for lvl, v in sorted(lt.items())}
                    for t, lt in st.items()}
                for s, st in tree.items()}

    return {"delta": _mean_tree(delta), "iou_saliency_vs_m": _mean_tree(iou_acc)}


def main():
    parser = argparse.ArgumentParser(description="Track A evaluation metrics")
    parser.add_argument("--data", default="dataset", help="Dataset directory.")
    args = parser.parse_args()

    info_path = os.path.join(args.data, "dataset_info.json")
    info = json.load(open(info_path)) if os.path.exists(info_path) else {"intensity": {}}
    report = evaluate(args.data)

if __name__ == "__main__":
    main()
