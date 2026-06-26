"""DL verification: does the dataset actually *measure* the target property?

  vanilla:      trained on clean frames x only.
  robust:       trained on clean frames + their NON-occluding distracted frames x'
                (a label-preserving augmentation). 

Then, on held-out scenes, we sweep distractor intensity and measure the
invariance gap Delta = total-variation distance between the predicted class
distributions of x and x'.

Run:
    python verify_dl.py --data /tmp/ds_big --out deliverables
"""

import argparse
import json
import os
from collections import defaultdict
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_metadata(data_dir):
    with open(os.path.join(data_dir, "metadata.jsonl")) as f:
        return [json.loads(line) for line in f]


def load_img_u8(data_dir, rel):
    """Load an RGB PNG as a (3, 84, 84) uint8 tensor."""
    arr = np.asarray(Image.open(os.path.join(data_dir, rel)), dtype=np.uint8)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def to_float(x_u8):
    """uint8 (N,3,84,84) -> float in [0,1]."""
    return x_u8.float() / 255.0


class TinyCNN(nn.Module):
    def __init__(self, n_classes=8):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),   # 84 -> 42
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),  # 42 -> 21
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),  # 21 -> 11
        )
        self.head = nn.Sequential(
            nn.Flatten(),                       # keep the 11x11 grid of features
            nn.Linear(32 * 11 * 11, 128), nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


def build_datasets(data_dir, rows, train_scene_ids):
    """Return train tensors and a structured eval index.

    train_clean:    (N,3,84,84), labels (N,) 
    train_aug:      same, for non-occluding distracted frames of train scenes
    eval_index:     list of dicts per test sample with x/x' paths + split/type/
                     intensity_level, plus the per-scene clean frame and label.
    """
    train_set = set(train_scene_ids)

    # per-scene clean frame + label (load each clean scene once, stored uint8).
    scene_clean, scene_label = {}, {}
    for r in rows:
        sid = r["scene_id"]
        if sid not in scene_clean:
            scene_clean[sid] = load_img_u8(data_dir, r["x"])
            scene_label[sid] = r["y"]

    train_clean_x, train_clean_y = [], []
    for sid in sorted(train_set):
        train_clean_x.append(scene_clean[sid])
        train_clean_y.append(scene_label[sid])

    train_aug_x, train_aug_y = [], []
    eval_index = []
    for r in rows:
        sid = r["scene_id"]
        if sid in train_set:
            # robust model also trains on NON-occluding distracted frames.
            if r["split"] == "non_occluding":
                train_aug_x.append(load_img_u8(data_dir, r["x_prime"]))
                train_aug_y.append(r["y"])
        else:
            eval_index.append(r)

    train_clean = (torch.stack(train_clean_x), torch.tensor(train_clean_y))
    train_aug = (torch.stack(train_aug_x), torch.tensor(train_aug_y))
    return train_clean, train_aug, eval_index, scene_clean, scene_label


def train_model(X, Y, epochs=12, bs=128, lr=1e-3, seed=0, log=print):
    torch.manual_seed(seed)
    model = TinyCNN()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    n = X.shape[0]
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        model.train()
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(to_float(X[idx])), Y[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        if ep == 0 or (ep + 1) % 4 == 0:
            log(f"    epoch {ep+1:2d}/{epochs}  loss {tot/n:.4f}")
    model.eval()
    return model


@torch.no_grad()
def probs(model, X_u8):
    return torch.softmax(model(to_float(X_u8)), dim=1)


@torch.no_grad()
def clean_accuracy(model, scene_clean, scene_label, test_ids):
    X = torch.stack([scene_clean[s] for s in test_ids])
    Y = torch.tensor([scene_label[s] for s in test_ids])
    pred = model(to_float(X)).argmax(1)
    return (pred == Y).float().mean().item()


@torch.no_grad()
def invariance_curves(model, data_dir, eval_index, scene_clean):
    """Delta (TV distance) + x' accuracy, aggregated by (split, type, level)."""
    # cache clean probs per scene
    clean_prob = {}
    delta = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    accp = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in eval_index:
        sid = r["scene_id"]
        if sid not in clean_prob:
            clean_prob[sid] = probs(model, scene_clean[sid].unsqueeze(0))[0]
        px = clean_prob[sid]
        xp = load_img_u8(data_dir, r["x_prime"]).unsqueeze(0)
        pxp = probs(model, xp)[0]
        tv = 0.5 * (px - pxp).abs().sum().item()        # total-variation distance
        split, t, lvl = r["split"], r["type"], r["intensity_level"]
        delta[split][t][lvl].append(tv)
        accp[split][t][lvl].append(float(pxp.argmax().item() == r["y"]))

    def mean_tree(tr):
        return {s: {t: {l: float(np.mean(v)) for l, v in sorted(lt.items())}
                    for t, lt in st.items()} for s, st in tr.items()}
    return mean_tree(delta), mean_tree(accp)


def split_mean_curve(delta_tree, split, intensity_levels=5):
    """Average Delta across distractor types -> one curve per split."""
    per_level = [[] for _ in range(intensity_levels)]
    for t in delta_tree[split]:
        for lvl, v in delta_tree[split][t].items():
            per_level[lvl].append(v)
    return [float(np.mean(p)) if p else float("nan") for p in per_level]


def plot_curves(results, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    levels = list(range(5))
    titles = {"non_occluding": "non_occluding  (expect: robust stays flat)",
              "occluding": "occluding  (negative control: both rise)"}
    for ax, split in zip(axes, ["non_occluding", "occluding"]):
        for name, style in [("vanilla", "o--"), ("robust", "s-")]:
            ax.plot(levels, results[name][split], style, label=name, linewidth=2)
        ax.set_title(titles[split])
        ax.set_xlabel("distractor intensity level (0 = weakest)")
        ax.set_xticks(levels)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("invariance gap  Delta  (TV distance)")
    axes[0].legend()
    fig.suptitle("Dataset as an instrument: invariance vs. distractor intensity", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"[plot] wrote {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/tmp/ds_big")
    ap.add_argument("--out", default="deliverables")
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)

    rows = load_metadata(args.data)
    scene_ids = sorted({r["scene_id"] for r in rows})
    n_test = max(1, len(scene_ids) // 5)
    test_ids = scene_ids[-n_test:]
    train_ids = scene_ids[:-n_test]
    print(f"[data] {len(scene_ids)} scenes -> {len(train_ids)} train / {len(test_ids)} test")

    train_clean, train_aug, eval_index, scene_clean, scene_label = build_datasets(
        args.data, rows, train_ids)
    # Robust training set = clean + non-occluding augmented.
    aug_X = torch.cat([train_clean[0], train_aug[0]])
    aug_Y = torch.cat([train_clean[1], train_aug[1]])
    print(f"[data] vanilla train {train_clean[0].shape[0]} | "
          f"robust train {aug_X.shape[0]} | eval samples {len(eval_index)}")

    print("[train] vanilla (clean only)")
    vanilla = train_model(*train_clean, epochs=args.epochs, seed=0)
    print("[train] robust (clean + non-occluding distractors)")
    robust = train_model(aug_X, aug_Y, epochs=args.epochs, seed=0)

    results, accents = {}, {}
    summary = {}
    for name, model in [("vanilla", vanilla), ("robust", robust)]:
        acc = clean_accuracy(model, scene_clean, scene_label, test_ids)
        delta_tree, accp_tree = invariance_curves(model, args.data, eval_index, scene_clean)
        results[name] = {
            "non_occluding": split_mean_curve(delta_tree, "non_occluding"),
            "occluding": split_mean_curve(delta_tree, "occluding"),
        }
        summary[name] = {"clean_acc": acc,
                         "delta_non_occluding": results[name]["non_occluding"],
                         "delta_occluding": results[name]["occluding"],
                         "delta_tree": delta_tree, "acc_xprime_tree": accp_tree}
        print(f"[eval] {name}: clean test acc = {acc:.3f}")
        print(f"       Delta non_occluding by level = "
              f"{[round(v,3) for v in results[name]['non_occluding']]}")
        print(f"       Delta occluding     by level = "
              f"{[round(v,3) for v in results[name]['occluding']]}")

    plot_curves(results, os.path.join(args.out, "verify_dl_curves.png"))
    with open(os.path.join(args.out, "verify_dl_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] results -> {os.path.join(args.out, 'verify_dl_results.json')}")


if __name__ == "__main__":
    main()
