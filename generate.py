"""Control-dataset generator CLI (Track A: masking / distractor-robustness).

One seed -> one byte-identical dataset. A single ``np.random.default_rng(seed)``
is threaded through the whole generation loop in a fixed iteration order; the
global ``np.random`` is never touched.

For every scene we render a clean frame `x`, its relevant-region mask `r`, and
its task label `y`. Then, for each distractor type x intensity level x split,
we composite one distractor to get `(x', m)`. Two splits are written:

    non_occluding/ -- distractor never touches `r`  (expected: invariance)
    occluding/     -- distractor covers part of `r` (negative control)

Layout written under --out:

    out/
      scenes/         scene_NNNNN_x.png, scene_NNNNN_r.png   (shared per scene)
      non_occluding/  sample_NNNNNN_xp.png, sample_NNNNNN_m.png
      occluding/      sample_NNNNNN_xp.png, sample_NNNNNN_m.png
      metadata.jsonl  one JSON row per sample (full provenance)
      dataset_info.json  run-level summary (seed, counts, balance)

Run:
    python generate.py --track A --seed 0 --out dataset
"""

import argparse
import json
import os
import numpy as np
from PIL import Image
import render
import distractors


def _save_rgb(arr, path):
    Image.fromarray(arr, mode="RGB").save(path)


def _save_mask(arr, path):
    # store 0/255 so the PNG is human-viewable; loaders divide by 255.
    Image.fromarray((arr * 255).astype(np.uint8), mode="L").save(path)


def generate_track_a(seed, out_dir, n_scenes):
    """Generate the Track A masking dataset. Returns the run summary dict."""
    rng = np.random.default_rng(seed)

    scenes_dir = os.path.join(out_dir, "scenes")
    splits = ["non_occluding", "occluding"]
    os.makedirs(scenes_dir, exist_ok=True)
    for s in splits:
        os.makedirs(os.path.join(out_dir, s), exist_ok=True)

    meta_path = os.path.join(out_dir, "metadata.jsonl")
    sample_id = 0
    # counts per (split, type, level).
    counts = {s: {t: [0] * len(distractors.INTENSITY[t]) for t in distractors.TYPES}
              for s in splits}

    with open(meta_path, "w") as meta_f:
        for scene_id in range(n_scenes):
            x, r, y, smeta = render.base_scene(rng)
            x_path = os.path.join("scenes", f"scene_{scene_id:05d}_x.png")
            r_path = os.path.join("scenes", f"scene_{scene_id:05d}_r.png")
            _save_rgb(x, os.path.join(out_dir, x_path))
            _save_mask(r, os.path.join(out_dir, r_path))

            # fixed iteration order for deterministic rng consumption.
            for split in splits:
                occlude = (split == "occluding")
                for dtype in distractors.TYPES:
                    for level, value in enumerate(distractors.INTENSITY[dtype]):
                        result = None
                        for _ in range(8):  # robust retry; deterministic via shared rng
                            result = distractors.add_distractor(
                                x, r, rng, dtype, value, occlude=occlude)
                            if result is not None:
                                break
                        if result is None:
                            raise RuntimeError(
                                f"no valid placement: scene={scene_id} split={split} "
                                f"type={dtype} level={level}")
                        x_prime, m = result

                        xp_path = os.path.join(split, f"sample_{sample_id:06d}_xp.png")
                        m_path = os.path.join(split, f"sample_{sample_id:06d}_m.png")
                        _save_rgb(x_prime, os.path.join(out_dir, xp_path))
                        _save_mask(m, os.path.join(out_dir, m_path))

                        row = {
                            "id": sample_id,
                            "split": split,
                            "occluding": occlude,
                            "scene_id": scene_id,
                            "seed": seed,
                            "type": dtype,
                            "intensity_level": level,
                            "intensity_value": value,
                            "y": y,
                            "y_name": smeta["y_name"],
                            "agent": smeta["agent"],
                            "goal": smeta["goal"],
                            "x": x_path,
                            "r": r_path,
                            "x_prime": xp_path,
                            "m": m_path,
                        }
                        meta_f.write(json.dumps(row) + "\n")
                        counts[split][dtype][level] += 1
                        sample_id += 1

    # every (split, type, level) cell is equal.
    cell_values = [v for s in counts for t in counts[s] for v in counts[s][t]]
    balanced = len(set(cell_values)) == 1

    info = {
        "track": "A",
        "seed": seed,
        "n_scenes": n_scenes,
        "frame": [render.H, render.W],
        "splits": splits,
        "types": distractors.TYPES,
        "intensity": distractors.INTENSITY,
        "directions": render.DIRECTIONS,
        "n_samples": sample_id,
        "balanced": balanced,
        "per_cell_count": cell_values[0] if cell_values else 0,
        "counts": counts,
    }
    with open(os.path.join(out_dir, "dataset_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    return info


def main():
    parser = argparse.ArgumentParser(description="Control-dataset generator")
    parser.add_argument("--track", choices=["A"], default="A",
                        help="Track to build. A = masking/distractor-robustness.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Master seed; one seed -> one byte-identical dataset.")
    parser.add_argument("--out", default="dataset", help="Output directory.")
    parser.add_argument("--n-scenes", type=int, default=30,
                        help="Number of base scenes to render.")
    args = parser.parse_args()

    if args.track != "A":
        raise SystemExit("Only Track A is implemented in this build.")

    os.makedirs(args.out, exist_ok=True)
    print(f"[generate] track=A seed={args.seed} out={args.out} n_scenes={args.n_scenes}")
    info = generate_track_a(args.seed, args.out, args.n_scenes)
    print(f"[generate] wrote {info['n_samples']} samples "
          f"({info['per_cell_count']} per type x intensity x split cell)")
    print(f"[generate] balanced across type x intensity: {info['balanced']}")
    print(f"[generate] metadata -> {os.path.join(args.out, 'metadata.jsonl')}")
    print(f"[generate] summary  -> {os.path.join(args.out, 'dataset_info.json')}")
    print(f"[generate] reproduce with: python generate.py --track A --seed {args.seed} "
          f"--out {args.out} --n-scenes {args.n_scenes}")


if __name__ == "__main__":
    main()
