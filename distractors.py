"""
Four types of corruptions, producing the distracted frame `x'` along with the 
binary mask matrix (uint8 0/1, HxW; needed to compute unsupervised loss). 
Each with a single swept intensity axis, which allows explicit control over 
the independent variable (therebymaking this a control set)

    shape:      PIL rectangle/ellipse/triangle; axis = area fraction of frame
    noise:      additive Gaussian noise patch; axis = sigma (0..255 scale)
    overlay:    alpha-blended colour tint patch; axis = opacity alpha in (0,1)
    texture:    repeating checkerboard patch; axis = contrast in (0,1)

Placement is occlusion-aware:
    occlude=False -> the distractor never touches `r` (the relevant region),
                     so a correct model's prediction must stay invariant.
    occlude=True  -> the distractor deliberately covers part of `r`, so a
                     correct model may change its output (negative control).
"""

import numpy as np
from PIL import Image, ImageDraw


PATCH_FRAC = 0.08

# Per-type intensity grids 
INTENSITY = {
    "shape":   [0.01, 0.02, 0.04, 0.08, 0.16],  
    "noise":   [10, 25, 50, 75, 100],            
    "overlay": [0.1, 0.3, 0.5, 0.7, 0.9],       
    "texture": [0.1, 0.3, 0.5, 0.7, 0.9],        
}
TYPES = list(INTENSITY.keys())


def _patch_side(area_frac, H, W):
    """Square side (>=3, capped to the frame) for a given area fraction."""
    side = int(round(np.sqrt(area_frac * H * W)))
    return max(3, min(side, min(H, W)))


def _sample_box(rng, bw, bh, r, occlude, max_tries=400):
    """Find a (x0, y0) top-left for a bw x bh box.

    occlude=False -> box must NOT overlap r (task-irrelevant placement).
    occlude=True  -> box MUST overlap r (centred on a random relevant pixel).
    """
    Hh, Ww = r.shape
    if bw > Ww or bh > Hh:
        return None

    if occlude:
        ys, xs = np.where(r > 0)
        if len(xs) == 0:
            return None
        for _ in range(max_tries):
            j = int(rng.integers(len(xs)))
            cx, cy = int(xs[j]), int(ys[j])
            x0 = int(np.clip(cx - bw // 2, 0, Ww - bw))
            y0 = int(np.clip(cy - bh // 2, 0, Hh - bh))
            if r[y0:y0 + bh, x0:x0 + bw].any():
                return x0, y0
        return None

    for _ in range(max_tries):
        x0 = int(rng.integers(0, Ww - bw + 1))
        y0 = int(rng.integers(0, Hh - bh + 1))
        if not r[y0:y0 + bh, x0:x0 + bw].any():
            return x0, y0
    return None


def add_shape(x, r, rng, area_frac, occlude=False):
    """Composite a filled geometric shape."""
    Hh, Ww = r.shape
    side = _patch_side(area_frac, Hh, Ww)
    box = _sample_box(rng, side, side, r, occlude)
    if box is None:
        return None
    x0, y0 = box

    canvas = Image.fromarray(x.copy())
    d = ImageDraw.Draw(canvas)
    mimg = Image.new("L", (Ww, Hh), 0)
    md = ImageDraw.Draw(mimg)

    color = tuple(int(c) for c in rng.integers(60, 256, size=3))
    bbox = [x0, y0, x0 + side - 1, y0 + side - 1]
    kind = int(rng.integers(0, 3))
    if kind == 0:                                  # rectangle
        d.rectangle(bbox, fill=color)
        md.rectangle(bbox, fill=1)
    elif kind == 1:                                # ellipse / circle
        d.ellipse(bbox, fill=color)
        md.ellipse(bbox, fill=1)
    else:                                          # triangle
        pts = [(x0 + side // 2, y0), (x0, y0 + side - 1), (x0 + side - 1, y0 + side - 1)]
        d.polygon(pts, fill=color)
        md.polygon(pts, fill=1)

    return np.asarray(canvas, dtype=np.uint8), np.asarray(mimg, dtype=np.uint8)


def add_noise(x, r, rng, sigma, occlude=False):
    """Add a Gaussian noise patch. """
    Hh, Ww = r.shape
    side = _patch_side(PATCH_FRAC, Hh, Ww)
    box = _sample_box(rng, side, side, r, occlude)
    if box is None:
        return None
    x0, y0 = box

    out = x.astype(np.float64)
    noise = rng.normal(0.0, float(sigma), size=(side, side, 3))
    out[y0:y0 + side, x0:x0 + side] += noise
    out = np.clip(out, 0, 255).astype(np.uint8)

    m = np.zeros((Hh, Ww), dtype=np.uint8)
    m[y0:y0 + side, x0:x0 + side] = 1
    return out, m


def add_overlay(x, r, rng, alpha, occlude=False):
    """Alpha-blend a solid colour tint over a patch."""
    Hh, Ww = r.shape
    side = _patch_side(PATCH_FRAC, Hh, Ww)
    box = _sample_box(rng, side, side, r, occlude)
    if box is None:
        return None
    x0, y0 = box

    tint = rng.integers(0, 256, size=3).astype(np.float64)
    out = x.astype(np.float64)
    region = out[y0:y0 + side, x0:x0 + side]
    out[y0:y0 + side, x0:x0 + side] = (1.0 - alpha) * region + alpha * tint
    out = np.clip(out, 0, 255).astype(np.uint8)

    m = np.zeros((Hh, Ww), dtype=np.uint8)
    m[y0:y0 + side, x0:x0 + side] = 1
    return out, m


def add_texture(x, r, rng, contrast, occlude=False):
    """Paint a repeating checkerboard patch. Swept axis: contrast.
        contrast=0 -> flat grey
        contrast=1 -> full black/white squares
    """
    Hh, Ww = r.shape
    side = _patch_side(PATCH_FRAC, Hh, Ww)
    box = _sample_box(rng, side, side, r, occlude)
    if box is None:
        return None
    x0, y0 = box

    period = int(rng.integers(2, 5))
    yy, xx = np.mgrid[0:side, 0:side]
    checker = ((xx // period + yy // period) % 2).astype(np.float64)   # 0/1
    amp = float(contrast) * 127.0
    patch = 128.0 + (checker * 2.0 - 1.0) * amp                        # [128-amp, 128+amp]

    out = x.astype(np.float64)
    out[y0:y0 + side, x0:x0 + side, :] = patch[:, :, None]
    out = np.clip(out, 0, 255).astype(np.uint8)

    m = np.zeros((Hh, Ww), dtype=np.uint8)
    m[y0:y0 + side, x0:x0 + side] = 1
    return out, m


_DISPATCH = {
    "shape": add_shape,
    "noise": add_noise,
    "overlay": add_overlay,
    "texture": add_texture,
}


def add_distractor(x, r, rng, dtype, value, occlude=False):
    """Dispatch to the distractor of type `dtype` at intensity `value`.
    """
    if dtype not in _DISPATCH:
        raise ValueError(f"unknown distractor type: {dtype!r}")
    return _DISPATCH[dtype](x, r, rng, value, occlude=occlude)
