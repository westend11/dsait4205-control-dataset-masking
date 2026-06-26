"""
Renders an 84x84 RGB observation containing one *agent* sprite and one *goal*
sprite on a plain background. Returns, for every scene:
    x  : the clean RGB frame (np.uint8, HxWx3)
    r  : the relevant-region mask (np.uint8, HxW; 1 on agent+goal pixels)
    y  : the task label (the 8-class direction from agent to goal)
    meta: agent/goal centres so downstream code can reason about placement
"""

import numpy as np
from PIL import Image, ImageDraw

H = 84
W = 84

BG_COLOR = (40, 40, 40)
AGENT_COLOR = (0, 200, 0)   # green circle
GOAL_COLOR = (200, 0, 0)    # red square
SPRITE_R = 4                # sprite half-size in pixels

MARGIN = 8                  # sprite centers should be placed away from border frames
MIN_SEP = 20                # min centre-to-centre distance (agent vs goal)

# 8-class direction label. Index 0 == East, increasing counter-clockwise.
DIRECTIONS = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]


def direction_label(ax, ay, gx, gy):
    """
    Image coordinates put +y downward, so we flip dy to use the natural
    "up is north" convention. Bin width is 45 degrees, centred on East.
    """
    dx = gx - ax
    dy = -(gy - ay)
    ang = np.arctan2(dy, dx)              # radians in (-pi, pi]
    idx = int(np.round(ang / (np.pi / 4))) % 8
    return idx


def relevant_mask(ax, ay, gx, gy, H=H, W=W):
    """Binary mask (uint8 0/1) marking the task-relevant pixels: agent + goal.
    """
    mimg = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(mimg)
    d.ellipse([ax - SPRITE_R, ay - SPRITE_R, ax + SPRITE_R, ay + SPRITE_R], fill=1)
    d.rectangle([gx - SPRITE_R, gy - SPRITE_R, gx + SPRITE_R, gy + SPRITE_R], fill=1)
    return np.asarray(mimg, dtype=np.uint8)


def sample_positions(rng, H=H, W=W):
    """Sample non-overlapping agent and goal centres inside the safe margin."""
    while True:
        ax = int(rng.integers(MARGIN, W - MARGIN))
        ay = int(rng.integers(MARGIN, H - MARGIN))
        gx = int(rng.integers(MARGIN, W - MARGIN))
        gy = int(rng.integers(MARGIN, H - MARGIN))
        if (ax - gx) ** 2 + (ay - gy) ** 2 >= MIN_SEP ** 2:
            return ax, ay, gx, gy


def base_scene(rng, H=H, W=W):
    """Render one scene (no distractors).

    Returns (x, r, y, meta):
        x:      uint8 HxWx3 RGB frame
        r:      uint8 HxW relevant-region mask (1 on agent+goal)
        y:      int in [0, 8) direction label (e.g. 'NW')
        meta:   dict with agent/goal centres and the human-readable label
    """
    ax, ay, gx, gy = sample_positions(rng, H, W)

    img = Image.new("RGB", (W, H), BG_COLOR)
    d = ImageDraw.Draw(img)
    d.ellipse([ax - SPRITE_R, ay - SPRITE_R, ax + SPRITE_R, ay + SPRITE_R], fill=AGENT_COLOR)
    d.rectangle([gx - SPRITE_R, gy - SPRITE_R, gx + SPRITE_R, gy + SPRITE_R], fill=GOAL_COLOR)

    x = np.asarray(img, dtype=np.uint8)
    r = relevant_mask(ax, ay, gx, gy, H, W)
    y = direction_label(ax, ay, gx, gy)
    meta = {
        "agent": [ax, ay],
        "goal": [gx, gy],
        "y": y,
        "y_name": DIRECTIONS[y],
    }
    return x, r, y, meta
