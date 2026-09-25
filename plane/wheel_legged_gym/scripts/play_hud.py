"""Stroke HUD drawn with Isaac Gym viewer lines; no extra GUI is required."""

import numpy as np


SEGMENTS = {
    "a": ((0.12, 1.00), (0.88, 1.00)),
    "b": ((0.88, 1.00), (0.88, 0.50)),
    "c": ((0.88, 0.50), (0.88, 0.00)),
    "d": ((0.12, 0.00), (0.88, 0.00)),
    "e": ((0.12, 0.50), (0.12, 0.00)),
    "f": ((0.12, 1.00), (0.12, 0.50)),
    "g": ((0.12, 0.50), (0.88, 0.50)),
}

SEGMENT_GLYPHS = {
    "0": "abcdef", "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc",
    "5": "afgcd", "6": "afgecd", "7": "abc", "8": "abcdefg", "9": "abcdfg",
    "A": "abcefg", "C": "afed", "D": "bcdeg", "E": "afged", "F": "afge",
    "G": "afedcg", "H": "fgebc", "L": "fed", "O": "abcdef",
    "P": "abfge", "S": "afgcd", "U": "fbcde", "Y": "fbgcd",
}

EXTRA_GLYPHS = {
    "I": (((0.12, 1), (0.88, 1)), ((0.5, 1), (0.5, 0)), ((0.12, 0), (0.88, 0))),
    "M": (((0.1, 0), (0.1, 1)), ((0.1, 1), (0.5, 0.5)),
          ((0.5, 0.5), (0.9, 1)), ((0.9, 1), (0.9, 0))),
    "N": (((0.1, 0), (0.1, 1)), ((0.1, 1), (0.9, 0)), ((0.9, 0), (0.9, 1))),
    "Q": (SEGMENTS["a"], SEGMENTS["b"], SEGMENTS["c"], SEGMENTS["d"],
          SEGMENTS["e"], SEGMENTS["f"], ((0.6, 0.25), (1.0, -0.15))),
    "R": (SEGMENTS["a"], SEGMENTS["b"], SEGMENTS["f"], SEGMENTS["g"],
          SEGMENTS["e"], ((0.5, 0.5), (0.95, 0))),
    "T": (SEGMENTS["a"], ((0.5, 1), (0.5, 0))),
    "W": (((0.1, 1), (0.1, 0)), ((0.1, 0), (0.5, 0.45)),
          ((0.5, 0.45), (0.9, 0)), ((0.9, 0), (0.9, 1))),
    "+": (((0.12, 0.5), (0.88, 0.5)), ((0.5, 0.12), (0.5, 0.88))),
    "-": (SEGMENTS["g"],),
    ".": (((0.4, 0), (0.58, 0)),),
    "/": (((0.1, 0), (0.9, 1)),),
    ":": (((0.45, 0.25), (0.55, 0.25)), ((0.45, 0.75), (0.55, 0.75))),
    " ": (),
}

WHITE = (0.90, 0.95, 1.00)
CYAN = (0.22, 0.85, 1.00)
GREEN = (0.25, 1.00, 0.42)
AMBER = (1.00, 0.73, 0.16)
RED = (1.00, 0.25, 0.22)


def _glyph_segments(char):
    if char in EXTRA_GLYPHS:
        return EXTRA_GLYPHS[char]
    if char in SEGMENT_GLYPHS:
        return tuple(SEGMENTS[key] for key in SEGMENT_GLYPHS[char])
    raise ValueError(f"HUD glyph missing for {char!r}")


def build_hud_lines(origin, right, rows):
    """Return viewer line vertices/colors in one environment's local frame.

    rows contains (label, value, RGB color). origin is the panel's lower-left
    corner; right is a unit horizontal vector pointing toward camera-right.
    """
    origin = np.asarray(origin, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    up = np.array((0.0, 0.0, 1.0), dtype=np.float32)
    if origin.shape != (3,) or right.shape != (3,):
        raise ValueError("HUD origin and right must be 3D vectors")
    if not np.isfinite(origin).all() or not np.isfinite(right).all():
        raise ValueError("HUD coordinates must be finite")

    vertices = []
    colors = []

    def add(x1, y1, x2, y2, color):
        vertices.extend((origin + x1 * right + y1 * up,
                         origin + x2 * right + y2 * up))
        colors.append(color)

    def write(text, x, y, scale, color):
        for index, char in enumerate(text.upper()):
            x0 = x + index * scale * 1.08
            for (sx, sy), (ex, ey) in _glyph_segments(char):
                add(x0 + sx * scale, y + sy * scale,
                    x0 + ex * scale, y + ey * scale, color)

    width = 2.75
    height = 0.47 + len(rows) * 0.22
    for x1, y1, x2, y2 in (
        (0, 0, width, 0), (width, 0, width, height),
        (width, height, 0, height), (0, height, 0, 0),
        (0.08, height - 0.38, width - 0.08, height - 0.38),
    ):
        add(x1, y1, x2, y2, CYAN)
    write("SPIN HUD", 0.13, height - 0.30, 0.16, CYAN)
    for index, (label, value, color) in enumerate(rows):
        y = height - 0.60 - index * 0.22
        write(label, 0.13, y, 0.14, WHITE)
        write(value, 0.77, y, 0.14, color)

    return (np.asarray(vertices, dtype=np.float32).reshape(-1, 3),
            np.asarray(colors, dtype=np.float32).reshape(-1, 3))
