"""
Higgs-field chess board in 3D, from the approved SVG design (src/higgs_board_svg.py).

Black and white with an AMS: a white body 6 mm thick and a black inlay 0.6 mm deep, flush on top.
The black part is: dark squares XOR the contour lines of the Mexican-hat potential, the frame
graphics (fermions with mass dots, the ball rolling down the hat on every rank, titles, corner icons).

372 x 384 mm, printed as four quarters joined with the dovetails proven in chess-set-naama
(src/build_board.py there): tabs only in the lower 2.6 mm, sockets 0.2 mm taller so the upper
part rests on the tabs, 0.15 mm clearance per side, a straight 0.2 mm seam on the surface along
a square edge. Assembly: a1-d4 first, then a5-d8 and e1-h4 dropped in from above, then e5-h8.

Usage:  python src/higgs_board.py [clearance]  -> stl/physics/board_higgs/*.stl
"""
import math
import os
import sys

import numpy as np
from manifold3d import CrossSection, JoinType, Manifold, OpType

from board import text_cs
from build_set import ROOT
from physics_set import strokes, write_stl
from higgs_board_svg import END, FERMIONS, HB, PHI, SIDE, SQ, V_RING, mass_r

W, D = 8 * SQ + 2 * SIDE, 8 * SQ + 2 * END
T, INK = 6.0, 0.6
LINE = 1.0                                     # ring and icon line width (>= 2 nozzle lines)
# --- connectors, as in chess-set-naama
TAB_H, LIFT = 2.6, 0.2
CLEAR = float(sys.argv[1]) if len(sys.argv) > 1 else 0.15
NECK, HEAD, DEPTH = 14.0, 22.0, 12.0
TAB_AT = (45.0, 125.0)                         # tab centres along each seam (scaled to this board)
OUT = os.path.join(ROOT, "stl", "physics", "board_higgs")


def u(parts):
    out = CrossSection()
    for p in parts:
        out = out + p
    return out


def circle(r, x=0.0, y=0.0):
    return CrossSection.circle(r, 192).translate((x, y))


def ring(r, w):
    return circle(r + w / 2) - circle(r - w / 2)


# ---------------------------------------------------------------- 2D artwork (y up, white at y < 0)
def rings_cs():
    out = [ring(V_RING, 2.4), circle(2.6)]
    out += [ring(V_RING * math.sqrt(1 - s), LINE) for s in (0.3, 0.55, 0.8)]
    s = 0.3
    while V_RING * math.sqrt(1 + s) < HB * math.sqrt(2):
        out.append(ring(V_RING * math.sqrt(1 + s), LINE))
        s += 0.55
    return u(out)


def hat_cs(phi):
    pot = lambda x: (x * x - 1) ** 2
    y = lambda x: pot(x) * 7.0 - 3.5          # peak drawn highest
    curve = strokes([(x * 7.0, y(x)) for x in np.linspace(-1.35, 1.35, 61)], LINE)
    return curve + circle(1.7, phi * 7.0, y(phi) + 2.0)


def feynman_cs():
    """H -> f fbar, Higgs line dashed, pointing to -x (rotated by the caller)."""
    parts = [strokes([(-9 + 3.4 * k, 0), (-9 + 3.4 * k + 2.0, 0)], LINE) for k in range(3)]
    parts += [strokes([(0, 0), (7, 7)], LINE), strokes([(0, 0), (7, -7)], LINE),
              strokes([(3.2, 4.6), (4.4, 2.2)], LINE), strokes([(3.2, 4.6), (5.6, 3.4)], LINE),
              strokes([(4.4, -2.2), (3.2, -4.6)], LINE), strokes([(4.4, -2.2), (5.6, -3.4)], LINE),
              text_cs("H", 3.0, -6.0, 3.4)]
    return u(parts)


def artwork():
    dark = u([CrossSection.square((SQ, SQ)).translate(((f - 4) * SQ, (r - 4) * SQ))
              for r in range(8) for f in range(8) if (f + r) % 2 == 0])
    board = CrossSection.square((2 * HB, 2 * HB), center=True)
    field = (rings_cs() ^ board)
    black = (dark - field) + (field - dark)
    black = black + (CrossSection.square((2 * HB + 2.4, 2 * HB + 2.4), center=True) - board)  # border line
    frame = []
    for i, (sym, m) in enumerate(FERMIONS):
        x = (i - 3.5) * SQ
        r = mass_r(m)
        frame += [circle(r, x - 8, -HB - 9.5), text_cs(sym, 6.5, x + 4.5, -HB - 9.5),
                  circle(r, x + 8, HB + 9.5), text_cs(sym, 6.5, x - 4.5, HB + 9.5, rot=180)]
    for rk in range(8):
        y = (rk - 3.5) * SQ
        for x in (-HB - SIDE / 2, HB + SIDE / 2):
            frame.append(hat_cs(PHI[rk]).translate((x, y)))
    frame.append(text_cs("SYMMETRIC · φ = 0", 4.8, 0, -HB - 24))
    frame.append(text_cs("BROKEN · ⟨φ⟩ = 246 GeV", 4.8, 0, HB + 24, rot=180))
    for sx in (-1, 1):
        frame.append(hat_cs(1.0).translate((sx * (HB + SIDE / 2), -HB - END / 2)))
        frame.append(feynman_cs().rotate(180).translate((sx * (HB + SIDE / 2), HB + END / 2)))
    return black + u(frame)


# ---------------------------------------------------------------- connectors (chess-set-naama)
def tab(x, y, ang):
    pts = [(-1.0, -NECK / 2), (DEPTH, -HEAD / 2), (DEPTH, HEAD / 2), (-1.0, NECK / 2)]
    t = CrossSection([pts]).offset(-1.0, JoinType.Round).offset(1.0, JoinType.Round)
    return t.rotate(ang).translate((x, y))


QUADS = {(-1, -1): "a1-d4", (1, -1): "e1-h4", (-1, 1): "a5-d8", (1, 1): "e5-h8"}


def extrude(cs, h, z0):
    return Manifold.extrude(cs, h).translate((0, 0, z0))


def region(sx, sy):
    big, gap = 400.0, 0.1
    x0, x1 = (-big, -gap) if sx < 0 else (gap, big)
    y0, y1 = (-big, -gap) if sy < 0 else (gap, big)
    solid = extrude(CrossSection.square((x1 - x0, y1 - y0)).translate((x0, y0)), T + 5, -1)
    tabs, sockets = [], []
    for d in TAB_AT:  # seam x = 0: left quarters carry the tabs
        (tabs if sx < 0 else sockets).append(tab(0, sy * d, 0))
    for d in TAB_AT:  # seam y = 0: bottom quarters carry the tabs
        (tabs if sy < 0 else sockets).append(tab(sx * d, 0, 90))
    if tabs:
        solid = solid + extrude(u(tabs), TAB_H + 1, -1)
    if sockets:
        solid = solid - extrude(u(sockets).offset(CLEAR, JoinType.Round), TAB_H + LIFT + 1, -1)
    return solid


def main():
    os.makedirs(OUT, exist_ok=True)
    outline = CrossSection.square((W - 8, D - 8), center=True).offset(4.0, JoinType.Round, 2.0, 48)
    slab = extrude(outline, T, 0)
    inlay = extrude(artwork(), INK + 1, T - INK) ^ slab
    white = slab - inlay
    for (sx, sy), name in QUADS.items():
        reg = region(sx, sy)
        for colour, body in (("white", white ^ reg), ("black", inlay ^ reg)):
            os.makedirs(os.path.join(OUT, colour), exist_ok=True)
            write_stl(body, os.path.join(OUT, colour, f"board_higgs_{name}_{colour}.stl"))
            b = body.bounding_box()
            print(f"{name} {colour}: parts {len(body.decompose())}, "
                  f"{b[3]-b[0]:.1f} x {b[4]-b[1]:.1f} x {b[5]-b[2]:.1f} mm", flush=True)
    print(f"done, clearance {CLEAR} mm per side")


if __name__ == "__main__":
    main()
