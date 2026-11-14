"""Strength check: will a printed piece snap in normal play?

Usage: python src/strength_check.py physics/classical physics/quantum   (folders under stl/)

A print is weakest between layers, so every horizontal slice is treated as a beam section:
  - whole piece: a 10 N sideways push at the top (a firm finger, or a piece knocked over),
    bending moment M = F * (height above the slice), checked on the body (bottom 80% of the height)
  - small features (ears, spikes, rings, cross arms, slit bars): every slice part under 40 mm2
    gets a 3 N push at the top of the feature it carries
Stress = M * c / I_min (I_min = weakest bending axis of the slice, c = farthest point from its centre).
It is compared with PLA layer adhesion, taken low at 15 MPa. A slice passes with a safety factor of 2
or more (stress <= 7.5 MPa). Plates are skipped (they are copies of the pieces).
"""
import glob
import os
import sys

import numpy as np
from manifold3d import JoinType

from print_check import _manifold, load_stl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAYER_MPA = 15.0
SAFETY = 2.0
F_PIECE = 10.0
F_FEATURE = 3.0
SMALL = 40.0
STEP = 0.4


def section(cs):
    """Area, I_min (mm4) and c (mm) of a 2D section (holes come with negative orientation)."""
    A = Sx = Sy = Ixx = Iyy = Ixy = 0.0
    pts_all = []
    for p in cs.to_polygons():
        p = np.asarray(p, float)
        pts_all.append(p)
        x, y = p[:, 0], p[:, 1]
        x1, y1 = np.roll(x, -1), np.roll(y, -1)
        cr = x * y1 - x1 * y
        A += cr.sum() / 2
        Sx += ((x + x1) * cr).sum() / 6
        Sy += ((y + y1) * cr).sum() / 6
        Iyy += ((x * x + x * x1 + x1 * x1) * cr).sum() / 12  # about y axis: integral x^2
        Ixx += ((y * y + y * y1 + y1 * y1) * cr).sum() / 12
        Ixy += ((x * y1 + 2 * x * y + 2 * x1 * y1 + x1 * y) * cr).sum() / 24
    if A <= 1e-9:
        return 0.0, 0.0, 0.0
    cx, cy = Sx / A, Sy / A
    Ixx, Iyy, Ixy = Ixx - A * cy * cy, Iyy - A * cx * cx, Ixy - A * cx * cy
    imin = (Ixx + Iyy) / 2 - np.sqrt(((Ixx - Iyy) / 2) ** 2 + Ixy ** 2)
    c = max(np.hypot(p[:, 0] - cx, p[:, 1] - cy).max() for p in pts_all)
    return A, max(imin, 1e-9), c


def check(path):
    m, v = _manifold(load_stl(path))
    z0, z1 = v[:, 2].min(), v[:, 2].max()
    zs = np.arange(z0 + STEP, z1 - STEP, STEP)
    slices = {z: m.slice(z) for z in zs}
    worst_piece = (0.0, None)
    worst_feat = (0.0, None)
    body_top = z0 + 0.8 * (z1 - z0)  # the push lands on the body; the top 20% (crosses, ears, crowns) gets the feature test
    for z in zs:
        sec = slices[z]
        A, I, c = section(sec)
        if A > 0 and z <= body_top:
            s = F_PIECE * (z1 - z) * c / I
            if s > worst_piece[0]:
                worst_piece = (s, (z, A))
        for part in sec.decompose():
            a = part.area()
            if a >= SMALL or a < 0.3:
                continue
            # follow the feature up while it stays over this part (grown a little each layer)
            # until it joins the body: once it is part of a big section, the body carries the load
            # the search widens at 45 degrees (one STEP per layer), as far as a printed feature can spread
            reg, top = part.offset(STEP, JoinType.Round), z
            for zz in zs[zs > z]:
                nxt = slices[zz] ^ reg
                if nxt.area() < 0.05:
                    break
                top = zz
                if any(c.area() >= SMALL and (c ^ reg).area() > 0.05 for c in slices[zz].decompose()):
                    break
                reg = nxt.offset(STEP, JoinType.Round)
            if top - z < 1.0:
                continue
            Ap, Ip, cp = section(part)
            s = F_FEATURE * (top - z) * cp / Ip
            if s > worst_feat[0]:
                b = part.bounds()
                worst_feat = (s, (z, a, (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, top - z))
    return worst_piece, worst_feat


def main():
    folders = sys.argv[1:] or ["physics/classical", "physics/quantum"]
    limit = LAYER_MPA / SAFETY
    bad = 0
    for f in folders:
        print(f"== {f} ==", flush=True)
        for path in sorted(glob.glob(os.path.join(ROOT, "stl", f, "*.stl"))):
            name = os.path.basename(path)[:-4]
            if name.startswith("plate"):
                continue
            (sp, wp), (sf, wf) = check(path)
            ok = sp <= limit and sf <= limit
            bad += not ok
            line = f"{name}: whole piece {sp:.1f} MPa (x{LAYER_MPA / max(sp, 1e-9):.0f})"
            if wp:
                line += f" at z{wp[0]:.0f}"
            if wf:
                line += (f", weakest feature {sf:.1f} MPa (x{LAYER_MPA / max(sf, 1e-9):.1f}) at z{wf[0]:.0f} "
                         f"({wf[2]:.0f},{wf[3]:.0f}), {wf[1]:.1f} mm2 carrying {wf[4]:.0f} mm")
            print(line + ("  -> OK" if ok else "  -> WEAK"), flush=True)
            print("@@sub done", flush=True)
    print("ALL STRONG" if bad == 0 else f"{bad} file(s) may snap", flush=True)


if __name__ == "__main__":
    main()
