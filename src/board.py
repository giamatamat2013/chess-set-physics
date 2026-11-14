"""
Periodic-table chess board for the physics set.

64 squares, 40 mm each, one element per square: H (1) on a1 up to Gd (64) on h8, so white
starts among the light elements and black among the heavy ones. Each square shows the atomic
number, the symbol and the name. Instead of a-h the frame has Standard Model particles
(nu e mu gamma H Z W tau), and instead of 1-8 a particle in a box: rank n is the standing wave
with n lobes. CLASSICAL on white's side, QUANTUM on black's side, atom / psi emblems in the corners.

Black and white, printed with an AMS: every quarter is two files, a white body and a black inlay
0.6 mm deep (the dark squares with white text left in them, and the black text on the light squares
and the frame). The top is completely flat, so pieces stand level. In Bambu Studio, load both files
of a quarter together and answer "Yes" to loading them as one object with several parts, then give
the black part the black filament.

The board is 348 x 368 mm, too big for a 256 mm bed, so it prints as four quarters that join
with dovetails (0.15 mm clearance).

Usage:  python src/board.py      -> stl/physics/board/board_quarter_*.stl, images/board_physics.png
"""
import os

import numpy as np
from manifold3d import CrossSection, FillRule, JoinType, Manifold

from build_set import ROOT, ccw, chamfer_slab, rrect
from physics_set import ATOM, PSI, strokes, write_stl

SQ = 40.0              # square size
BASE_H = 4.0           # white below the inlay
TOP_H = 0.6            # depth of the black inlay (3 layers of 0.2 mm)
FRAME_SIDE = 14.0      # frame width left/right
FRAME_END = 24.0       # frame width on white's and black's side
W = 8 * SQ + 2 * FRAME_SIDE
D = 8 * SQ + 2 * FRAME_END
CLEAR = 0.15           # dovetail clearance per side
OUT = os.path.join(ROOT, "stl", "physics", "board")

ELEMENTS = [
    ("H", "Hydrogen"), ("He", "Helium"), ("Li", "Lithium"), ("Be", "Beryllium"), ("B", "Boron"), ("C", "Carbon"),
    ("N", "Nitrogen"), ("O", "Oxygen"), ("F", "Fluorine"), ("Ne", "Neon"), ("Na", "Sodium"), ("Mg", "Magnesium"),
    ("Al", "Aluminium"), ("Si", "Silicon"), ("P", "Phosphorus"), ("S", "Sulfur"), ("Cl", "Chlorine"), ("Ar", "Argon"),
    ("K", "Potassium"), ("Ca", "Calcium"), ("Sc", "Scandium"), ("Ti", "Titanium"), ("V", "Vanadium"),
    ("Cr", "Chromium"), ("Mn", "Manganese"), ("Fe", "Iron"), ("Co", "Cobalt"), ("Ni", "Nickel"), ("Cu", "Copper"),
    ("Zn", "Zinc"), ("Ga", "Gallium"), ("Ge", "Germanium"), ("As", "Arsenic"), ("Se", "Selenium"), ("Br", "Bromine"),
    ("Kr", "Krypton"), ("Rb", "Rubidium"), ("Sr", "Strontium"), ("Y", "Yttrium"), ("Zr", "Zirconium"),
    ("Nb", "Niobium"), ("Mo", "Molybdenum"), ("Tc", "Technetium"), ("Ru", "Ruthenium"), ("Rh", "Rhodium"),
    ("Pd", "Palladium"), ("Ag", "Silver"), ("Cd", "Cadmium"), ("In", "Indium"), ("Sn", "Tin"), ("Sb", "Antimony"),
    ("Te", "Tellurium"), ("I", "Iodine"), ("Xe", "Xenon"), ("Cs", "Caesium"), ("Ba", "Barium"), ("La", "Lanthanum"),
    ("Ce", "Cerium"), ("Pr", "Praseodymium"), ("Nd", "Neodymium"), ("Pm", "Promethium"), ("Sm", "Samarium"),
    ("Eu", "Europium"), ("Gd", "Gadolinium"),
]


# ---------------------------------------------------------------- text
_FP = None


def text_cs(s, cap, cx, cy, max_w=None, rot=0.0, align="center"):
    """Text as a 2D shape: cap = capital-letter height in mm, centred on (cx, cy)."""
    global _FP
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    if _FP is None:
        _FP = FontProperties(family="DejaVu Sans", weight="bold")
    tp = TextPath((0, 0), s, size=cap / 0.729, prop=_FP)
    polys = [np.asarray(p[:-1], float) for p in tp.to_polygons(closed_only=True) if len(p) > 3]
    cs = CrossSection([[tuple(q) for q in p] for p in polys], FillRule.EvenOdd)
    x0, y0, x1, y1 = cs.bounds()
    k = 1.0
    if max_w and x1 - x0 > max_w:  # long names shrink to fit the square
        k = max_w / (x1 - x0)
        cs = cs.scale((k, k))
        x0, y0, x1, y1 = cs.bounds()
    ax = {"center": (x0 + x1) / 2, "left": x0}[align]
    # centre on the capital height above the baseline, so descenders (g, y, p) hang below like in print
    cs = cs.translate((-ax, -cap * k / 2))
    return cs.rotate(rot).translate((cx, cy))


# ---------------------------------------------------------------- frame marks
# files a-h are Standard Model particles: the photon on d (the queen, an EM wave) and the Higgs on e (the king)
FILES = ["ν", "e", "μ", "γ", "H", "Z", "W", "τ"]  # nu e mu gamma H Z W tau


def box_mode(n, length=32.0, amp=3.4, w=0.55):
    """Rank n = particle in a box: the n-th standing wave (n lobes) between two walls, along y."""
    t = np.linspace(0, 1, 30 * n + 20)
    wave = strokes([(amp * np.sin(np.pi * n * s), (s - 0.5) * length) for s in t], w)
    walls = [CrossSection.square((amp * 2 + 2.4, 1.2), center=True).translate((0, sy * (length / 2 + 0.9))) for sy in (-1, 1)]
    return wave + walls[0] + walls[1]


# ---------------------------------------------------------------- board in 2D
def board_top():
    """Everything raised above the base, as one 2D shape."""
    dark, on_light, on_dark = [], [], []
    for r in range(8):
        for f in range(8):
            z = r * 8 + f + 1
            sym, name = ELEMENTS[z - 1]
            cx, cy = (f - 3.5) * SQ, (r - 3.5) * SQ
            is_dark = (f + r) % 2 == 0  # a1 is dark
            txt = [text_cs(str(z), 4.6, cx - SQ / 2 + 3.0, cy + SQ / 2 - 5.6, align="left"),
                   text_cs(sym, 13.0, cx, cy + 2.2, max_w=30),
                   text_cs(name, 3.4, cx, cy - SQ / 2 + 5.4, max_w=SQ - 5)]
            if is_dark:
                dark.append(CrossSection.square((SQ - 0.6, SQ - 0.6), center=True).translate((cx, cy)))
                on_dark += txt
            else:
                on_light += txt
    # frame: files and ranks, the two sides' names and the corner emblems
    half_w, half_d = 4 * SQ, 4 * SQ
    for i in range(8):
        c = (i - 3.5) * SQ
        p = FILES[i]
        on_light.append(text_cs(p, 6.0, c, -half_d - 7.0))
        on_light.append(text_cs(p, 6.0, c, half_d + 7.0, rot=180))
        for sx in (-1, 1):
            on_light.append(box_mode(i + 1).translate((sx * (half_w + FRAME_SIDE / 2), c)))
    on_light.append(text_cs("CLASSICAL", 7.0, 0, -half_d - 17.0))
    on_light.append(text_cs("QUANTUM", 7.0, 0, half_d + 17.0, rot=180))
    for sx in (-1, 1):
        on_light.append(ATOM.scale((0.85, 0.85)).translate((sx * (half_w + 1.0), -half_d - 12.0)))
        on_light.append(PSI.scale((0.85, 0.85)).rotate(180).translate((sx * (half_w + 1.0), half_d + 12.0)))
    u = lambda xs: CrossSection.batch_boolean(xs, __import__("manifold3d").OpType.Add)
    return (u(dark) - u(on_dark)) + u(on_light)


# ---------------------------------------------------------------- quarters with dovetails
def dovetail(at, axis):
    """Tab belonging to the + side of a seam, reaching 8 mm into the - side."""
    pts = [(0.6, -4.0), (0.6, 4.0), (-8.0, 6.5), (-8.0, -6.5)]
    if axis == "x":  # seam x = 0, tab centred at y = at
        pts = [(x, y + at) for x, y in pts]
    else:  # seam y = 0, tab centred at x = at
        pts = [(y + at, x) for x, y in pts]
    return CrossSection([ccw(pts)])


def quarter_masks():
    big = 400.0
    tabs_x = dovetail(92.0, "x") + dovetail(-92.0, "x")
    tabs_y = dovetail(87.0, "y") + dovetail(-87.0, "y")
    xpos = CrossSection.square((big, big)).translate((0, -big / 2)) + tabs_x
    xneg = CrossSection.square((big, big)).translate((-big, -big / 2)) - tabs_x.offset(CLEAR, JoinType.Miter)
    ypos = CrossSection.square((big, big)).translate((-big / 2, 0)) + tabs_y
    yneg = CrossSection.square((big, big)).translate((-big / 2, -big)) - tabs_y.offset(CLEAR, JoinType.Miter)
    names = {(-1, -1): "a1_white_left", (1, -1): "h1_white_right", (-1, 1): "a8_black_left", (1, 1): "h8_black_right"}
    for (sx, sy), name in names.items():
        yield name, (xpos if sx > 0 else xneg) ^ (ypos if sy > 0 else yneg)


def main():
    os.makedirs(OUT, exist_ok=True)
    print("@@now drawing the 64 elements", flush=True)
    black_cs = board_top()
    slab = chamfer_slab(W, D, BASE_H + TOP_H, 4.0, ch_top=0.6, ch_bot=0.4)
    inlay = Manifold.extrude(black_cs, TOP_H + 1.0).translate((0, 0, BASE_H))
    white, black = slab - inlay, slab ^ inlay  # flush top: the black inlay fills the white body's pockets
    for name, mask in quarter_masks():
        print(f"@@now building quarter {name}", flush=True)
        cut = Manifold.extrude(mask, BASE_H + TOP_H + 2).translate((0, 0, -1))
        for colour, body in (("white", white), ("black", black)):
            q = body ^ cut
            write_stl(q, os.path.join(OUT, f"board_quarter_{name}_{colour}.stl"))
            b = q.bounding_box()
            print(f"board_quarter_{name}_{colour}: parts {len(q.decompose())}, "
                  f"{b[3]-b[0]:.1f} x {b[4]-b[1]:.1f} x {b[5]-b[2]:.1f} mm", flush=True)
    render(black_cs)


def render(top):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    def patch(cs, color):
        verts, codes = [], []
        for p in cs.to_polygons():
            p = np.asarray(p)
            verts += list(p) + [p[0]]
            codes += [Path.MOVETO] + [Path.LINETO] * (len(p) - 1) + [Path.CLOSEPOLY]
        return PathPatch(Path(verts, codes), facecolor=color, edgecolor="none")

    fig, ax = plt.subplots(figsize=(10, 10.6), facecolor="#0e1117")
    ax.add_patch(patch(rrect(W, D, 4.0), "#f4f4f2"))
    ax.add_patch(patch(top, "#151515"))
    for name, mask in quarter_masks():  # seams
        ax.add_patch(PathPatch(Path(np.r_[np.asarray((mask ^ rrect(W, D, 4.0)).to_polygons()[0]), [[np.nan, np.nan]]][:-1]),
                               facecolor="none", edgecolor="#c04040", lw=0.6, ls="--"))
    ax.set_xlim(-W / 2 - 4, W / 2 + 4)
    ax.set_ylim(-D / 2 - 4, D / 2 + 4)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.subplots_adjust(0, 0, 1, 1)
    path = os.path.join(ROOT, "images", "board_physics.png")
    plt.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    print("preview saved:", path, flush=True)


if __name__ == "__main__":
    main()
