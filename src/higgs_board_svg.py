"""
Higgs-field chess board, 2D design (SVG). The 3D version is built later from this layout.

- Squares: 40 mm. Over the whole board run the contour lines of the Higgs potential
  V(phi) = (|phi|^2 - v^2)^2 (the "Mexican hat") seen from above: peak in the centre, the
  vacuum valley as a thick ring, colours flipped on the dark squares.
- Files (instead of a-h): the 8 fermions ordered by mass, nu e u d s c b t. Mass comes from
  the Higgs coupling, so the dot next to each symbol grows with the mass (log scale).
- Ranks (instead of 1-8): symmetry breaking. A ball rolls down the Mexican hat: on rank 1 it
  sits on the peak (symmetric, phi = 0), on rank 8 it rests in the valley (phi = v).

Usage:  python src/higgs_board_svg.py   -> images/board_higgs.svg
"""
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SQ = 40.0
SIDE = 26.0          # frame left/right (rank icons)
END = 32.0           # frame on white's / black's side (files + title)
W, D = 8 * SQ + 2 * SIDE, 8 * SQ + 2 * END
HB = 4 * SQ          # half board
INK, PAPER = "#151515", "#f4f4f2"
V_RING = 2.5 * SQ    # radius of the vacuum valley

FERMIONS = [("ν", 1e-7), ("e", 0.511), ("u", 2.2), ("d", 4.7),
            ("s", 95.0), ("c", 1275.0), ("b", 4180.0), ("t", 172760.0)]  # MeV
PHI = [0.0, 0.22, 0.40, 0.55, 0.68, 0.80, 0.91, 1.0]  # ball position on each rank, in units of v


def mass_r(m):
    lo, hi = math.log10(1e-7), math.log10(172760.0)
    return 1.3 + 4.7 * (math.log10(m) - lo) / (hi - lo)


def hat_icon(cx, cy, phi, rot=0):
    """Cross-section of the Mexican hat with a ball at field value phi."""
    pot = lambda x: (x * x - 1) ** 2
    sy = lambda x: 3.5 - pot(x) * 7.0  # SVG y grows downward: the peak is drawn highest
    path = "M" + " L".join(f"{x * 7.0:.2f},{sy(x):.2f}"
                           for x in (-1.35 + 2.7 * i / 60 for i in range(61)))
    ball = f'<circle cx="{phi * 7.0:.2f}" cy="{sy(phi) - 2.0:.2f}" r="1.7" fill="{INK}"/>'
    return (f'<g transform="translate({cx:.2f},{cy:.2f}) rotate({rot})">'
            f'<path d="{path}" fill="none" stroke="{INK}" stroke-width="0.9" stroke-linecap="round"/>{ball}</g>')


def feynman(cx, cy, rot=0):
    """H -> f fbar vertex: dashed Higgs line, two fermion lines with arrows."""
    return (f'<g transform="translate({cx:.2f},{cy:.2f}) rotate({rot})" stroke="{INK}" stroke-width="0.9" fill="none">'
            f'<line x1="-9" y1="0" x2="0" y2="0" stroke-dasharray="2,1.4"/>'
            f'<line x1="0" y1="0" x2="7" y2="-7"/><line x1="0" y1="0" x2="7" y2="7"/>'
            f'<path d="M3.2,-4.6 L4.4,-2.2 M3.2,-4.6 L5.6,-3.4" />'
            f'<path d="M4.4,2.2 L3.2,4.6 M4.4,2.2 L5.6,3.4" />'
            f'<text x="-6" y="-2.2" font-size="4" fill="{INK}" stroke="none" font-family="DejaVu Sans, Arial" font-weight="bold">H</text></g>')


def rings():
    """Equal-potential contours of V(r) = (r^2/v^2 - 1)^2 across the board."""
    out = [f'<circle r="{V_RING:.2f}" stroke-width="2.4"/>', '<circle r="2.6" stroke="none" fill="currentColor"/>']
    for s in (0.3, 0.55, 0.8):  # inside the valley, towards the peak
        out.append(f'<circle r="{V_RING * math.sqrt(1 - s):.2f}" stroke-width="1.0"/>')
    s = 0.3
    while V_RING * math.sqrt(1 + s) < HB * math.sqrt(2):  # outside, up the outer wall
        out.append(f'<circle r="{V_RING * math.sqrt(1 + s):.2f}" stroke-width="1.0"/>')
        s += 0.55
    return "".join(out)


def text(s, x, y, size, rot=0, weight="bold", spacing=0):
    return (f'<text x="0" y="0" transform="translate({x:.2f},{y:.2f}) rotate({rot})" font-size="{size}" '
            f'text-anchor="middle" dominant-baseline="central" font-family="DejaVu Sans, Arial" '
            f'font-weight="{weight}" letter-spacing="{spacing}" fill="{INK}">{s}</text>')


def build():
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}mm" height="{D}mm" '
         f'viewBox="{-W/2} {-D/2} {W} {D}">',
         f'<rect x="{-W/2}" y="{-D/2}" width="{W}" height="{D}" rx="4" fill="{PAPER}"/>']
    dark = []
    for r in range(8):
        for f in range(8):
            if (f + r) % 2 == 0:  # a1 dark; y grows downward, white at the bottom
                dark.append(f'<rect x="{(f - 4) * SQ:.1f}" y="{(3 - r) * SQ:.1f}" width="{SQ}" height="{SQ}"/>')
    o.append(f'<clipPath id="dark">{"".join(dark)}</clipPath>')
    o.append(f'<clipPath id="board"><rect x="{-HB}" y="{-HB}" width="{2*HB}" height="{2*HB}"/></clipPath>')
    o.append(f'<g fill="{INK}">{"".join(dark)}</g>')
    o.append(f'<g clip-path="url(#board)" fill="none" stroke="{INK}" color="{INK}">{rings()}</g>')
    o.append(f'<g clip-path="url(#dark)" fill="none" stroke="{PAPER}" color="{PAPER}">{rings()}</g>')
    o.append(f'<rect x="{-HB}" y="{-HB}" width="{2*HB}" height="{2*HB}" fill="none" stroke="{INK}" stroke-width="1.2"/>')

    # files: fermion symbol + mass dot, upright for white (bottom), turned for black (top)
    for i, (sym, m) in enumerate(FERMIONS):
        x = (i - 3.5) * SQ
        rr = mass_r(m)
        for y, rot in ((HB + 9.5, 0), (-HB - 9.5, 180)):
            k = 1 if rot == 0 else -1
            o.append(f'<circle cx="{x - k * 8:.2f}" cy="{y:.2f}" r="{rr:.2f}" fill="{INK}"/>')
            o.append(text(sym, x + k * 4.5, y, 9, rot))
    # ranks: ball rolling into the valley, on both sides
    for r in range(8):
        y = (3.5 - r) * SQ
        for x in (-HB - SIDE / 2, HB + SIDE / 2):
            o.append(hat_icon(x, y, PHI[r]))
    # titles
    o.append(text("SYMMETRIC · φ = 0", 0, HB + 24, 6.5, 0, spacing=0.8))
    o.append(text("BROKEN · ⟨φ⟩ = 246 GeV", 0, -HB - 24, 6.5, 180, spacing=0.8))
    # corners: Mexican hat on white's side, Higgs vertex on black's side
    for sx in (-1, 1):
        o.append(hat_icon(sx * (HB + SIDE / 2), HB + END / 2, 1.0))
        o.append(feynman(sx * (HB + SIDE / 2), -HB - END / 2, 180))
    o.append("</svg>")
    return "\n".join(o)


if __name__ == "__main__":
    path = os.path.join(ROOT, "images", "board_higgs.svg")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(build())
    print("saved", path)
