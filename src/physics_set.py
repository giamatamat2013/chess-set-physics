"""
Physics chess set - one physics idea per piece type. White and black are the same models,
printed in two colours.

  Pawn     the atom: a nucleus of protons and neutrons, three tilted orbits with their electrons
  Rook     the strong and the weak nuclear force: a proton on top of the tower (three quarks held
           by curly gluon lines), beta decay (n -> p + e + anti-neutrino through a W boson) and
           quark-gluon-quark Feynman diagrams cut into the walls
  Knight   quantum tunnelling (the knight is the one piece that jumps through): on both sides of
           the head a wave runs into a raised barrier, dies down inside it and leaves smaller.
           A psi eye and a mane of beads (quantised energy levels)
  Bishop   the speed of light: the bishop moves on diagonals and light moves at 45 degrees in
           space-time. Two tilted rings cross at an event on the mitre (its light cone); the slit
           is the light ray leaving it. A photon is the ball on top
  Queen    the electromagnetic force: the electric and magnetic fields climb the body as two fins
           at right angles; the crown is the field lines of a magnet, from the body up to the pole
  King     gravity: a funnel of curved space-time, a ringed planet in its mouth, the cross on top

Every piece stands on a round dial base: tick marks round the rim and a groove below them.

Usage:  python src/physics_set.py            # 6 pieces, the 16-piece plate, preview
        python src/physics_set.py pawn king  # only these piece types (no plate)
Output: stl/physics/, images/preview_physics.png. Print every file once in each colour.
Check:  python src/print_check.py physics   and   python src/strength_check.py physics
"""
import os
import sys

import numpy as np
from manifold3d import CrossSection, JoinType, Manifold

from build_set import (KNIGHT_PTS, ROOT, bevel_extrude, box, ccw, chamfer_slab, cyl, revolve, ring_groove,
                       rounded_box, rrect, smooth_closed, sphere, to_xz, tube, union, ycyl)

VERSION = "physics"
OUT = os.path.join(ROOT, "stl", VERSION)


# ---------------------------------------------------------------- 2D emblems (about 8.8 mm radius)
def strokes(pts, r, closed=False):
    pts = [(float(x), float(y)) for x, y in pts]
    if closed:
        pts = pts + pts[:1]
    cs = None
    for a, b in zip(pts, pts[1:]):
        h = CrossSection.batch_hull([CrossSection.circle(r, 16).translate(a), CrossSection.circle(r, 16).translate(b)])
        cs = h if cs is None else cs + h
    return cs


def atom_cs():
    """Nucleus and three electron orbits."""
    cs = CrossSection.circle(2.4, 32)
    t = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    for k in range(3):
        a = np.radians(90 + 60 * k)
        x, y = 7.6 * np.cos(t), 2.9 * np.sin(t)
        cs = cs + strokes(list(zip(np.cos(a) * x - np.sin(a) * y, np.sin(a) * x + np.cos(a) * y)), 1.2, closed=True)
    return cs


def psi_cs():
    """The wave function letter psi (used by the board)."""
    arc = [(4.6 * np.cos(t), 1.0 + 4.6 * np.sin(t)) for t in np.linspace(np.pi, 2 * np.pi, 13)]
    cup = [(-5.2, 6.4)] + arc + [(5.2, 6.4)]
    return strokes(cup, 1.3) + strokes([(0, -8.0), (0, 8.2)], 1.3) + strokes([(-2.8, -8.0), (2.8, -8.0)], 1.3)


ATOM, PSI = atom_cs(), psi_cs()


# ---------------------------------------------------------------- engraved labels
def _arc(cx, cy, r, a0, a1, n=14, ry=None):
    t = np.radians(np.linspace(a0, a1, n))
    return [(cx + r * np.cos(u), cy + (ry or r) * np.sin(u)) for u in t]


# stroke font, cap height 1: char -> (advance width, polylines)
GLYPHS = {
    "A": (0.6, [[(0, 0), (0.3, 1), (0.6, 0)], [(0.12, 0.4), (0.48, 0.4)]]),
    "B": (0.6, [[(0, 0.5), (0.42, 0.5), (0.56, 0.62), (0.56, 0.88), (0.42, 1), (0, 1), (0, 0), (0.45, 0), (0.6, 0.12),
                (0.6, 0.38), (0.45, 0.5)]]),
    "E": (0.55, [[(0.55, 1), (0, 1), (0, 0), (0.55, 0)], [(0, 0.5), (0.42, 0.5)]]),
    "G": (0.6, [[(0.6, 0.85), (0.45, 1), (0.15, 1), (0, 0.8), (0, 0.2), (0.15, 0), (0.45, 0), (0.6, 0.15), (0.6, 0.45),
                (0.35, 0.45)]]),
    "K": (0.58, [[(0, 0), (0, 1)], [(0.56, 1), (0, 0.42)], [(0.2, 0.62), (0.58, 0)]]),
    "N": (0.6, [[(0, 0), (0, 1), (0.6, 0), (0.6, 1)]]),
    "O": (0.6, [[(0.15, 0), (0.45, 0), (0.6, 0.2), (0.6, 0.8), (0.45, 1), (0.15, 1), (0, 0.8), (0, 0.2), (0.15, 0)]]),
    "R": (0.6, [[(0, 0), (0, 1), (0.42, 1), (0.57, 0.86), (0.57, 0.64), (0.42, 0.5), (0, 0.5)], [(0.25, 0.5), (0.6, 0)]]),
    "S": (0.6, [[(0.6, 0.85), (0.45, 1), (0.15, 1), (0, 0.85), (0, 0.64), (0.15, 0.52), (0.45, 0.48), (0.6, 0.36),
                (0.6, 0.15), (0.45, 0), (0.15, 0), (0, 0.15)]]),
    "T": (0.6, [[(0, 1), (0.6, 1)], [(0.3, 1), (0.3, 0)]]),
    "W": (0.76, [[(0, 1), (0.19, 0), (0.38, 0.62), (0.57, 0), (0.76, 1)]]),
    "c": (0.56, [_arc(0.3, 0.32, 0.29, 40, 320, 18, 0.32)]),
    "d": (0.56, [_arc(0.27, 0.32, 0.27, 0, 360, 20, 0.32), [(0.56, 0), (0.56, 1)]]),
    "e": (0.58, [[(0.02, 0.33), (0.57, 0.33)] + _arc(0.3, 0.33, 0.28, 0, 320, 18, 0.32)]),
    "g": (0.56, [_arc(0.27, 0.36, 0.27, 0, 360, 20, 0.28), [(0.56, 0.64), (0.56, -0.08)] + _arc(0.28, -0.08, 0.28, 0, -170, 10, 0.24)]),
    "n": (0.56, [[(0, 0), (0, 0.64)], [(0, 0.36)] + _arc(0.28, 0.36, 0.28, 180, 0, 12) + [(0.56, 0)]]),
    "p": (0.58, [[(0, -0.36), (0, 0.64)], _arc(0.29, 0.32, 0.29, 0, 360, 20, 0.32)]),
    "u": (0.56, [[(0, 0.64), (0, 0.3)] + _arc(0.28, 0.3, 0.28, 180, 360, 12) + [(0.56, 0.64)], [(0.56, 0.64), (0.56, 0)]]),
    "v": (0.6, [[(0, 0.64), (0.3, 0), (0.6, 0.64)], [(0.08, 0.86), (0.52, 0.86)]]),  # anti-neutrino (nu with a bar)
}


def text_cs(s, h, w=0.75):
    """Text as a 2D shape: cap height h (mm), stroke width w (mm), centred on x = 0, baseline y = 0."""
    gap = 0.32
    total = sum(GLYPHS[c][0] for c in s) + gap * (len(s) - 1)
    x, cs = -total / 2, None
    for c in s:
        adv, lines = GLYPHS[c]
        for pl in lines:
            part = strokes([((x + px) * h, py * h) for px, py in pl], w / 2)
            cs = part if cs is None else cs + part
        x += adv + gap
    return cs


def engrave_front(cs, x, z, inner, turn=0.0, reach=40.0):
    """A cutter for text drawn in (x, z), projected onto the side of the piece facing -y (turned
    by `turn` degrees about z). `inner` is the piece shrunk by the groove depth: subtracting the
    cutter leaves grooves exactly that deep, on flat and round faces alike."""
    prism = Manifold.extrude(cs.translate((x, z)), reach).rotate((90, 0, 0))  # y from -reach to 0
    return (prism - inner).rotate((0, 0, turn))


def shrunk(prof, d):
    """A revolve profile moved d mm in toward the axis (for groove depths on round bodies)."""
    return revolve([(max(r - d, 0.0), z) for r, z in prof])


def emb_xz(cs, radius, thick):
    """Emblem standing up in the x-z plane, centred on the origin."""
    f = radius / 8.8
    return Manifold.extrude(cs.scale((f, f)), thick).rotate((90, 0, 0)).translate((0, thick / 2, 0))


# ---------------------------------------------------------------- base
BASE_TOP = 6.6


def dial_base(R):
    """Round base like a measuring dial: 48 tick marks round the rim (every 4th long) and a groove
    round the side. Returns (manifold, z where the piece starts)."""
    m = revolve([(0, 0), (R - 0.5, 0), (R, 0.5), (R, 3.6), (R - 1.8, 5.4), (R - 2.6, 5.4), (R - 3.6, BASE_TOP), (0, BASE_TOP)])
    cuts = [ring_groove(R, 0.6, 0.7, 1.9)]
    for k in range(48):
        L = 2.6 if k % 4 == 0 else 1.6
        cuts.append(box(L, 0.7, 2.6, cx=R - L / 2 + 0.4, z0=3.8).rotate((0, 0, 7.5 * k)))
    return m - union(cuts), BASE_TOP - 0.3


def face_grooves(paths, w, depth, th, x0=0.0, z0=0.0):
    """Grooves drawn in (x, z) on both flat faces (y = +-th/2) of a standing piece."""
    out = []
    for pts in paths:
        for sy in (-1, 1):
            caps = [ycyl(w / 2, sy * (th / 2 - depth), sy * (th / 2 + 1), z + z0, x=x + x0, seg=16) for x, z in pts]
            out += caps if len(caps) == 1 else [Manifold.batch_hull([a, b]) for a, b in zip(caps, caps[1:])]
    return out


def cross_top(zb, post_h=10.0, arm=11.0, w=4.0):
    """King's cross; the arms rise from the post at under 50 degrees, so they print without supports."""
    za = zb + post_h - 4.5
    arms = Manifold.batch_hull([box(w, w, 0.4, z0=za - 3.0), box(arm, w, 3.0, z0=za)])
    return box(w, w, post_h, z0=zb) + arms


def fib_sphere(n):
    """n points spread evenly over a unit sphere (golden spiral), as an (n, 3) array."""
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    r = np.sqrt(1 - z * z)
    a = i * np.pi * (3 - np.sqrt(5))
    return np.stack([r * np.cos(a), r * np.sin(a), z], 1)


# ---------------------------------------------------------------- PAWN
def pawn():
    """Atom: a nucleus packed with protons and neutrons, three tilted electron orbits resting on
    the collar, an electron riding on top of each orbit."""
    base, z0 = dial_base(13.0)
    zc = 21.0  # collar top (local)
    body = revolve([(0, 0), (8.5, 0), (8.5, 1.0), (6.0, 3.0), (4.3, 9.0), (3.7, 12.6), (4.5, 13.4), (4.5, 14.4),
                    (3.7, 15.2), (8.6, 20.1), (8.6, zc), (0, zc)])
    parts = [base, body.translate((0, 0, z0 - 0.01))]
    # the orbits run round the nucleus just inside its bumps, so they are fused to it all the way round
    Ro, rt, tilt, Rn = 7.2, 1.5, np.radians(35), 4.6
    hc = z0 + zc + Ro * np.sin(tilt) + rt - 0.25  # lowest point of every orbit sits on the collar
    parts.append(cyl(hc - z0 - zc + 0.6, 3.6, z0=z0 + zc - 0.3))
    parts.append(sphere(Rn, z=hc, seg=48))
    for p in fib_sphere(26):  # nucleons; none on the underside, where they would hang in the air
        if p[2] > -0.1:
            parts.append(sphere(2.0, *(p * (Rn + 0.3) + [0, 0, hc]), seg=20))
    t = np.linspace(0, 2 * np.pi, 61)
    for k in range(3):
        pts = np.stack([Ro * np.cos(t), Ro * np.sin(t) * np.cos(tilt), Ro * np.sin(t) * np.sin(tilt)], 1)
        a = np.radians(120 * k + 30)
        rot = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        pts = pts @ rot.T + [0, 0, hc]
        parts.append(tube([tuple(p) for p in pts], rt, seg=16))
        top = np.array([0, Ro * np.cos(tilt), Ro * np.sin(tilt) + 0.35]) @ rot.T + [0, 0, hc]
        parts.append(sphere(2.0, *top, seg=24))  # the electron, on top of its orbit
    return union(parts)


# ---------------------------------------------------------------- ROOK
def battlements(s, z, cap_w=23.0, wall_h=6.0):
    """Square cap flaring out at 45 degrees and a notched wall on a tower of width s ending at z.
    Returns (manifold, z of the floor inside the wall)."""
    flare = (cap_w - s) / 2
    cap = Manifold.extrude(rrect(s, s, 1.2), flare, 0, 0.0, (cap_w / s, cap_w / s)).translate((0, 0, z - 0.3))
    cap = cap + chamfer_slab(cap_w, cap_w, 2.5, 1.4).translate((0, 0, z + flare - 0.6))
    z += flare + 1.9
    ring = chamfer_slab(cap_w, cap_w, wall_h, 1.4, ch_top=0.5).translate((0, 0, z - 0.3)) - box(
        cap_w - 5, cap_w - 5, wall_h + 1, z0=z + 0.6)
    ring = ring - union([box(6.0, cap_w + 2, 3.6, z0=z + wall_h - 3.4).rotate((0, 0, a)) for a in (0, 90)])
    return cap + ring, z + 0.6


def curly(a, b, loops, d, n=24):
    """A Feynman gluon line (a row of loops) from a to b, as 2D points. d: loop radius."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = np.linalg.norm(b - a)
    u, v = (b - a) / L, np.array([-(b - a)[1], (b - a)[0]]) / L
    th = np.linspace(0, 2 * np.pi * loops, n * loops + 1)
    roll = L / (2 * np.pi * loops)
    along = roll * th - d * np.sin(th)
    side = d * (1 - np.cos(th))
    return [tuple(a + u * p + v * q) for p, q in zip(along, side)]


def wavy(a, b, waves, amp, n=16):
    """A Feynman boson line (a sine wave) from a to b, as 2D points."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    v = np.array([-d[1], d[0]]) / np.linalg.norm(d)
    t = np.linspace(0, 1, n * waves + 1)
    return [tuple(a + d * ti + v * amp * np.sin(2 * np.pi * waves * ti)) for ti in t]


def rook():
    """The strong and the weak nuclear force. On top of the tower stands a proton: three quarks
    held together by curly gluon lines (the strong force). Two walls carry the Feynman diagram of
    beta decay, a neutron turning into a proton by sending out a W boson that splits into an
    electron and an anti-neutrino (the weak force); the other two walls carry two quarks swapping
    a gluon."""
    base, z0 = dial_base(15.0)
    s, h = 17.0, 22.0
    tower = chamfer_slab(s, s, h, 1.2).translate((0, 0, z0 - 0.01))

    def drawing(lines, dots, labels, word):
        cs = None
        for pl in lines:
            part = strokes(pl, 0.6)
            cs = part if cs is None else cs + part
        for x, z in dots:  # the vertices
            cs = cs + CrossSection.circle(1.15, 24).translate((x, z))
        for t, x, z, hh in labels:
            cs = cs + text_cs(t, hh).translate((x, z))
        return cs + text_cs(word, 2.3).translate((0, 1.6))

    weak = drawing([[(-4.0, 7.2), (-2.5, 11.5)], [(-2.5, 11.5), (-4.4, 18.0)],  # neutron in, proton out
                    wavy((-2.5, 11.5), (2.5, 14.5), 3, 0.75),  # W boson
                    [(2.5, 14.5), (4.8, 18.0)], [(2.5, 14.5), (1.0, 18.0)]],  # electron, anti-neutrino
                   [(-2.5, 11.5), (2.5, 14.5)],
                   [("n", -5.6, 5.0, 2.8), ("p", -6.0, 14.2, 2.8), ("W", 2.3, 9.6, 2.3), ("e", 5.6, 18.6, 2.6),
                    ("v", 0.8, 18.6, 2.6)], "WEAK")
    strong = drawing([[(-4.5, 6.4), (-4.5, 18.0)], [(4.5, 6.4), (4.5, 18.0)], curly((-4.5, 11.6), (4.5, 11.6), 3, 1.15)],
                     [(-4.5, 11.6), (4.5, 11.6)],
                     [("u", -4.5, 18.6, 2.5), ("d", 4.5, 18.6, 2.3), ("g", 0.0, 8.6, 2.6)], "STRONG")
    inner = box(s - 2.0, s - 2.0, h + 2, z0=z0 - 1)  # 1 mm deep grooves
    cuts = [engrave_front(weak.translate((0, z0)), 0, 0, inner, turn) for turn in (0, 180)]
    cuts += [engrave_front(strong.translate((0, z0)), 0, 0, inner, turn) for turn in (90, -90)]
    tower = tower - union(cuts)
    top, floor = battlements(s, z0 + h)
    # the proton, raised on a dais so it shows over the wall: three quarks standing in a triangle
    # facing the front, each pair held by a gluon spring wound round a rod
    dz = floor + 3.0
    parts = [base, tower, top, cyl(3.3, 8.6, z0=floor - 0.3, seg=64)]
    rq, side = 2.8, 10.0
    zq = dz + rq - 0.5
    q = [np.array([-side / 2, 0, zq]), np.array([side / 2, 0, zq]), np.array([0, 0, zq + side * np.sqrt(3) / 2])]
    labels = ["u", "u", "d"]
    for p, lab in zip(q, labels):
        parts.append(sphere(rq, *p, seg=48))
    for i, j in ((0, 1), (0, 2), (1, 2)):
        a, b = q[i].copy(), q[j].copy()
        if i == 0 and j == 1:  # the bottom pair: the rod lies on the dais, so it never hangs in the air
            a[2] = b[2] = dz + 0.6
        u = (b - a) / np.linalg.norm(b - a)
        a, b = a + u * (rq - 0.6), b - u * (rq - 0.6)
        L = np.linalg.norm(b - a)
        v = np.cross(u, [0, 1, 0]) if abs(u[1]) < 0.9 else np.cross(u, [1, 0, 0])
        v /= np.linalg.norm(v)
        w = np.cross(u, v)
        th = np.linspace(0, 2 * np.pi * 4, 4 * 16 + 1)
        coil = [tuple(a + u * L * t / th[-1] + 1.5 * (v * np.cos(t) + w * np.sin(t))) for t in th]
        parts += [tube([tuple(a), tuple(b)], 0.8, seg=12), tube(coil, 0.5, seg=10)]
    m = union(parts)
    for p, lab in zip(q, labels):  # u, u, d engraved on the front of each quark
        hh = 2.4 if lab == "d" else 3.2
        zc = p[2] - (0.5 if lab == "d" else 0.32) * hh
        m = m - engrave_front(text_cs(lab, hh, 0.7), p[0], zc, sphere(rq - 0.6, *p, seg=48), reach=4.0)
    return m


# ---------------------------------------------------------------- KNIGHT
# the mane ends 5 mm above the pedestal, where the neck still carries it (lower down it would hang past the pedestal)
BACK = [(4.6, 42.0), (6.2, 40.6), (8.8, 36.8), (11.0, 31.0), (11.8, 25.0), (11.2, 18.0), (11.0, 11.0), (9.8, 4.0)]


def back_path(n):
    """The back of the neck, resampled to n points (x, z), top first."""
    P = np.array(BACK)
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    s = np.linspace(0, d[-1], n)
    return np.interp(s, d, P[:, 0]), np.interp(s, d, P[:, 1]), s


def knight():
    """Quantum tunnelling: the knight is the one piece that gets through where others are blocked.
    On both sides of the head a wave runs in from the back, meets a raised barrier, dies down
    inside it and comes out the front side smaller but still there. The eye is psi, and the mane
    is a row of beads: quantised energy levels."""
    base, z0 = dial_base(16.0)
    ped_h = 2.5
    parts = [base, chamfer_slab(22, 13, ped_h + 0.3, 1.5, ch_top=0.6).translate((0, 0, z0 - 0.3))]
    zb, th, dx = z0 + ped_h - 0.3, 12.0, 2.0
    # the throat widens toward the pedestal (widening downward needs no supports): a 3 mm neck would snap
    outline = KNIGHT_PTS + [(-4.6, 10.0), (-6.5, 6.0), (-8.0, 2.5), (-8.5, -3.0)]
    shape = CrossSection([ccw(smooth_closed(outline, 6))])
    head = to_xz(bevel_extrude(shape, th, bev=1.2), th)
    parts.append((head ^ box(80, 40, 80, z0=0)).translate((dx, 0, zb)))
    for sy in (-1, 1):
        parts.append(emb_xz(PSI, 4.0, 1.6).translate((-5.8 + dx, sy * (th / 2 - 0.2), 33.0 + zb)))
        parts.append(rounded_box(3.0, 3.0, 8.0, 0.6, seg=10).translate((3.0 + dx, sy * 2.6, 42.5 + zb)))  # ears
    # the barrier: a raised block 1.2 mm proud of each face; the wave on the face runs through it
    xb0, xb1, zw, A, lam = -1.0, 3.0, 20.0, 2.4, 5.0
    for sy in (-1, 1):
        parts.append(box(xb1 - xb0, 1.4, 13.0, cx=(xb0 + xb1) / 2 + dx, cy=sy * (th / 2 + 0.5), z0=zb + zw - 6.5))
        pts = []
        for x in np.arange(9.0, -11.0, -0.3):
            if x > xb1:
                a, yo = A, 0.0
            elif x >= xb0:
                a, yo = A * np.exp(-(xb1 - x) / 3.2), 1.2
            else:
                a, yo = A * np.exp(-(xb1 - xb0) / 3.2), 0.0
            pts.append((x + dx, sy * (th / 2 + yo + 0.05), zb + zw + a * np.sin(2 * np.pi * (x - xb1) / lam)))
        parts.append(tube(pts, 0.7, seg=12))
    x, z, s = back_path(18)  # the mane: a row of beads
    for xi, zi in zip(x, z):
        parts.append(sphere(2.1, xi + dx + 0.6, 0, zi + zb, seg=20))
    cuts = face_grooves([[(-14.0, 24.6)]], 1.8, 1.0, th, dx, zb)  # nostril
    cuts += face_grooves([[(-16.6, 21.0), (-13.0, 19.6)]], 0.9, 0.9, th, dx, zb)  # mouth
    return union(parts) - union(cuts)


# ---------------------------------------------------------------- BISHOP
MITRE = [(7.6, 31.2), (8.3, 32.6), (8.5, 34.0), (8.3, 36.5), (7.6, 39.5), (6.4, 42.5), (4.8, 45.3), (3.0, 47.6),
         (1.8, 48.8), (0, 48.8)]


def bishop():
    """The speed of light. The bishop moves on diagonals, and light moves on the 45 degree lines of
    space-time. Two rings tilted at +45 and -45 degrees cross at an event on the mitre: its light
    cone. The slit is the light ray leaving the event at 45 degrees. A photon is the ball on top."""
    base, z0 = dial_base(15.0)
    prof = [(0, 0), (9.5, 0), (9.5, 1.4), (8.2, 2.8), (6.5, 7.0), (4.6, 19.0), (4.2, 23.6), (6.6, 26.0), (6.6, 27.2),
            (4.8, 28.4)] + MITRE
    body = revolve(prof).translate((0, 0, z0 - 0.01))
    body = body - union([ring_groove(6.6, 0.5, 0.6, z0 + 26.6), ring_groove(9.5, 0.5, 0.5, z0 + 0.7)])
    ze = z0 + 40.5  # the event, on the axis
    # the light ray: a 1.8 mm slit from the event down to the +x side at 45 degrees (both faces print at 45).
    # It stops 3 mm short of the sides, so the mitre above it stays tied on: a slit cut right
    # through would leave the top resting on a thin wedge
    slit = box(14.0, 10.0, 1.8, z0=-0.9).translate((7.0, 0, 0)).rotate((0, 45, 0)).translate((0, 0, ze))
    # the light cone: two 0.9 mm grooves, 0.8 mm deep, where the +-45 degree planes through the event meet the mitre
    inner = revolve([(0, 29.0)] + [(max(r - 0.8, 0.0), z) for r, z in MITRE[:-2]] + [(0, 47.6)]).translate((0, 0, z0 - 0.01))
    planes = union([box(40, 40, 0.9, z0=-0.45).rotate((0, ang, 0)).translate((0, 0, ze)) for ang in (45, -45)])
    cone = (planes ^ box(40, 40, 16.0, z0=z0 + 31.8)) - inner
    # c, the speed of light, cut big into the front and back of the stem
    c_inner = shrunk(prof, 0.8).translate((0, 0, z0 - 0.01))
    big_c = [engrave_front(text_cs("c", 9.5, 1.3), 0.0, z0 + 9.8, c_inner, turn) for turn in (0, 180)]
    ball = sphere(2.7, z=z0 + 48.8 + 1.7, seg=40)
    events = [sphere(1.3, 0, sy * 7.0, ze, seg=24) for sy in (-1, 1)]  # the event, where the two light lines cross
    return union([base, body - slit - cone - union(big_c), ball] + events)


# ---------------------------------------------------------------- QUEEN
def field_crown(zt, n=6, H=15.0, rr=1.6):
    """The magnet's field lines as a crown: n ribs rise from the body (the S pole), bulge out and
    close in on the N pole ball on top. No rib leans more than 45 degrees, so it prints without
    supports. A ring at the waist ties the ribs together. Returns (parts, z of the N pole)."""
    t = np.linspace(0, 1, 30)
    r = 8.0 - 6.0 * t + 10.0 * t * (1 - t)
    parts = []
    for k in range(n):
        a = 2 * np.pi * k / n
        parts.append(tube([(ri * np.cos(a), ri * np.sin(a), zt + H * ti) for ri, ti in zip(r, t)], rr, seg=16))
    zr, rm = zt + 0.4 * H, 8.0
    parts.append(revolve([(rm - 1.3, 0), (rm, -1.3), (rm + 1.3, 0), (rm, 1.3)]).translate((0, 0, zr)))
    zn = zt + H + 1.4
    parts.append(sphere(3.6, z=zn, seg=48))
    return parts, zn


def queen():
    """The electromagnetic force: the electric and magnetic fields of a light wave climb the body as
    two sine-shaped fins at right angles, and the crown is the field lines of a magnet, rising from
    the body (its S pole) and closing on the ball on top (its N pole)."""
    base, z0 = dial_base(16.5)
    rcol, za, zb, A = 5.0, 10.0, 44.0, 8.5
    body = revolve([(0, 0), (11, 0), (11, 1.6), (9.5, 3.5), (6.5, 7.0), (rcol, za), (rcol, 47.6),
                    (9.6, 52.2), (9.6, 53.6), (0, 53.6)]).translate((0, 0, z0 - 0.01))
    body = body - ring_groove(9.6, 0.5, 0.6, z0 + 52.9)
    lam = zb - za
    k = 2 * np.pi / lam
    fin_cs = None
    for lo in (za, za + lam / 2):
        zz = np.linspace(lo, lo + lam / 2, 30)
        c = CrossSection([ccw([(float(A * np.sin(k * (z - za))), float(z)) for z in zz])])
        fin_cs = c if fin_cs is None else fin_cs + c
    fin = to_xz(Manifold.extrude(fin_cs, 2.0), 2.0)
    rim = tube([(A * np.sin(k * (z - za)), 0.0, z) for z in np.linspace(za, zb, 60)], 1.1, seg=12)
    fin = (fin + rim).translate((0, 0, z0))
    parts = [base, body, fin, fin.rotate((0, 0, 90))]
    crown, zn = field_crown(z0 + 53.6 - 0.3)
    m = union(parts + crown)
    col = cyl(60, rcol - 0.7, z0=z0)  # letters 0.7 mm deep
    m = m - union([engrave_front(text_cs("N", 3.0, 0.7), 0, zn - 1.5, sphere(3.0, z=zn, seg=48), reach=5.0),  # N pole on top
                   engrave_front(text_cs("S", 2.6, 0.7), 0, z0 + 44.7, col, reach=8.0),  # S pole under the crown
                   engrave_front(text_cs("E", 4.2, 0.8), 0, z0 + 15.0, col, reach=8.0),  # electric field
                   engrave_front(text_cs("B", 4.2, 0.8), 0, z0 + 32.0, col, turn=90, reach=8.0)])  # magnetic field
    return m


# ---------------------------------------------------------------- KING
KING_BODY = [(0, 0), (12, 0), (12, 1.5), (10.4, 3), (8.4, 8), (6.4, 30)]


def king():
    """Gravity well: the body opens into a funnel of curved space-time (the grid is cut into it),
    a ringed planet sits in its mouth, and the cross stands on the planet."""
    base, z0 = dial_base(17.5)
    zf, rn = 43.6, 5.8
    body = revolve(KING_BODY + [(rn, zf + 0.4), (0, zf + 0.4)]).translate((0, 0, z0 - 0.01))
    body = body - union([ring_groove(r, 0.5, 0.7, z0 + z) for r, z in ((8.0, 10.0), (6.95, 24.0))])
    f = lambda t: 0.47 * t + 0.53 * (1 - (1 - t) ** 2)  # slope never flatter than 45 degrees
    ts = np.linspace(0, 1, 24)
    outer = [(rn + 6.6 * t, zf + 14 * f(t)) for t in ts]
    rim, ztop = 11.0, zf + 14.6
    D, a = 9.0, 3.2
    well = lambda r: ztop - D * (1 / (1 + (r / a) ** 2) - 1 / (1 + (rim / a) ** 2))
    inner = [(r, well(r)) for r in np.linspace(rim, 0, 30)]
    funnel = revolve([(0, zf), (rn, zf)] + outer[1:] + [(12.4, ztop)] + inner)
    # the space-time grid: rings and meridians cut 0.6 mm into the outside of the funnel
    inset = revolve([(0, zf - 1), (rn - 0.6, zf - 1)] + [(r - 0.6, z) for r, z in outer] + [(11.8, ztop + 1), (0, ztop + 1)])
    skin = cyl(20, 20, z0=zf - 2) - inset
    cuts = []
    for t in (0.3, 0.55, 0.8):
        r, z = rn + 6.6 * t, zf + 14 * f(t)
        cuts.append(cyl(0.7, 20, z0=z - 0.35) - cyl(0.9, r - 0.6, z0=z - 0.45))
    for k in range(12):
        cuts.append((box(14, 0.8, 11.5, cx=7.0, z0=zf + 1.0) ^ skin).rotate((0, 0, 30 * k)))
    funnel = (funnel - union(cuts)).translate((0, 0, z0 - 0.01))
    zp, rp = z0 + ztop + 1.5, 6.5
    planet = sphere(rp, z=zp, seg=64) + cyl(zp - z0 - zf - 6.0, 2.6, z0=z0 + zf + 6.0)
    planet = planet - ring_groove(np.sqrt(rp ** 2 - 3.2 ** 2), 0.5, 0.7, zp + 3.2)  # a cloud band
    # the planet's ring: its underside slopes at 45 degrees, so it prints without supports
    ring = revolve([(rp - 1.2, -2.2), (rp + 2.6, 1.6), (rp + 2.6, 2.3), (rp - 1.2, 2.3)]).translate((0, 0, zp))
    m = union([base, body, funnel, planet, ring, cross_top(zp + rp - 1.0)])
    inner = shrunk(KING_BODY + [(rn, zf + 0.4), (0, zf + 0.4)], 0.8).translate((0, 0, z0 - 0.01))
    return m - engrave_front(text_cs("G", 7.5, 1.2), 0.0, z0 + 13.0, inner)  # G, the gravitational constant


# ---------------------------------------------------------------- set
PIECES = [  # (type, function, name, count per colour)
    ("pawn", pawn, "atom", 8),
    ("rook", rook, "nuclear_forces", 2),
    ("knight", knight, "tunnelling", 2),
    ("bishop", bishop, "speed_of_light", 2),
    ("queen", queen, "electromagnetism", 1),
    ("king", king, "gravity", 1),
]
ORDER = ["rook", "knight", "bishop", "king", "queen", "bishop", "knight", "rook"] + ["pawn"] * 8


def write_stl(m, path):
    mesh = m.to_mesh()
    v = np.asarray(mesh.vert_properties)[:, :3].astype(np.float32)
    tri = v[np.asarray(mesh.tri_verts).astype(np.int64)]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n = (n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)).astype(np.float32)
    rec = np.zeros(len(tri), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = n, tri
    with open(path, "wb") as f:
        f.write(b"Physics chess set by GIAMAT13".ljust(80, b" "))
        f.write(np.uint32(len(tri)).tobytes())
        f.write(rec.tobytes())


def stem(kind, name):
    return f"{kind}_{name}"


def build(only=None):
    os.makedirs(OUT, exist_ok=True)
    built = {}
    jobs = [p for p in PIECES if not only or p[0] in only]
    for i, (kind, fn, name, cnt) in enumerate(jobs):
        print(f"@@now building {kind} ({i + 1} of {len(jobs)})", flush=True)
        m = fn()
        m = Manifold.compose([p for p in m.decompose() if p.volume() > 1.0])
        if m.status().name != "NoError":
            raise SystemExit(f"ERROR {kind}: {m.status()}")
        bb = m.bounding_box()
        write_stl(m, os.path.join(OUT, stem(kind, name) + ".stl"))
        built[kind] = m
        print(f"{stem(kind, name)}: parts {len(m.decompose())}, genus {m.genus()}, "
              f"size {bb[3]-bb[0]:.1f} x {bb[4]-bb[1]:.1f} x {bb[5]-bb[2]:.1f} mm, x{cnt} per colour", flush=True)
    if not only:
        plate = [built[k].translate(((j % 4 - 1.5) * 44.0, (j // 4 - 1.5) * 44.0, 0)) for j, k in enumerate(ORDER)]
        write_stl(Manifold.compose(plate), os.path.join(OUT, "plate_16_pieces.stl"))
        print("plate_16_pieces: done", flush=True)


def _load_stl(path):
    with open(path, "rb") as f:
        f.read(80)
        n = np.frombuffer(f.read(4), np.uint32)[0]
        rec = np.frombuffer(f.read(), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")], count=n)
    return rec["v"], rec["n"]


COLOURS = {"white": (0.95, 0.85, 0.62), "black": (0.42, 0.42, 0.50)}


def render():
    """images/preview_physics.png: the six pieces in white on top, the same pieces in black below."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    bg = "#1b1f27"
    fig = plt.figure(figsize=(21, 9.6), facecolor=bg)
    light = np.array([0.4, -0.7, 0.6])
    light /= np.linalg.norm(light)
    for col, (kind, fn, name, cnt) in enumerate(PIECES):
        tri, nrm = _load_stl(os.path.join(OUT, stem(kind, name) + ".stl"))
        shade = np.clip(nrm @ light, 0, 1) * 0.8 + 0.2
        pts = tri.reshape(-1, 3)
        c = (pts.min(0) + pts.max(0)) / 2
        for row, side in enumerate(COLOURS):
            rgb = COLOURS[side]
            fc = np.stack([rgb[0] * shade + 0.05, rgb[1] * shade + 0.05, rgb[2] * shade + 0.05, np.ones_like(shade)], 1)
            ax = fig.add_subplot(2, 6, row * 6 + col + 1, projection="3d", facecolor=bg)
            ax.add_collection3d(Poly3DCollection(tri, facecolors=np.clip(fc, 0, 1), edgecolors="none"))
            r = 92 / 1.6 * 1.05  # same scale for every piece, so heights compare
            ax.set_xlim(c[0] - r / 2, c[0] + r / 2)
            ax.set_ylim(c[1] - r / 2, c[1] + r / 2)
            ax.set_zlim(0, r * 1.6)
            ax.set_box_aspect((1, 1, 1.6))
            ax.view_init(elev=14, azim=-90 if kind in ("knight", "queen") else -62)
            ax.set_axis_off()
            ax.set_title(f"{kind.capitalize()}: {name.replace('_', ' ')}", color="white", fontsize=12, pad=0)
    plt.subplots_adjust(left=0, right=1, top=0.93, bottom=0, wspace=0, hspace=0.08)
    path = os.path.join(ROOT, "images", f"preview_{VERSION}.png")
    plt.savefig(path, dpi=100, facecolor=bg)
    print("preview saved:", path, flush=True)


if __name__ == "__main__":
    only = [a for a in sys.argv[1:] if not a.startswith("-")]
    build(only)
    if not only:
        render()
