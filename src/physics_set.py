"""
Physics chess set - white is classical physics, black is quantum physics.

  piece    white (classical)                     black (quantum)
  Pawn     atom: nucleus and three orbits         p orbital: two lobes, + and - phase
  Rook     Tesla coil with a top-load torus       tunnelling: a wave through a barrier wall
  Knight   horse, horseshoe-magnet ear, wave mane horse in superposition, quantised mane
  Bishop   prism splitting a ray of light         Schrodinger's box, a cat peeking out
  Queen    electromagnetic wave (E and B)         double slit with its interference fringes
  King     gravity well with a planet             Bloch sphere (a qubit), state arrow

Both sides stand on a round dial base: ticks for classical, energy-level grooves for quantum.
Same height and outline per piece type, so the two sides read as one set.

Usage:  python src/physics_set.py            # all 12 pieces, 2 plates, preview
        python src/physics_set.py pawn king  # only these piece types (no plates)
Output: stl/physics/classical/, stl/physics/quantum/, images/preview_physics.png.
Check:  python src/print_check.py physics/classical physics/quantum
"""
import os
import sys

import numpy as np
from manifold3d import CrossSection, JoinType, Manifold

from build_set import (KNIGHT_PTS, ROOT, bevel_extrude, box, ccw, chamfer_slab, cyl, revolve, ring_groove,
                       rounded_box, smooth_closed, sphere, to_xz, tube, union, ycyl)

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
    """The wave function letter psi."""
    arc = [(4.6 * np.cos(t), 1.0 + 4.6 * np.sin(t)) for t in np.linspace(np.pi, 2 * np.pi, 13)]
    cup = [(-5.2, 6.4)] + arc + [(5.2, 6.4)]
    return strokes(cup, 1.3) + strokes([(0, -8.0), (0, 8.2)], 1.3) + strokes([(-2.8, -8.0), (2.8, -8.0)], 1.3)


ATOM, PSI = atom_cs(), psi_cs()


def emb_xz(cs, radius, thick):
    """Emblem standing up in the x-z plane, centred on the origin."""
    f = radius / 8.8
    return Manifold.extrude(cs.scale((f, f)), thick).rotate((90, 0, 0)).translate((0, thick / 2, 0))


def medallion(cs, radius, thick):
    """Emblem on a round coin (thinner than the emblem, so it stands out on both faces)."""
    reach = max(np.hypot(*np.array(p).T).max() for p in cs.to_polygons()) * radius / 8.8
    return emb_xz(cs, radius, thick) + ycyl(reach + 0.5, -thick / 2 + 0.8, thick / 2 - 0.8, 0.0, seg=64), reach + 0.5


# ---------------------------------------------------------------- base
BASE_TOP = 6.6


def dial_base(R, quantum):
    """Round base like a measuring dial. Classical: tick marks round the rim.
    Quantum: two grooves round the side (energy levels). Returns (manifold, z where the piece starts)."""
    m = revolve([(0, 0), (R - 0.5, 0), (R, 0.5), (R, 3.6), (R - 1.8, 5.4), (R - 2.6, 5.4), (R - 3.6, BASE_TOP), (0, BASE_TOP)])
    if quantum:
        cuts = [ring_groove(R, 0.6, 0.7, z) for z in (1.5, 2.7)]
    else:
        cuts = []
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


# ---------------------------------------------------------------- PAWN
def pawn_classic():
    """Atom: a nucleus of protons and neutrons, three tilted electron orbits resting on the collar."""
    base, z0 = dial_base(13.0, False)
    zc = 20.2  # collar top (local)
    body = revolve([(0, 0), (8.5, 0), (8.5, 1.0), (6.0, 3.0), (4.0, 10.0), (3.6, 14.4), (8.6, 19.4), (8.6, zc), (0, zc)])
    parts = [base, body.translate((0, 0, z0 - 0.01))]
    # the orbits run round the nucleus just inside its surface, so they are fused to it all the way
    # round instead of standing as free arches (those would snap off)
    Ro, rt, tilt, Rn = 7.2, 1.5, np.radians(35), 5.9
    hc = z0 + zc + Ro * np.sin(tilt) + rt - 0.25  # lowest point of every orbit sits on the collar
    parts.append(cyl(hc - z0 - zc + 0.6, 3.6, z0=z0 + zc - 0.3))
    parts.append(sphere(Rn, z=hc, seg=48))
    for k in range(7):  # protons and neutrons bulging out of the nucleus
        a, el = np.radians(51.4 * k), np.radians(55 if k % 2 else 20)
        parts.append(sphere(1.7, Rn * np.cos(el) * np.cos(a), Rn * np.cos(el) * np.sin(a), hc + Rn * np.sin(el), seg=20))
    t = np.linspace(0, 2 * np.pi, 49)
    for k in range(3):
        pts =np.stack([Ro * np.cos(t), Ro * np.sin(t) * np.cos(tilt), Ro * np.sin(t) * np.sin(tilt)], 1)
        a = np.radians(120 * k + 30)
        rot = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        pts = pts @ rot.T + [0, 0, hc]
        parts.append(tube([tuple(p) for p in pts], rt, seg=16))
        top = np.array([0, Ro * np.cos(tilt), Ro * np.sin(tilt) + 0.35]) @ rot.T + [0, 0, hc]
        parts.append(sphere(1.9, *top, seg=20))  # the electron, on top of its orbit
    return union(parts)


def pawn_quantum():
    """p orbital: two lobes meeting at the nucleus, marked with the + and - phase of the wave function."""
    base, z0 = dial_base(13.0, True)
    zs, h, W, rw = 12.0, 10.5, 6.6, 3.0  # rw: waist radius at the node, thick enough not to snap

    def lobe(u):  # u = 0 at the far end of a lobe, 1 at the nucleus
        return W * np.sin(np.pi * u ** 0.75) ** 0.7

    prof = [(0, 0), (8.5, 0), (8.5, 1.0), (6.0, 3.0), (4.6, 8.0), (3.9, zs)]  # stem thick enough not to snap
    for u in np.linspace(0.04, 1.0, 24):
        prof.append((max(lobe(u), 3.9 if u < 0.3 else rw), zs + h * u))
    for u in np.linspace(0.0, 1.0, 25)[1:-1]:
        prof.append((max(lobe(1 - u), rw if u < 0.5 else 0.0), zs + h + h * u))
    prof.append((0, zs + 2 * h))
    m = base + revolve(prof).translate((0, 0, z0 - 0.01))
    m = m + sphere(3.5, z=z0 + zs + h, seg=32)  # the nucleus at the node
    cuts = []
    zp, zm = z0 + zs + h + h * (1 - 0.5 ** (4 / 3)), z0 + zs + h * 0.5 ** (4 / 3)
    for sy in (-1, 1):
        cy = sy * (W + 1.0)
        cuts += [box(4.2, 4.0, 1.2, cy=cy, z0=zp - 0.6), box(1.2, 4.0, 4.2, cy=cy, z0=zp - 2.1),  # +
                 box(4.2, 4.0, 1.2, cy=cy, z0=zm - 0.6)]  # -
    return m - union(cuts)


# ---------------------------------------------------------------- ROOK
def battlements(s, z, cap_w=23.0, wall_h=6.0):
    """Square cap flaring out at 45 degrees and a notched wall on a tower of width s ending at z."""
    from build_set import rrect
    flare = (cap_w - s) / 2
    cap = Manifold.extrude(rrect(s, s, 1.2), flare, 0, 0.0, (cap_w / s, cap_w / s)).translate((0, 0, z - 0.3))
    cap = cap + chamfer_slab(cap_w, cap_w, 2.5, 1.4).translate((0, 0, z + flare - 0.6))
    z += flare + 1.9
    ring = chamfer_slab(cap_w, cap_w, wall_h, 1.4, ch_top=0.5).translate((0, 0, z - 0.3)) - box(
        cap_w - 5, cap_w - 5, wall_h + 1, z0=z + 0.6)
    ring = ring - union([box(6.0, cap_w + 2, 3.6, z0=z + wall_h - 3.4).rotate((0, 0, a)) for a in (0, 90)])
    return cap + ring


def rook_classic():
    """Tesla coil: primary winding at the foot, a tall secondary coil, and the top-load torus
    whose notches make the battlements. The spark terminal stands in the middle."""
    base, z0 = dial_base(15.0, False)
    rc, H = 8.0, 28.0
    body = revolve([(0, 0), (10.0, 0), (10.0, 1.2), (rc, 3.2), (rc, H), (0, H)])
    prim = revolve([(rc - 0.1, 4.0), (10.6, 6.6), (10.6, 9.6), (rc - 0.1, 9.6)])
    cap = revolve([(0, H - 0.2), (rc - 0.1, H - 0.2), (11.5, H + 3.3), (11.5, H + 4.5), (0, H + 4.5)])
    m = union([body, prim, cap]).translate((0, 0, z0 - 0.01))
    # ridges between the grooves stay at least 1.1 mm (3 nozzle lines)
    cuts = [ring_groove(10.6, 0.6, 0.7, z0 + z) for z in (6.9, 8.7)]
    cuts += [ring_groove(rc, 0.5, 0.7, z0 + z) for z in np.arange(11.5, H - 0.8, 1.8)]
    m = base + (m - union(cuts))
    Rt, rt = 7.8, 3.0
    zt = z0 + H + 4.5 + rt - 0.4
    torus = Manifold.revolve(CrossSection.circle(rt, 32).translate((Rt, 0)), 64).translate((0, 0, zt))
    torus = torus - union([box(4.0, 30, 10, z0=zt - 0.2).rotate((0, 0, a)) for a in (0, 90)])
    return union([m, torus, cyl(zt - z0 - H - 4.5 + 1.7, 2.2, z0=z0 + H + 4.2), sphere(2.8, z=zt + 2.2, seg=32)])


def rook_quantum():
    """Quantum tunnelling: on two faces a wave runs into a barrier (the raised band), dies down
    inside it and comes out the far side smaller. Battlements on top, wall bricks on the sides."""
    base, z0 = dial_base(15.0, True)
    s, h = 17.0, 30.0
    tower = chamfer_slab(s, s, h, 1.2).translate((0, 0, z0 - 0.01))
    adds, cuts = [], []
    zc = z0 + h * 0.5
    for sy in (-1, 1):
        adds.append(box(5.0, 2.4, h - 5.0, cy=sy * (s / 2), z0=z0 + 3.0))  # the barrier, 1.2 mm proud
        pts = []
        for x in np.arange(-s / 2 + 1.4, s / 2 - 1.4 + 1e-6, 0.35):
            if x < -2.5:
                A, yo = 3.0, 0.0
            elif x <= 2.5:
                A, yo = 3.0 * np.exp(-(x + 2.5) / 4.7), 1.2
            else:
                A, yo = 1.1, 0.0
            pts.append((x, sy * (s / 2 + 0.1 + yo), zc + A * np.sin(2 * np.pi * x / 5.0)))
        adds.append(tube(pts, 0.9, seg=12))
    for i in range(1, 6):  # wall courses on the sides
        for sx in (-1, 1):
            cuts.append(box(1.4, s + 2, 0.8, cx=sx * s / 2, z0=z0 + i * h / 6 - 0.4))
            for k in (-1, 1):  # staggered joints
                cy = k * 3.5 + (2.0 if i % 2 else -2.0)
                cuts.append(box(1.4, 0.8, h / 6 - 0.8, cx=sx * s / 2, cy=cy, z0=z0 + i * h / 6 + 0.4))
    tower = (tower - union(cuts)) + union(adds)
    return union([base, tower, battlements(s, z0 + h)])


# ---------------------------------------------------------------- KNIGHT
# the mane ends 5 mm above the pedestal, where the neck still carries it (lower down it would hang past the pedestal)
BACK = [(4.6, 42.0), (6.2, 40.6), (8.8, 36.8), (11.0, 31.0), (11.8, 25.0), (11.2, 18.0), (11.0, 11.0), (9.8, 4.0)]  # last point tucked into the neck


def back_path(n):
    """The back of the neck, resampled to n points (x, z), top first."""
    P = np.array(BACK)
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    s = np.linspace(0, d[-1], n)
    return np.interp(s, d, P[:, 0]), np.interp(s, d, P[:, 1]), s


def knight_head(quantum):
    base, z0 = dial_base(16.0, quantum)
    ped_h = 2.5
    parts = [base, chamfer_slab(22, 13, ped_h + 0.3, 1.5, ch_top=0.6).translate((0, 0, z0 - 0.3))]
    zb, th, dx = z0 + ped_h - 0.3, 12.0, 2.0
    # the throat widens toward the pedestal (widening downward needs no supports): a 3 mm neck would snap
    outline = KNIGHT_PTS + [(-4.6, 10.0), (-6.5, 6.0), (-8.0, 2.5), (-8.5, -3.0)]
    head = to_xz(bevel_extrude(CrossSection([ccw(smooth_closed(outline, 6))]), th, bev=1.2), th)
    parts.append((head ^ box(80, 40, 80, z0=0)).translate((dx, 0, zb)))
    for sy in (-1, 1):  # the eye
        parts.append(emb_xz(PSI if quantum else ATOM, 4.0, 1.6).translate((-5.8 + dx, sy * (th / 2 - 0.2), 33.0 + zb)))
    cuts = face_grooves([[(-14.0, 24.6)]], 1.8, 1.0, th, dx, zb) + face_grooves([[(-16.6, 21.0), (-13.0, 19.6)]], 0.9, 0.9, th, dx, zb)
    return parts, cuts, zb, th, dx


def knight_classic():
    """Horse with a horseshoe magnet for an ear and a mane that runs down the back as a wave."""
    parts, cuts, zb, th, dx = knight_head(False)
    xc, za, ra = 3.0 + dx, 47.6 + zb, 2.8
    arch = [(xc, ra, za - 4.6), (xc, ra, za)] + [(xc, ra * np.cos(t), za + ra * np.sin(t)) for t in np.linspace(0, np.pi, 13)[1:-1]]
    arch += [(xc, -ra, za), (xc, -ra, za - 4.6)]
    parts.append(tube(arch, 1.7, seg=16))
    x, z, s = back_path(44)
    parts.append(tube([(xi + dx + 0.6, 2.0 * np.sin(2 * np.pi * si / 12.0), zi + zb) for xi, zi, si in zip(x, z, s)], 1.7, seg=16))
    return union(parts) - union(cuts)


def knight_quantum():
    """Horse in superposition: the front of the head is smooth, the back half is drawn as
    waves. The mane is quantised, a row of beads."""
    parts, cuts, zb, th, dx = knight_head(True)
    for sy in (-1, 1):
        parts.append(rounded_box(3.0, 3.0, 8.0, 0.6, seg=10).translate((3.0 + dx, sy * 2.6, 42.5 + zb)))
    x, z, s = back_path(18)
    for xi, zi in zip(x, z):
        parts.append(sphere(2.1, xi + dx + 0.6, 0, zi + zb, seg=20))
    waves = []
    for zj in (6.0, 10.5, 15.0, 19.5):
        waves.append([(xx, zj + 1.0 * np.sin(2 * np.pi * xx / 4.0)) for xx in np.arange(1.0, 9.01, 0.4)])
    cuts += face_grooves(waves, 1.0, 0.8, th, dx, zb)
    return union(parts) - union(cuts)


# ---------------------------------------------------------------- BISHOP
def bishop_body(quantum):
    base, z0 = dial_base(15.0, quantum)
    body = revolve([(0, 0), (9.5, 0), (9.5, 1.4), (8.2, 2.8), (6.5, 7), (4.6, 20), (4.2, 25), (6.6, 26.2),
                    (6.6, 27.4), (4.6, 28.2), (5.4, 29), (5.4, 30.4), (0, 30.4)]).translate((0, 0, z0 - 0.01))
    # the top piece sits in a cone that widens at 45 degrees from the neck, so its corners never hang in the air
    cone = revolve([(0, -1), (4.8, -1), (30.8, 25), (0, 25)])
    return base + body, z0 + 30.0, cone


def bishop_classic():
    """Prism: a ray of white light enters on one side and leaves as a fan of colours.
    The atom emblem stands on top."""
    m, zs, cone = bishop_body(False)
    th = 10.0
    tri = CrossSection([ccw([(-8.5, 0), (8.5, 0), (0, 18)])]).offset(-1.5, JoinType.Round, 2.0, 24).offset(1.5, JoinType.Round, 2.0, 24)
    prism = (to_xz(bevel_extrude(tri, th, bev=1.2), th) ^ cone).translate((0, 0, zs - 0.3))
    rays = [[(-5.6, 5.2), (0.6, 6.2)]] + [[(0.6, 6.2), (8.5 * (1 - zz / 18) - 1.6, zz)] for zz in (3.2, 6.4, 9.6)]
    prism = prism - union(face_grooves(rays, 1.0, 0.8, th, 0.0, zs - 0.3))
    med, reach = medallion(ATOM, 4.2, 4.6)  # 3 mm coin: a thin coin snaps when pushed face-on
    apex = zs - 0.3 + 16.0
    return union([m, prism, med.translate((0, 0, apex + reach - 3.0))])


def bishop_quantum():
    """Schrodinger's box: a closed box with psi on its faces, and the cat's ears poking out under the lid."""
    m, zs, cone = bishop_body(True)
    bx = (rounded_box(13, 13, 12, 1.2) ^ cone).translate((0, 0, zs - 0.3))
    top = zs - 0.3 + 12.0
    lid = rounded_box(14.2, 14.2, 2.2, 0.6).translate((0, 0, top - 0.5))
    parts = [m, bx, lid]
    zl = top - 0.5 + 2.2 - 0.3
    for sx in (-1, 1):
        ear = CrossSection([ccw([(-2.3, 0), (2.3, 0), (sx * 0.9, 4.8)])]).offset(-0.5, JoinType.Round, 2.0, 12).offset(0.5, JoinType.Round, 2.0, 12)
        parts.append(to_xz(Manifold.extrude(ear, 2.6), 2.6).translate((sx * 3.4, 0, zl)))
    for sy in (-1, 1):
        parts.append(emb_xz(PSI, 4.0, 1.6).translate((0, sy * (6.5 - 0.2), zs - 0.3 + 6.2)))
    return union(parts)


# ---------------------------------------------------------------- QUEEN
def queen_crown_top(zt):
    """Nine spikes with round tips round the rim of the crown."""
    parts = []
    tilt = np.radians(16)
    for k in range(9):
        sp = cyl(7.0, 1.6, 1.2, z0=-0.5, seg=20).rotate((0, 16, 0)).translate((8.4, 0, zt))
        tip = sphere(1.5, 8.4 + 7 * np.sin(tilt), 0, zt + 7 * np.cos(tilt) - 0.6, seg=20)
        parts.append((sp + tip).rotate((0, 0, 360 / 9 * k)))
    return parts


def queen_classic():
    """Electromagnetic wave: the electric and magnetic fields as two sine-shaped fins at right
    angles, climbing the body. A crown of photons and the atom emblem on top."""
    base, z0 = dial_base(16.5, False)
    rcol, za, zb, A = 5.0, 10.0, 44.0, 8.5
    body = revolve([(0, 0), (11, 0), (11, 1.6), (9.5, 3.5), (6.5, 7.0), (rcol, za), (rcol, 47.6),
                    (9.6, 52.2), (9.6, 53.6), (0, 53.6)]).translate((0, 0, z0 - 0.01))
    lam = zb - za
    k = 2 * np.pi / lam
    fin_cs = None
    for lo in (za, za + lam / 2):
        zz = np.linspace(lo, lo + lam / 2, 30)
        poly = [(float(A * np.sin(k * (z - za))), float(z)) for z in zz]
        poly = ccw(poly)
        c = CrossSection([poly])
        fin_cs = c if fin_cs is None else fin_cs + c
    fin = to_xz(Manifold.extrude(fin_cs, 2.0), 2.0)
    rim = tube([(A * np.sin(k * (z - za)), 0.0, z) for z in np.linspace(za, zb, 60)], 1.1, seg=12)
    fin = (fin + rim).translate((0, 0, z0))
    parts = [base, body, fin, fin.rotate((0, 0, 90))]
    zt = z0 + 53.6
    parts += queen_crown_top(zt)
    med, reach = medallion(ATOM, 6.0, 4.8)
    parts += [cyl(6.5, 2.4, z0=zt - 1.3), med.translate((0, 0, zt + 5.2))]
    return union(parts)


def queen_quantum():
    """Double slit: a plane wave (rings) climbs the body to a wall with two slits; above it the
    bars of the interference pattern stand like a crown, brightest in the middle."""
    base, z0 = dial_base(16.5, True)
    rcol = 5.0
    body = revolve([(0, 0), (11, 0), (11, 1.6), (9.5, 3.5), (6.5, 7.0), (rcol, 10.0), (rcol, 40.0), (0, 40.0)])
    body = body.translate((0, 0, z0 - 0.01)) - union([ring_groove(rcol, 0.5, 0.8, z0 + z) for z in np.arange(13.0, 38.0, 3.0)])
    pt = 6.4  # wall thickness: a thinner wall bends at the slits
    wall =CrossSection([ccw([(-4.8, 39.6), (4.8, 39.6), (9.5, 44.3), (9.5, 52.0), (-9.5, 52.0), (-9.5, 44.3)])])
    wall = to_xz(Manifold.extrude(wall, pt), pt).translate((0, 0, z0))
    wall = wall - union([box(1.4, 10, 6.0, cx=sx * 2.2, z0=z0 + 44.5) for sx in (-1, 1)])
    parts = [base, body, wall]
    zt = z0 + 52.0 - 0.3
    for x, hgt in zip((-7.2, -3.6, 0.0, 3.6, 7.2), (2.5, 5.5, 8.5, 5.5, 2.5)):  # bars 3.2 x 4.0: strong enough to grab
        parts.append(box(3.2, 4.0, hgt, cx=x, z0=zt))
        parts.append(sphere(1.6, x, 0, zt + hgt, seg=20))  # photon cap, half sunk: no neck under it
    return union(parts)


# ---------------------------------------------------------------- KING
KING_BODY = [(0, 0), (12, 0), (12, 1.5), (10.4, 3), (8.4, 8), (6.4, 30)]


def king_classic():
    """Gravity well: the body opens into a funnel of curved space-time (the grid is cut into
    it), a planet sits in its mouth, and the cross stands on the planet."""
    base, z0 = dial_base(17.5, False)
    zf, rn = 43.6, 5.8
    body = revolve(KING_BODY + [(rn, zf + 0.4), (0, zf + 0.4)]).translate((0, 0, z0 - 0.01))
    f = lambda t: 0.47 * t + 0.53 * (1 - (1 - t) ** 2)  # slope never flatter than 45 degrees
    ts = np.linspace(0, 1, 24)
    outer = [(rn + 6.6 * t, zf + 14 * f(t)) for t in ts]
    rim, ztop = 11.0, zf + 14.6
    D, a = 9.0, 3.2
    well = lambda r: ztop - D * (1 / (1 + (r / a) ** 2) - 1 / (1 + (rim / a) ** 2))
    inner = [(r, well(r)) for r in np.linspace(rim, 0, 30)]
    prof = [(0, zf), (rn, zf)] + outer[1:] + [(12.4, ztop)] + inner
    funnel = revolve(prof)
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
    return union([base, body, funnel, planet, cross_top(zp + rp - 1.0)])


def king_quantum():
    """Bloch sphere: a qubit's sphere with its equator and two meridians, the state arrow
    leaving it at an angle, and the z axis rising out of the pole as the cross, topped by an arrowhead."""
    base, z0 = dial_base(17.5, True)
    body = revolve(KING_BODY + [(6.2, 41.5), (7.4, 42.7), (7.4, 44.0), (0, 44.0)]).translate((0, 0, z0 - 0.01))
    Rb = 10.0
    c = z0 + 44.0 - 0.3 + Rb * np.sin(np.radians(45))  # rests on the body at 45 degrees
    ball = sphere(Rb, z=c, seg=96)
    eq = revolve([(9.5, -1.4), (10.6, -0.3), (10.6, 0.9), (9.5, 0.9)]).translate((0, 0, c))
    mer = to_xz(Manifold.extrude(CrossSection.circle(10.6, 96) - CrossSection.circle(9.5, 96), 1.8), 1.8).translate((0, 0, c))
    parts = [base, body, ball, eq, mer, mer.rotate((0, 0, 90))]
    th, ph = 40.0, -60.0  # the state |psi>
    d = np.array([np.sin(np.radians(th)) * np.cos(np.radians(ph)), np.sin(np.radians(th)) * np.sin(np.radians(ph)), np.cos(np.radians(th))])
    head = Manifold.cylinder(5.1, 2.2, 0.3, 24).translate((0, 0, Rb - 0.6)).rotate((0, th, ph)).translate((0, 0, c))
    parts.append(head)
    parts.append(tube([(0, 0, c), tuple(np.array([0, 0, c]) + d * (Rb - 0.4))], 1.2, seg=12))
    zr = c + Rb - 0.8
    parts.append(cyl(5.0, 2.6, z0=zr))  # the z axis: thick, it carries the cross
    parts.append(cross_top(zr + 4.0, post_h=9.0))
    parts.append(cyl(3.6, 2.8, 0.3, z0=zr + 4.0 + 9.0 - 0.3, seg=32))  # arrowhead: the |0> axis
    return union(parts)


# ---------------------------------------------------------------- set
PIECES = [  # (type, white function, white name, black function, black name, count per side)
    ("pawn", pawn_classic, "atom", pawn_quantum, "p_orbital", 8),
    ("rook", rook_classic, "tesla_coil", rook_quantum, "tunnelling", 2),
    ("knight", knight_classic, "magnet", knight_quantum, "superposition", 2),
    ("bishop", bishop_classic, "prism", bishop_quantum, "schrodinger_box", 2),
    ("queen", queen_classic, "em_wave", queen_quantum, "double_slit", 1),
    ("king", king_classic, "gravity_well", king_quantum, "bloch_sphere", 1),
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


def stem(side, kind, name):
    return f"{side}_{kind}_{name}"


def build(only=None):
    for side in ("classical", "quantum"):  # one folder per side
        os.makedirs(os.path.join(OUT, side), exist_ok=True)
    built = {}
    jobs = [(kind, side, fn, name, cnt) for kind, wf, wn, bf, bn, cnt in PIECES
            for side, fn, name in (("classical", wf, wn), ("quantum", bf, bn)) if not only or kind in only]
    for i, (kind, side, fn, name, cnt) in enumerate(jobs):
        print(f"@@now building {side} {kind} ({i + 1} of {len(jobs)})", flush=True)
        m = fn()
        m = Manifold.compose([p for p in m.decompose() if p.volume() > 1.0])
        if m.status().name != "NoError":
            raise SystemExit(f"ERROR {side} {kind}: {m.status()}")
        bb = m.bounding_box()
        write_stl(m, os.path.join(OUT, side, stem(side, kind, name) + ".stl"))
        built[(side, kind)] = m
        print(f"{stem(side, kind, name)}: parts {len(m.decompose())}, genus {m.genus()}, "
              f"size {bb[3]-bb[0]:.1f} x {bb[4]-bb[1]:.1f} x {bb[5]-bb[2]:.1f} mm, x{cnt}", flush=True)
    if not only:
        for side in ("classical", "quantum"):
            plate = [built[(side, k)].translate(((j % 4 - 1.5) * 44.0, (j // 4 - 1.5) * 44.0, 0)) for j, k in enumerate(ORDER)]
            write_stl(Manifold.compose(plate), os.path.join(OUT, side, f"plate_{side}_16_pieces.stl"))
            print(f"plate_{side}_16_pieces: done", flush=True)


def _load_stl(path):
    with open(path, "rb") as f:
        f.read(80)
        n = np.frombuffer(f.read(4), np.uint32)[0]
        rec = np.frombuffer(f.read(), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")], count=n)
    return rec["v"], rec["n"]


def render():
    """images/preview_physics.png: white (classical) row on top, black (quantum) row below."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    bg = "#0e1117"
    fig = plt.figure(figsize=(21, 9.6), facecolor=bg)
    light = np.array([0.4, -0.7, 0.6])
    light /= np.linalg.norm(light)
    for row, side in enumerate(("classical", "quantum")):
        for col, (kind, wf, wn, bf, bn, cnt) in enumerate(PIECES):
            name = wn if side == "classical" else bn
            tri, nrm = _load_stl(os.path.join(OUT, side, stem(side, kind, name) + ".stl"))
            ax = fig.add_subplot(2, 6, row * 6 + col + 1, projection="3d", facecolor=bg)
            shade = np.clip(nrm @ light, 0, 1) * 0.8 + 0.2
            if side == "classical":
                col_ = np.stack([0.95 * shade + 0.05, 0.85 * shade + 0.05, 0.62 * shade + 0.04, np.ones_like(shade)], 1)
            else:
                col_ = np.stack([0.55 * shade + 0.05, 0.42 * shade + 0.05, 1.0 * shade, np.ones_like(shade)], 1)
            ax.add_collection3d(Poly3DCollection(tri, facecolors=np.clip(col_, 0, 1), edgecolors="none"))
            pts = tri.reshape(-1, 3)
            mn, mx = pts.min(0), pts.max(0)
            c = (mn + mx) / 2
            r = 92 / 1.6 * 1.05  # same scale for every piece, so heights compare
            ax.set_xlim(c[0] - r / 2, c[0] + r / 2)
            ax.set_ylim(c[1] - r / 2, c[1] + r / 2)
            ax.set_zlim(0, r * 1.6)
            ax.set_box_aspect((1, 1, 1.6))
            ax.view_init(elev=14, azim=-90 if kind in ("knight", "bishop", "queen") else -62)
            ax.set_axis_off()
            ax.set_title(f"{kind.capitalize()}: {name.replace('_', ' ')}", color="white", fontsize=12, pad=0)
    fig.text(0.01, 0.94, "WHITE  classical physics", color="#f3d9a0", fontsize=15)
    fig.text(0.01, 0.46, "BLACK  quantum physics", color="#9f8cff", fontsize=15)
    plt.subplots_adjust(left=0, right=1, top=0.93, bottom=0, wspace=0, hspace=0.08)
    path = os.path.join(ROOT, "images", f"preview_{VERSION}.png")
    plt.savefig(path, dpi=100, facecolor=bg)
    print("preview saved:", path, flush=True)


if __name__ == "__main__":
    only = [a for a in sys.argv[1:] if not a.startswith("-")]
    build(only)
    if not only:
        render()
