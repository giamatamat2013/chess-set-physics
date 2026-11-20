"""Physics chess set, modelled in Blender.

White and black share the same six models; only the print colour differs.

    pawn   - the atom: a nucleus of nucleons inside three electron orbits
    rook   - strong + weak nuclear force: a proton (u, u, d quarks held by gluon springs)
             on a tower engraved with Feynman diagrams (beta decay through a W boson, gluon exchange)
    knight - quantum tunnelling: a wave packet hits a barrier on the neck, decays inside it
             and leaks out on the far side with a smaller amplitude; psi eyes
    bishop - speed of light: the body is a pair of 45-degree light cones meeting at an event,
             the slit is a light ray, a "c" on the stem, a photon on top
    queen  - electromagnetic force: E and B wave fins in perpendicular planes up the column,
             a crown of magnetic field lines running from an N pole down to the collar
    king   - gravity: a gravity well (curved spacetime grid) holding a ringed planet, cross on top

Run inside Blender (blender --background --python src/blender_pieces.py) or exec() it
from a running Blender.  1 Blender unit = 1 mm.  Every piece prints without supports:
outward overhangs stay at 45 degrees or steeper.
"""
import math
import os

import bmesh
import bpy
from mathutils import Matrix, Vector

ROOT = globals().get("ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT_PATH = r"C:\Windows\Fonts\arialbd.ttf"
STL_DIR = os.path.join(ROOT, "stl", "physics")
IMG_DIR = os.path.join(ROOT, "images", "physics_pieces")
rad = math.radians


# ----------------------------------------------------------------------------- objects

def coll(name):
    c = bpy.data.collections.get(name)
    if c is None:
        c = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(c)
    return c


def new_obj(name, me, c="build"):
    ob = bpy.data.objects.new(name, me)
    coll(c).objects.link(ob)
    return ob


def from_bm(name, bm):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    return clean(new_obj(name, me))


def from_data(name, verts, faces):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], faces)
    me.validate()
    me.update()
    return clean(new_obj(name, me))


def delete(*obs):
    for ob in obs:
        data = ob.data
        bpy.data.objects.remove(ob, do_unlink=True)
        if data is not None and data.users == 0:
            if isinstance(data, bpy.types.Mesh):
                bpy.data.meshes.remove(data)
            elif isinstance(data, bpy.types.Curve):
                bpy.data.curves.remove(data)


def xform(ob, M):
    ob.data.transform(M)
    ob.data.update()
    return ob


def clean(ob, dist=1e-4):
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=dist)
    bmesh.ops.dissolve_degenerate(bm, edges=bm.edges, dist=1e-6)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(ob.data)
    bm.free()
    ob.data.update()
    return ob


def is_manifold(ob):
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    ok = len(bm.faces) > 0 and all(e.is_manifold for e in bm.edges)
    bm.free()
    return ok


def volume(ob):
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    v = bm.calc_volume(signed=True)
    bm.free()
    return v


def apply_mods(ob):
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
    old = ob.data
    ob.modifiers.clear()
    ob.data = me
    if old.users == 0:
        bpy.data.meshes.remove(old)
    return ob


def boolean(target, tools, op="UNION"):
    """target op= tools (a list); the tools are deleted."""
    tools = list(tools) if isinstance(tools, (list, tuple)) else [tools]
    if not tools:
        return target
    c = bpy.data.collections.new("bool_ops")
    bpy.context.scene.collection.children.link(c)
    for t in tools:
        c.objects.link(t)
    solvers = ["MANIFOLD", "EXACT"] if all(is_manifold(t) for t in tools + [target]) else ["EXACT"]
    backup = target.data.copy()
    for solver in solvers:
        m = target.modifiers.new("b", "BOOLEAN")
        m.operation = op
        m.operand_type = "COLLECTION"
        m.collection = c
        m.solver = solver
        apply_mods(target)
        if is_manifold(target) and volume(target) > 0:
            break
        print(f"boolean {op} with {solver} failed on {target.name}, retrying")
        old = target.data
        target.data = backup.copy()
        bpy.data.meshes.remove(old)
    bpy.data.meshes.remove(backup)
    delete(*tools)
    bpy.data.collections.remove(c)
    return target


def union(objs):
    return boolean(objs[0], objs[1:], "UNION")


def carve(target, cutters):
    """Subtract cutters one by one (a batched collection difference can tear holes)."""
    for c in cutters:
        boolean(target, [c], "DIFFERENCE")
    return target


# ----------------------------------------------------------------------------- 2D profiles

def arcp(cx, cz, r, a0, a1, n=8):
    return [(cx + r * math.cos(rad(a0 + (a1 - a0) * i / n)), cz + r * math.sin(rad(a0 + (a1 - a0) * i / n)))
            for i in range(n + 1)]


def bez(p0, p1, p2, p3, n=16):
    out = []
    for i in range(n + 1):
        t = i / n
        u = 1 - t
        out.append(tuple(u ** 3 * p0[k] + 3 * u * u * t * p1[k] + 3 * u * t * t * p2[k] + t ** 3 * p3[k]
                         for k in range(2)))
    return out


def chain(*parts):
    out = []
    for part in parts:
        for p in part:
            p = (max(p[0], 0.0), p[1])
            if not out or abs(out[-1][0] - p[0]) > 1e-6 or abs(out[-1][1] - p[1]) > 1e-6:
                out.append(p)
    return out


def smooth01(t):
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


def chaikin(pts, it=2):
    for _ in range(it):
        new = []
        n = len(pts)
        for i in range(n):
            p, q = pts[i], pts[(i + 1) % n]
            new.append((0.75 * p[0] + 0.25 * q[0], 0.75 * p[1] + 0.25 * q[1]))
            new.append((0.25 * p[0] + 0.75 * q[0], 0.25 * p[1] + 0.75 * q[1]))
        pts = new
    return pts


def foot(R):
    """Shared foot: rounded bottom edge and a bead with a 45-degree underside."""
    return chain([(0, 0), (R - 0.7, 0)], arcp(R - 0.7, 0.7, 0.7, -90, 0, 6), [(R, 1.35)],
                 arcp(R - 1.6, 3.4, 1.1, -45, 110, 12))


def base_prof(R, rt, zt):
    """Foot plus a cove that ends on the stem at radius rt, height zt."""
    b = foot(R)[-1]
    return chain(foot(R), bez(b, (R - 3.4, 4.0), (rt + 0.6, zt - 2.4), (rt, zt), 14))


def radius_at(prof, z):
    best = 0.0
    for (r0, z0), (r1, z1) in zip(prof, prof[1:]):
        if min(z0, z1) <= z <= max(z0, z1) and z1 != z0:
            best = max(best, r0 + (r1 - r0) * (z - z0) / (z1 - z0))
    return best


# ----------------------------------------------------------------------------- primitives

def lathe(name, pts, segs=160):
    verts, faces, rings = [], [], []
    for r, z in pts:
        if r < 1e-6:
            rings.append([len(verts)])
            verts.append((0, 0, z))
        else:
            rings.append(list(range(len(verts), len(verts) + segs)))
            for i in range(segs):
                a = 2 * math.pi * i / segs
                verts.append((r * math.cos(a), r * math.sin(a), z))
    for A, B in zip(rings, rings[1:]):
        if len(A) == 1 and len(B) == 1:
            continue
        for i in range(segs):
            j = (i + 1) % segs
            if len(A) == 1:
                faces.append((A[0], B[j], B[i]))
            elif len(B) == 1:
                faces.append((A[i], A[j], B[0]))
            else:
                faces.append((A[i], A[j], B[j], B[i]))
    return from_data(name, verts, faces)


def sphere(r, c=(0, 0, 0), seg=48):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=seg, v_segments=seg // 2, radius=r)
    bmesh.ops.translate(bm, vec=Vector(c), verts=bm.verts)
    return from_bm("sphere", bm)


def ellipsoid(rx, ry, rz, c):
    ob = sphere(1.0, seg=32)
    return xform(ob, Matrix.Translation(c) @ Matrix.Diagonal((rx, ry, rz, 1)))


def torus(R, r, M=Matrix(), nu=128, nv=20):
    verts, faces = [], []
    for i in range(nu):
        u = 2 * math.pi * i / nu
        for j in range(nv):
            v = 2 * math.pi * j / nv
            verts.append(((R + r * math.cos(v)) * math.cos(u), (R + r * math.cos(v)) * math.sin(u), r * math.sin(v)))
    for i in range(nu):
        for j in range(nv):
            a = i * nv + j
            b = ((i + 1) % nu) * nv + j
            c = ((i + 1) % nu) * nv + (j + 1) % nv
            d = i * nv + (j + 1) % nv
            faces.append((a, b, c, d))
    return xform(from_data("torus", verts, faces), M)


def prism(poly, z0, z1, name="prism"):
    """Polygon in XY extruded from z0 to z1."""
    n = len(poly)
    verts = [(x, y, z0) for x, y in poly] + [(x, y, z1) for x, y in poly]
    faces = [tuple(reversed(range(n))), tuple(range(n, 2 * n))]
    faces += [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    return from_data(name, verts, faces)


def loft(poly0, z0, poly1, z1, name="loft"):
    n = len(poly0)
    verts = [(x, y, z0) for x, y in poly0] + [(x, y, z1) for x, y in poly1]
    faces = [tuple(reversed(range(n))), tuple(range(n, 2 * n))]
    faces += [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    return from_data(name, verts, faces)


def box(x0, x1, y0, y1, z0, z1):
    return prism([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], z0, z1, "box")


def rounded_square(s, rc, n=6):
    h = s / 2 - rc
    pts = []
    for cx, cy, a0 in ((h, -h, -90), (h, h, 0), (-h, h, 90), (-h, -h, 180)):
        for i in range(n + 1):
            a = rad(a0 + 90 * i / n)
            pts.append((cx + rc * math.cos(a), cy + rc * math.sin(a)))
    return pts


def curve_to_mesh(ob):
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
    name = ob.name
    delete(ob)
    return clean(new_obj(name, me))


def tube(pts, r, cyclic=False, res=3, name="tube"):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_mode = "ROUND"
    cu.bevel_depth = r
    cu.bevel_resolution = res
    cu.use_fill_caps = not cyclic
    sp = cu.splines.new("POLY")
    sp.points.add(len(pts) - 1)
    for p, q in zip(sp.points, pts):
        p.co = (q[0], q[1], q[2], 1.0)
    sp.use_cyclic_u = cyclic
    return curve_to_mesh(new_obj(name, cu))


def rounded_tube(pts, r, res=3):
    """Tube with ball ends (unioned)."""
    return union([tube(pts, r, res=res), sphere(r, pts[0], 16), sphere(r, pts[-1], 16)])


def slab2d(outline, half, bevel, name="slab"):
    """Closed outline in the XZ plane, extruded +-half in y, edges rounded by `bevel`."""
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "2D"
    cu.fill_mode = "BOTH"
    cu.extrude = half - bevel
    cu.bevel_depth = bevel
    cu.bevel_resolution = 4
    sp = cu.splines.new("POLY")
    sp.points.add(len(outline) - 1)
    for p, (x, z) in zip(sp.points, outline):
        p.co = (x, z, 0, 1.0)
    sp.use_cyclic_u = True
    ob = curve_to_mesh(new_obj(name, cu))
    return xform(ob, Matrix.Rotation(rad(90), 4, "X"))


_font = None


def text(s, size, depth):
    """Text mesh centred on the origin in XY, extruded +-depth/2 in z."""
    global _font
    if _font is None:
        _font = bpy.data.fonts.load(FONT_PATH, check_existing=True)
    cu = bpy.data.curves.new("text", "FONT")
    cu.body = s
    cu.font = _font
    cu.size = size
    cu.extrude = depth / 2
    cu.resolution_u = 6
    ob = curve_to_mesh(new_obj("text", cu))
    xs = [v.co.x for v in ob.data.vertices]
    ys = [v.co.y for v in ob.data.vertices]
    return xform(ob, Matrix.Translation((-(min(xs) + max(xs)) / 2, -(min(ys) + max(ys)) / 2, 0)))


def front_text(s, size, z, x=0.0, reach=30.0, turn=0.0, tilt=0.0):
    """Text prism standing on the -y side from the axis outwards, rotated by `turn` about z."""
    t = text(s, size, reach)
    M = (Matrix.Rotation(rad(turn), 4, "Z") @ Matrix.Translation((x, -reach / 2, z))
         @ Matrix.Rotation(rad(tilt), 4, "Y") @ Matrix.Rotation(rad(90), 4, "X"))
    return xform(t, M)


def shell_cutter(cutter, inner):
    """Keep only the part of `cutter` outside `inner`: an engraving of even depth."""
    return boolean(cutter, [inner], "DIFFERENCE")


def finish(ob, name):
    ob.name = name
    ob.data.name = name
    ob.data.shade_smooth()
    ob.data.set_sharp_from_angle(angle=rad(32))
    for c in list(ob.users_collection):
        c.objects.unlink(ob)
    coll("pieces").objects.link(ob)
    print(name, "faces", len(ob.data.polygons), "manifold", is_manifold(ob), "volume", round(volume(ob), 1),
          "height", round(max(v.co.z for v in ob.data.vertices), 2))
    return ob


# ----------------------------------------------------------------------------- pieces

def fib_sphere(n):
    pts = []
    g = math.pi * (3 - math.sqrt(5))
    for i in range(n):
        z = 1 - 2 * (i + 0.5) / n
        r = math.sqrt(1 - z * z)
        pts.append((r * math.cos(g * i), r * math.sin(g * i), z))
    return pts


def pawn():
    prof = chain(base_prof(13, 5.4, 9.5), bez((5.4, 9.5), (4.6, 13), (3.8, 17.5), (3.7, 21.0)),
                 [(7.2, 24.5)], arcp(6.8, 24.7, 0.447, -26.6, 90, 6))
    top = prof[-1][1]
    prof = chain(prof, [(0, top)])
    parts = [lathe("pawn", prof)]
    zc = top + 4.6  # the rings' low points must land on the collar flare
    parts.append(sphere(2.2, (0, 0, zc), 32))
    # hidden core that carries the lower nucleons (otherwise some start 0.2 mm above the collar)
    parts.append(lathe("pawn_core", [(0, top - 0.3), (3.3, top - 0.3), (3.3, zc - 0.5), (0, zc)]))
    for i, p in enumerate(fib_sphere(18)):
        parts.append(sphere(2.05 if i % 2 else 1.85, (3.1 * p[0], 3.1 * p[1], zc + 3.1 * p[2]), 32))
    Ro, tilt, rt, re = 8.2, 38, 1.25, 1.7
    for az, phi in ((0, 205), (120, 75), (240, 320)):
        M = Matrix.Translation((0, 0, zc)) @ Matrix.Rotation(rad(az), 4, "Z") @ Matrix.Rotation(rad(tilt), 4, "X")
        parts.append(torus(Ro, rt, M))
        e = M @ Vector((Ro * math.cos(rad(phi)), Ro * math.sin(rad(phi)), 0))
        # raise the electron so it sits on its orbit instead of hanging below it
        parts.append(sphere(re, e + Vector((0, 0, re - rt)), 32))
    return finish(union(parts), "pawn")


def feynman_weak(h, turn):
    """Beta decay n -> p + e + anti-nu through a W boson, engraved on the face at -y (turned)."""
    d = 0.42
    cut = []
    Mf = Matrix.Rotation(rad(turn), 4, "Z")

    def P(u, v):
        return Mf @ Vector((u, -h, v))

    def line(a, b):
        cut.append(xform(rounded_tube([(a[0], -h, a[1]), (b[0], -h, b[1])], d), Mf))

    def arrow(at, ang):
        L, w = 1.9, 1.05
        ca, sa = math.cos(rad(ang)), math.sin(rad(ang))
        tri = [(at[0] + ca * L / 2, at[1] + sa * L / 2),
               (at[0] - ca * L / 2 - sa * w, at[1] - sa * L / 2 + ca * w),
               (at[0] - ca * L / 2 + sa * w, at[1] - sa * L / 2 - ca * w)]
        cut.append(face_prism(tri, h, 0.6, turn))

    V1, V2 = (-3.0, 17.0), (3.0, 21.0)
    line((-3.0, 9.6), V1)
    line(V1, (-3.0, 26.4))
    arrow((-3.0, 13.0), 90)
    arrow((-3.0, 22.4), 90)
    # W boson: wavy line V1 -> V2
    pts = []
    L = math.dist(V1, V2)
    ux, uy = (V2[0] - V1[0]) / L, (V2[1] - V1[1]) / L
    for i in range(61):
        t = i / 60
        s = 0.62 * math.sin(2 * math.pi * 2 * t)
        pts.append((V1[0] + ux * L * t - uy * s, -h, V1[1] + uy * L * t + ux * s))
    cut.append(xform(tube(pts, 0.38), Mf))
    line(V2, (0.8, 27.2))
    arrow((1.75, 24.6), math.degrees(math.atan2(27.2 - 21.0, 0.8 - 3.0)))
    line(V2, (6.3, 26.2))
    arrow((5.0, 24.15), math.degrees(math.atan2(21.0 - 26.2, 3.0 - 6.3)))  # antiparticle: arrow back
    for V in (V1, V2):
        cut.append(xform(sphere(1.0, (V[0], -h, V[1]), 20), Mf))
    for s, u, v, size in (("n", -5.4, 11.0, 3.0), ("p", -5.4, 25.0, 3.0), ("W", 1.9, 17.2, 2.8),
                          ("e", -0.6, 28.9, 3.0), ("\u03bd", 6.6, 28.4, 3.0), ("WEAK", 0, 32.0, 3.0)):
        cut.append(face_text(s, size, u, v, h, 0.6, turn))
    return cut


def feynman_strong(h, turn):
    """Two quarks exchanging a gluon (curly line), engraved on the face at -y (turned)."""
    d = 0.4
    cut = []
    Mf = Matrix.Rotation(rad(turn), 4, "Z")
    for u in (-4.6, 4.6):
        cut.append(xform(rounded_tube([(u, -h, 9.6), (u, -h, 27.0)], d), Mf))
        for v in (13.2, 23.6):
            tri = [(u, v + 0.95), (u - 1.05, v - 0.95), (u + 1.05, v - 0.95)]
            cut.append(face_prism(tri, h, 0.6, turn))
    # the loops cross themselves, so build the curly line from half-loops and union them
    n, L, c, z0 = 4, 9.2, 1.05, 18.4
    halves = []
    for k in range(2 * n):
        pts = []
        for i in range(31):
            t = (k + i / 30) / (2 * n)
            th = 2 * math.pi * n * t
            pts.append((-4.6 + L * t - c * math.sin(th), -h, z0 + c * (1 - math.cos(th))))
        halves.append(tube(pts, 0.36))
        halves.append(sphere(0.36, pts[-1], 12))
    cut.append(xform(union(halves), Mf))
    for u in (-4.6, 4.6):
        cut.append(xform(sphere(1.0, (u, -h, z0), 20), Mf))
    for s, u, v, size in (("q", -6.6, 10.6, 3.0), ("q", 6.6, 10.6, 3.0), ("g", 0, 23.4, 3.0),
                          ("STRONG", 0, 32.0, 2.9)):
        cut.append(face_text(s, size, u, v, h, 0.6, turn))
    return cut


def face_prism(poly_uv, h, depth, turn):
    """Polygon in face coordinates (u right, v up) cut `depth` into the face at y=-h."""
    ob = prism(poly_uv, h - depth, h + 3)
    return xform(ob, Matrix.Rotation(rad(turn), 4, "Z") @ Matrix.Rotation(rad(90), 4, "X"))


def face_text(s, size, u, v, h, depth, turn):
    t = text(s, size, 3 + depth)
    t = xform(t, Matrix.Translation((u, v, h + 1.5 - depth / 2)))
    return xform(t, Matrix.Rotation(rad(turn), 4, "Z") @ Matrix.Rotation(rad(90), 4, "X"))


def helix(a, b, rc, turns, n=160):
    a, b = Vector(a), Vector(b)
    ax = (b - a).normalized()
    n1 = ax.orthogonal().normalized()
    n2 = ax.cross(n1)
    pts = []
    for i in range(n + 1):
        t = i / n
        th = 2 * math.pi * turns * t
        p = a + (b - a) * t + rc * (math.cos(th) * n1 + math.sin(th) * n2)
        pts.append(tuple(p))
    return pts


def rook():
    s, h = 17.0, 8.5
    prof = chain(foot(15), [(12.9, 4.9), (12.9, 6.2)], arcp(12.5, 6.2, 0.4, 0, 90, 4), [(0, 6.6)])
    parts = [lathe("rook", prof)]
    parts.append(prism(rounded_square(s, 1.4), 6.3, 34.6))
    parts.append(loft(rounded_square(s, 1.4), 34.6, rounded_square(19.2, 1.6), 35.7))
    wall = prism(rounded_square(19.2, 1.6), 35.7, 43.2)
    hollow = prism(rounded_square(14.0, 0.9), 39.6, 44.0)
    cren = [box(-2.3, 2.3, -11, 11, 40.4, 44.0), box(-11, 11, -2.3, 2.3, 40.4, 44.0)]
    boolean(wall, [hollow] + cren, "DIFFERENCE")
    parts.append(wall)
    # proton: u u d quarks with gluon springs, on a round dais
    dais_top = 41.2
    parts.append(lathe("dais", [(0, 39.0), (6.4, 39.0), (6.4, dais_top - 0.4)] + arcp(6.0, dais_top - 0.4, 0.4, 0, 90, 4)
                       + [(0, dais_top)]))
    rq, side = 2.35, 9.0
    qz = dais_top + 0.6  # the bottom spring is sunk into the dais (a bar hanging over it would start in mid-air)
    Q = [Vector((-side / 2, 0, qz)), Vector((side / 2, 0, qz)), Vector((0, 0, qz + side * math.sqrt(3) / 2))]
    for q in Q:
        parts.append(sphere(rq, q, 40))
    for a, b in ((Q[0], Q[1]), (Q[0], Q[2]), (Q[1], Q[2])):
        u = (b - a).normalized()
        a1, b1 = a + u * (rq - 0.4), b - u * (rq - 0.4)
        # the coil overlaps its rod, so the spring is one solid bar (a free coil would start in mid-air)
        # the coil sits mostly inside the rod (sticks out 0.45 mm), so its low points never float
        parts.append(tube([tuple(a1), tuple(b1)], 1.15))
        parts.append(tube(helix(a1, b1, 1.05, 3.0), 0.55))
    body = union(parts)
    # engravings
    cut = feynman_weak(h, 0) + feynman_weak(h, 180) + feynman_strong(h, 90) + feynman_strong(h, -90)
    for q, letter in zip(Q, ("u", "d", "u")):
        t = front_text(letter, 3.0, q.z + 0.6, x=q.x, reach=rq + 3)  # above the dais top
        t = xform(t, Matrix.Translation((0, 0, 0)))
        cut.append(shell_cutter(t, sphere(rq - 0.5, q, 40)))
    carve(body, cut)
    return finish(body, "rook")


# knight outline in (x, z) above the base top; nose towards -x, mane towards +x
KNIGHT_OUTLINE = [
    (9.6, -4.0), (-8.6, -4.0), (-9.6, 4.0), (-9.8, 8.0), (-9.0, 12.0), (-7.5, 15.5),
    (-12.0, 20.0), (-16.0, 24.0), (-17.2, 25.6), (-17.8, 28.0), (-17.2, 30.2), (-15.4, 31.8),
    (-12.6, 34.0), (-9.0, 37.0), (-5.8, 40.4), (-3.8, 43.2), (-3.0, 46.4), (-1.3, 50.2),
    (0.5, 47.0), (2.4, 45.4), (5.4, 43.2), (8.4, 39.8), (10.6, 35.0), (11.9, 28.0),
    (12.2, 20.0), (11.6, 13.0), (10.6, 6.0), (10.0, 2.4),
]
KZ = 7.8          # base top
KHALF = 6.0       # half thickness of the neck


def knight_f(x, zl):
    return 1 - 0.36 * smooth01((-8.0 - x) / 9.0) - 0.14 * smooth01((zl - 30.0) / 14.0)


def _gauss(x, z, cx, cz, s):
    return math.exp(-((x - cx) ** 2 + (z - cz) ** 2) / (2 * s * s))


def knight_H(x, z):
    """Half thickness of the knight away from its edges: tapered head, cheeks, brow, nostrils."""
    zl = z - KZ
    h = KHALF * knight_f(x, zl)
    h += 0.95 * _gauss(x, zl, -7.0, 33.0, 4.0)     # cheek
    h += 0.55 * _gauss(x, zl, -5.0, 40.6, 1.7)     # brow over the eye
    h += 0.60 * _gauss(x, zl, -16.0, 29.6, 1.9)    # nostril flare
    h -= 0.45 * _gauss(x, zl, -12.2, 25.6, 2.0)    # hollow under the cheek
    return h


def _seg_dist(px, pz, ax, az, bx, bz):
    dx, dz = bx - ax, bz - az
    t = max(0.0, min(1.0, ((px - ax) * dx + (pz - az) * dz) / (dx * dx + dz * dz)))
    return math.hypot(px - ax - t * dx, pz - az - t * dz)


def _inside(poly, x, z):
    c = False
    for (x1, z1), (x2, z2) in zip(poly, poly[1:] + poly[:1]):
        if (z1 > z) != (z2 > z) and x < x1 + (z - z1) * (x2 - x1) / (z2 - z1):
            c = not c
    return c


def pillow(outline, H, b=3.4, step=0.38, name="pillow"):
    """Solid whose half thickness is H(x, z) inside the outline, rounding to zero at the edge over `b` mm."""
    from mathutils.geometry import delaunay_2d_cdt
    bnd = []
    for (x1, z1), (x2, z2) in zip(outline, outline[1:] + outline[:1]):
        n = max(1, int(math.dist((x1, z1), (x2, z2)) / step))
        bnd += [(x1 + (x2 - x1) * i / n, z1 + (z2 - z1) * i / n) for i in range(n)]
    xs = [p[0] for p in bnd]
    zs = [p[1] for p in bnd]
    segs = list(zip(bnd, bnd[1:] + bnd[:1]))
    inner = []
    x = min(xs)
    while x < max(xs):
        z = min(zs)
        while z < max(zs):
            if _inside(bnd, x, z) and min(_seg_dist(x, z, a[0], a[1], c[0], c[1]) for a, c in segs) > step * 0.6:
                inner.append((x, z))
            z += step * 0.866
        x += step
        z0 = 0
    nb = len(bnd)
    pts = [Vector(p) for p in bnd + inner]
    vs, _, fs, orig, _, _ = delaunay_2d_cdt(pts, [(i, (i + 1) % nb) for i in range(nb)], [list(range(nb))], 1, 1e-6)
    out_of = {}
    for o, ins in enumerate(orig):
        for i in ins:
            out_of[i] = o

    def thick(px, pz):
        d = min(_seg_dist(px, pz, a[0], a[1], c[0], c[1]) for a, c in segs)
        s = min(d / b, 1.0)
        return max(0.06, H(px, pz) * math.sqrt(max(0.0, 1 - (1 - s) ** 2)))

    T = [thick(v.x, v.y) for v in vs]
    n = len(vs)
    verts = [(v.x, T[i], v.y) for i, v in enumerate(vs)] + [(v.x, -T[i], v.y) for i, v in enumerate(vs)]
    faces = [tuple(f) for f in fs] + [tuple(n + i for i in reversed(f)) for f in fs]
    for i in range(nb):  # thin rim wall joining the two halves
        a, c = out_of[i], out_of[(i + 1) % nb]
        faces.append((a, c, n + c, n + a))
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    ob = new_obj(name, me)
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    return ob


def surf_y(ob, x, z, sgn):
    """y of the knight's side surface at (x, z), on the +y (sgn=1) or -y side."""
    ok, loc, _, _ = ob.ray_cast(Vector((x, sgn * 30.0, z)), Vector((0, -sgn, 0)))
    return loc.y if ok else sgn * KHALF


def knight():
    prof = chain(base_prof(15, 11.8, 6.8), [(11.8, 7.3)], arcp(11.4, 7.3, 0.4, 0, 90, 4), [(0, 7.7)])
    parts = [lathe("knight_base", prof)]
    outline = chaikin([(x, z + KZ) for x, z in KNIGHT_OUTLINE], 2)
    head = pillow(outline, knight_H, b=3.4, name="knight_head")
    parts.append(head)
    probe = head
    # mane: beads along the back
    mane = [(1.4, 46.6), (3.6, 44.9), (6.2, 42.7), (8.7, 39.6), (10.8, 35.0), (12.0, 29.6), (12.5, 24.0)]
    path = []
    for (x0, z0), (x1, z1) in zip(mane, mane[1:]):
        for i in range(10):
            t = i / 10
            path.append((x0 + (x1 - x0) * t, z0 + (z1 - z0) * t))
    path.append(mane[-1])
    acc, last = 0.0, None
    for p in path:
        if last is not None:
            acc += math.dist(p, last)
        if last is None or acc >= 2.9:
            parts.append(sphere(1.85, (p[0] - 0.5, 0, p[1] + KZ + 0.3), 24))  # inset so no bead hangs off the thin rim
            acc = 0.0
        last = p
    # barrier + wave packet on both sides
    # the barrier, and a wave packet that tunnels through it, on both sides of the neck
    xb0, xb1, zw = 1.4, 4.6, 22.0 + KZ
    A, lam, kap = 2.0, 6.8, 0.30
    Bout = A * math.exp(-kap * (xb1 - xb0))
    cut = []
    for sgn in (-1, 1):
        s_mid = abs(surf_y(probe, (xb0 + xb1) / 2, zw, sgn))
        yin, yout = s_mid - 1.2, s_mid + 1.35
        z0b, z1b = 12.5 + KZ, 33.0 + KZ
        sec = [(yin, z0b), (yout, z0b + (yout - yin)), (yout, z1b), (yin, z1b)]
        bar = prism([(sgn * y, z) for y, z in sec], xb0, xb1, "barrier")
        xform(bar, Matrix(((0, 0, 1, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 1))))
        parts.append(clean(bar))

        def wave(x0, x1, f, n):
            pts = []
            for i in range(n + 1):
                x = x0 + (x1 - x0) * i / n
                z = f(x)
                pts.append((x, surf_y(probe, x, z, sgn) - sgn * 0.1, z))
            return pts

        inc = wave(-9.4, xb0 + 0.3, lambda x: zw + A * math.cos(2 * math.pi * (x - xb0) / lam), 110)
        parts.append(union([tube(inc, 0.56), sphere(0.56, inc[0], 16)]))
        out = wave(xb1 - 0.3, 11.2, lambda x: zw + Bout * math.cos(2 * math.pi * (x - xb1) / lam), 80)
        parts.append(union([tube(out, 0.56), sphere(0.56, out[-1], 16)]))
        dec = [(x, sgn * yout, zw + A * math.exp(-kap * (x - xb0))) for x in
               [xb0 - 0.4 + (xb1 - xb0 + 0.8) * i / 30 for i in range(31)]]
        cut.append(tube(dec, 0.45))
        # psi eye
        ex, ez = -5.4, 37.4 + KZ
        ey = abs(surf_y(probe, ex, ez, sgn))
        t = text("\u03c8", 8.0, 6.0)
        xform(t, Matrix.Translation((0, 0, ey)))
        xform(t, Matrix.Rotation(rad(90), 4, "X"))
        xform(t, Matrix.Translation((ex, 0, ez)))
        if sgn > 0:
            xform(t, Matrix.Diagonal((1, -1, 1, 1)))
            clean(t)
        inner = pillow(outline, lambda x, z: knight_H(x, z) - 0.75, b=3.4, step=0.5, name="eye_inner")
        cut.append(shell_cutter(t, inner))
        # nostril
        nx, nz = -15.6, 30.0 + KZ
        cut.append(ellipsoid(1.2, 1.0, 1.6, (nx, surf_y(probe, nx, nz, sgn), nz)))
    # open mouth, a gap between the ears
    mouth = prism([(-21.0, 27.0 + KZ), (-15.2, 26.3 + KZ), (-21.0, 25.4 + KZ)], -10, 10, "mouth")
    cut.append(clean(xform(mouth, Matrix(((1, 0, 0, 0), (0, 0, 1, 0), (0, 1, 0, 0), (0, 0, 0, 1))))))
    cut.append(box(-4.5, 3.0, -1.0, 1.0, 46.3 + KZ, 60.0))
    body = union(parts)
    carve(body, cut)
    return finish(body, "knight")


def bishop():
    prof = chain(base_prof(15, 5.8, 9.5), bez((5.8, 9.5), (5.0, 13), (4.6, 17), (4.6, 20.5)),
                 [(8.2, 24.5), (8.4, 24.9), (8.4, 25.6), (8.2, 25.8)],   # past light cone rim
                 [(2.8, 31.2), (2.8, 31.8)],                             # event (waist)
                 [(9.2, 38.2), (9.5, 38.6), (9.5, 39.4)],               # future light cone + rim bead
                 bez((9.5, 39.4), (10.0, 44.0), (7.5, 52.0), (1.8, 57.0)),
                 [(1.6, 57.6)], arcp(0, 59.9, 2.6, -52, 90, 14))
    # the event sphere is the waist: thick enough (min r ~4 mm) that the piece does not snap there
    body = union([lathe("bishop", prof), sphere(4.2, (0, 0, 31.5), 48)])
    cut = []
    # light-cone generators: grooves running at 45 degrees on both cones
    for k in range(16):
        a = 2 * math.pi * (k + 0.5) / 16
        ca, sa = math.cos(a), math.sin(a)
        cut.append(rounded_tube([(8.0 * ca, 8.0 * sa, 26.0), (4.6 * ca, 4.6 * sa, 29.4)], 0.4))
        cut.append(rounded_tube([(4.6 * ca, 4.6 * sa, 33.6), (8.9 * ca, 8.9 * sa, 37.9)], 0.4))
    # the slit: a light ray at 45 degrees, cut into the front half
    slab = box(-11.0, 15.0, -0.85, 0.85, -14.0, 0.9)  # u along x, thickness along y, depth along z
    M = Matrix.Translation((0, 0, 47.5)) @ Matrix.Rotation(rad(-45), 4, "Y") @ Matrix.Rotation(rad(-90), 4, "X")
    cut.append(xform(slab, M))
    # "c" on the stem, front and back
    inner_prof = [(max(r - 0.7, 0.05) if r > 0 else 0, z) for r, z in prof]
    for turn in (0, 180):
        t = front_text("c", 9.5, 14.2, reach=12, turn=turn)
        cut.append(shell_cutter(t, lathe("inner", inner_prof)))
    carve(body, cut)
    return finish(body, "bishop")


def queen():
    zf0, zf1 = 12.0, 52.0
    prof_col = chain(base_prof(16.5, 6.2, 10), bez((6.2, 10), (5.4, 16), (4.6, 30), (4.4, 50)))
    prof = chain(prof_col, [(7.6, 53.2)], [(7.8, 53.6), (7.8, 57.0)], arcp(7.3, 57.0, 0.5, 0, 90, 4), [(0, 57.5)])
    parts = [lathe("queen", prof)]
    A, half = 6.2, 0.8
    rimpts, hatch = [], []
    for plane in (0, 90):  # E wave in x-z, B wave in y-z, in phase
        Mp = Matrix.Rotation(rad(plane), 4, "Z")
        for sgn, za in ((1, zf0), (-1, (zf0 + zf1) / 2)):
            zb = za + (zf1 - zf0) / 2
            edge = []
            for i in range(41):
                z = za + (zb - za) * i / 40
                edge.append((sgn * (radius_at(prof_col, z) + A * math.sin(math.pi * (z - za) / (zb - za))), z))
            fin = slab2d([(0.0, za)] + edge + [(0.0, zb)] if sgn > 0 else [(0.0, zb)] + edge[::-1] + [(0.0, za)],
                         half, 0.3, "fin")
            parts.append(xform(fin, Mp))
            # the rim starts and ends 1.2 mm inside the column, so its round ends are buried
            ends = [(edge[0][0] - sgn * 1.2, edge[0][1])], [(edge[-1][0] - sgn * 1.2, edge[-1][1])]
            rim = [Mp @ Vector((x, 0, z)) for x, z in ends[0] + edge + ends[1]]
            parts.append(tube([tuple(p) for p in rim], 0.85))
            rimpts.append(rim)
            # field vectors: grooves across the fin from the column to the wave
            z = za + 2.4
            while z < zb - 1.6:
                xe = radius_at(prof_col, z) + A * math.sin(math.pi * (z - za) / (zb - za))
                rc = radius_at(prof_col, z)
                if xe - rc > 1.8:
                    for side in (-1, 1):
                        g = tube([(sgn * (rc - 0.3), side * half, z), (sgn * (xe - 1.0), side * half, z)], 0.3)
                        hatch.append(xform(g, Mp))
                z += 2.5
    # crown of magnetic field lines from the N pole down into the collar (S)
    zc0, zball, rball = 57.2, 70.0, 3.7
    # the ribs run on into the ball so their ends are buried (a poking end makes a thin neck)
    rib2d = [(5.4, 56.0)] + bez((7.0, zc0), (8.9, 58.9), (9.4, 63.2), (3.0, 68.6), 40) + [(2.1, 69.2), (1.3, 69.7)]
    parts.append(lathe("core", [(0, 56.5), (2.4, 56.5), (2.4, zball)] + [(0, zball)]))
    parts.append(sphere(rball, (0, 0, zball), 48))
    ribs = 8
    for k in range(ribs):
        a = 2 * math.pi * (k + 0.5) / ribs
        ca, sa = math.cos(a), math.sin(a)
        parts.append(tube([(r * ca, r * sa, z) for r, z in rib2d], 1.75))
    body = union(parts)
    cut = hatch
    inner_prof = [(max(r - 0.6, 0.05) if r > 0 else 0, z) for r, z in prof]
    for s_, turn in (("B", 0), ("E", -90), ("B", 180), ("E", 90)):
        cut.append(shell_cutter(front_text(s_, 3.6, 55.3, reach=12, turn=turn), lathe("inner", inner_prof)))
    cut.append(shell_cutter(front_text("N", 3.8, zball + 0.2, reach=rball + 3, turn=-35),
                            sphere(rball - 0.6, (0, 0, zball), 48)))
    cut.append(torus(2.4, 0.35, Matrix.Translation((0, 0, 63.4)), 64, 12))
    carve(body, cut)
    return finish(body, "queen")


def king():
    # the body is a gravity well: a funnel that flares (at most 45 degrees) with a spacetime grid on it
    zf0, zf1, rf0, rf1 = 24.0, 46.0, 4.8, 14.6
    kf = (rf1 - rf0) / (zf1 - zf0) ** 2

    def rflare(z):
        return rf0 + kf * (z - zf0) ** 2

    r_rim, r_in, z_rim, D = 14.3, 4.6, 47.1, 3.7

    def zw(r):
        return z_rim - D * (r_rim / r - 1) / (r_rim / r_in - 1)

    flare = [(rflare(zf0 + (zf1 - zf0) * i / 40), zf0 + (zf1 - zf0) * i / 40) for i in range(41)]
    well = [(r_rim - (r_rim - r_in) * i / 40, zw(r_rim - (r_rim - r_in) * i / 40)) for i in range(41)]
    prof = chain(base_prof(17.5, 6.4, 10.5), bez((6.4, 10.5), (5.4, 16), (4.8, 20.5), (4.8, zf0)), flare,
                 arcp(14.3, 46.5, 0.6, -56, 90, 8), well, [(0, zw(r_in))])
    parts = [lathe("king", prof)]
    gr = 0.45
    for k in range(16):  # meridians on the outside of the funnel and across the well
        a = 2 * math.pi * (k + 0.5) / 16
        ca, sa = math.cos(a), math.sin(a)
        pts = [(rflare(z) * ca, rflare(z) * sa, z) for z in [25.5 + (45.7 - 25.5) * i / 40 for i in range(41)]]
        parts.append(rounded_tube(pts, gr))
        pts = [(r * ca, r * sa, zw(r)) for r in [13.7 - (13.7 - 9.9) * i / 12 for i in range(13)]]
        parts.append(rounded_tube(pts, gr))
    for z in (29.0, 34.0, 38.6, 42.6):
        r = rflare(z)
        parts.append(tube([(r * math.cos(2 * math.pi * i / 180), r * math.sin(2 * math.pi * i / 180), z)
                           for i in range(180)], gr, cyclic=True))
    r = 12.0
    parts.append(tube([(r * math.cos(2 * math.pi * i / 160), r * math.sin(2 * math.pi * i / 160), zw(r))
                       for i in range(160)], gr, cyclic=True))
    # a ringed planet sitting in the well; the ring has a 45-degree underside so it prints
    rp = 7.0
    zp = zw(r_in) + rp - 1.3
    parts.append(sphere(rp, (0, 0, zp), 72))
    zr = zp + 0.35
    parts.append(lathe("ring", [(0, zr - 5.4), (4.5, zr - 5.4), (9.4, zr - 0.5), (9.4, zr - 0.15)]
                       + arcp(9.1, zr - 0.15, 0.3, 0, 90, 4) + [(0, zr + 0.15)]))
    # cross pattee: a thick stem sunk into the planet, with a 45-degree collar, so it does not snap off
    w, t = 3.6, 3.6
    zb, za0, za1, zt = zp + 3.5, 73.0, 76.6, 84.0
    cross = [(-w, zb), (w, zb), (w, za0 - 4.2), (7.2, za0), (7.2, za1), (w, za1), (w + 0.5, zt), (-w - 0.5, zt),
             (-w, za1), (-7.2, za1), (-7.2, za0), (-w, za0 - 4.2)]
    parts.append(slab2d(cross, t, 0.6, "cross"))
    zcol = zp + 4.5
    parts.append(lathe("cross_collar", [(0, zcol), (5.2, zcol), (3.0, zcol + 2.2), (0, zcol + 2.2)]))
    body = union(parts)
    cut = [torus(8.2, 0.28, Matrix.Translation((0, 0, zr + 0.15)), 128, 10)]
    for lat in (25, 47):
        cut.append(torus(rp * math.cos(rad(lat)), 0.32, Matrix.Translation((0, 0, zp + rp * math.sin(rad(lat)))), 128, 10))
    carve(body, cut)
    return finish(body, "king")


PIECES = [("pawn", pawn, "atom", 8), ("rook", rook, "nuclear_forces", 2), ("knight", knight, "tunnelling", 2),
          ("bishop", bishop, "speed_of_light", 2), ("queen", queen, "electromagnetism", 1), ("king", king, "gravity", 1)]


# ----------------------------------------------------------------------------- export

def clear_all():
    for ob in list(bpy.data.objects):
        delete(ob)
    for c in list(bpy.data.collections):
        bpy.data.collections.remove(c)
    for me in list(bpy.data.meshes):
        if me.users == 0:
            bpy.data.meshes.remove(me)


def export_stl(objs, path):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.wm.stl_export(filepath=path, export_selected_objects=True, global_scale=1.0, apply_modifiers=True,
                          ascii_format=False)
    bpy.ops.object.select_all(action="DESELECT")


def export_all():
    os.makedirs(STL_DIR, exist_ok=True)
    plate = []
    for kind, _, name, count in PIECES:
        ob = bpy.data.objects[kind]
        export_stl([ob], os.path.join(STL_DIR, f"{kind}_{name}.stl"))
    # print plate: 16 pieces on a 4 x 4 grid, 38 mm apart
    order = ["king", "queen", "rook", "rook", "bishop", "bishop", "knight", "knight"] + ["pawn"] * 8
    for i, kind in enumerate(order):
        src = bpy.data.objects[kind]
        cp = new_obj(f"plate_{i}", src.data, "plate")
        cp.location = ((i % 4 - 1.5) * 38.0, (i // 4 - 1.5) * 38.0, 0)
        plate.append(cp)
    export_stl(plate, os.path.join(STL_DIR, "plate_16_pieces.stl"))
    delete(*plate)


# ----------------------------------------------------------------------------- rendering

COLOURS = {"white": ((0.80, 0.69, 0.50), 0.36), "black": ((0.030, 0.030, 0.036), 0.30)}
VIEW = {"pawn": -30, "rook": -35, "knight": -12, "bishop": -25, "queen": -40, "king": -22}
TITLES = {"pawn": "the atom", "rook": "strong + weak force", "knight": "quantum tunnelling",
          "bishop": "speed of light", "queen": "electromagnetic force", "king": "gravity"}


def material(name, rgb, rough):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (*rgb, 1)
    bsdf.inputs["Roughness"].default_value = rough
    return m


def look_at(ob, target):
    d = Vector(target) - ob.location
    ob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def studio():
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for t in ("OPTIX", "CUDA", "HIP", "ONEAPI"):
            try:
                prefs.compute_device_type = t
            except TypeError:
                continue
            prefs.get_devices()
            devs = [d for d in prefs.devices if d.type == t]
            if devs:
                for d in devs:
                    d.use = True
                sc.cycles.device = "GPU"
                break
    except Exception as e:  # noqa: BLE001
        print("no GPU:", e)
    sc.cycles.samples = 160
    sc.cycles.use_denoising = True
    sc.render.resolution_x, sc.render.resolution_y = 1000, 1250
    sc.render.image_settings.file_format = "PNG"
    world = sc.world or bpy.data.worlds.new("World")
    sc.world = world
    world.use_nodes = True
    bg = next(n for n in world.node_tree.nodes if n.type == "BACKGROUND")
    bg.inputs[0].default_value = (0.05, 0.055, 0.065, 1)
    bg.inputs[1].default_value = 1.0
    st = coll("studio")
    for o in list(st.objects):
        delete(o)
    # cyclorama: floor that curves up into a back wall
    prof = [(y, 0.0) for y in (-900, -200, 0, 60)] + [(60 + 160 * math.sin(rad(a)), 160 - 160 * math.cos(rad(a)))
                                                       for a in range(6, 91, 6)] + [(220, 900)]
    verts, faces = [], []
    for x in (-900, 900):
        for y, z in prof:
            verts.append((x, y, z))
    n = len(prof)
    for i in range(n - 1):
        faces.append((i, i + 1, n + i + 1, n + i))
    me = bpy.data.meshes.new("cyc")
    me.from_pydata(verts, [], faces)
    me.shade_smooth()
    cyc = new_obj("cyclorama", me, "studio")
    cyc.data.materials.append(material("backdrop", (0.11, 0.12, 0.14), 0.6))
    for name, d, e, ang in (("key", (-0.55, -0.75, 0.85), 3.4, 7), ("fill", (0.9, -0.5, 0.35), 0.9, 25),
                            ("rim", (0.35, 1.0, 0.75), 5.0, 4), ("top", (0.0, 0.0, 1.0), 0.8, 30)):
        L = bpy.data.lights.new(name, "SUN")
        L.energy = e
        L.angle = rad(ang)
        ob = new_obj(name, L, "studio")
        ob.location = Vector(d) * 300
        look_at(ob, (0, 0, 0))
    cam = bpy.data.cameras.new("cam")
    cam.lens = 100
    camo = new_obj("camera", cam, "studio")
    sc.camera = camo
    for k, (rgb, rough) in COLOURS.items():
        material(k, rgb, rough)


def aim(az, el=13.0, dist=300.0, target_z=40.0):
    """Turn camera, lights and backdrop together so the camera sits at azimuth `az` (0 = front, -y)."""
    Rz = Matrix.Rotation(rad(az), 4, "Z")
    st = coll("studio")
    cam = bpy.data.objects["camera"]
    cam.location = Rz @ Vector((0, -dist * math.cos(rad(el)), target_z + dist * math.sin(rad(el))))
    look_at(cam, (0, 0, target_z))
    cyc = bpy.data.objects["cyclorama"]
    cyc.rotation_euler = (0, 0, rad(az))
    base = {"key": (-0.55, -0.75, 0.85), "fill": (0.9, -0.5, 0.35), "rim": (0.35, 1.0, 0.75), "top": (0, 0, 1.0)}
    for name, d in base.items():
        ob = st.objects[name]
        ob.location = Rz @ (Vector(d) * 300)
        look_at(ob, (0, 0, 0))


def render_piece(kind, colour, path, az=None):
    for o in coll("pieces").objects:
        o.hide_render = o.name != kind
    ob = bpy.data.objects[kind]
    ob.data.materials.clear()
    ob.data.materials.append(bpy.data.materials[colour])
    aim(VIEW[kind] if az is None else az)
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


def build(kinds=None):
    made = []
    for kind, fn, _, _ in PIECES:
        if kinds and kind not in kinds:
            continue
        old = bpy.data.objects.get(kind)
        if old is not None:
            delete(old)
        made.append(fn())
        leftover = [o for o in coll("build").objects]
        if leftover:
            delete(*leftover)
    return made


if __name__ == "__main__":
    clear_all()
    build()
    export_all()
