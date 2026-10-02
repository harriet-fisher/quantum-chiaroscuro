// Geometry for shared edges in the pen tool: snapping, coincident-vertex groups, welding. No DOM, no state: every function takes the
// polygons (arrays of [x, y], live references) and works in image pixels. Used by the pen tool page and tested under node.
//
// The model, like Photoshop's perspective editing: shapes that touch share vertices. Two vertices are "the same" when they are
// coincident (within EPS), so nothing is stored about sharing; moving a vertex moves every vertex coincident with it. To make an
// edge truly shared, both shapes need a vertex wherever the other has one, which weld() guarantees by inserting vertices where one
// shape's vertex lies on the interior of another's edge.
(function (g) {
  const EPS = 0.25;                 // vertices closer than this are the same vertex (snapped points are copied exactly, so 0 in practice)
  const ON_EDGE = 0.35;             // a vertex this close to an edge's interior lies on it

  const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  const round1 = v => Math.round(v * 10) / 10;

  // Nearest point to p on segment a-b: { q: [x, y], d: distance, t: 0..1 along the segment }
  function nearestOnSegment(p, a, b) {
    const dx = b[0] - a[0], dy = b[1] - a[1], len2 = dx * dx + dy * dy;
    const t = len2 === 0 ? 0 : Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2));
    const q = [a[0] + t * dx, a[1] + t * dy];
    return { q, d: dist(p, q), t };
  }

  // Nearest vertex within maxd: { pt, si, vi, d } or null. skip(si, vi) true = ignore that vertex.
  function nearestVertex(p, polys, maxd, skip) {
    let best = null;
    polys.forEach((poly, si) => poly.forEach((v, vi) => {
      if (skip && skip(si, vi)) return;
      const d = dist(p, v);
      if (d <= maxd && (!best || d < best.d)) best = { pt: [v[0], v[1]], si, vi, d };
    }));
    return best;
  }

  // Nearest point on any edge within maxd that is not (nearly) an endpoint: { pt (rounded), si, ei, d } or null. Edge ei joins vertex ei and ei+1.
  function nearestEdge(p, polys, maxd, skip) {
    let best = null;
    polys.forEach((poly, si) => {
      const n = poly.length;
      if (n < 2) return;
      for (let ei = 0; ei < n; ei++) {
        if (skip && skip(si, ei)) continue;
        const a = poly[ei], b = poly[(ei + 1) % n], r = nearestOnSegment(p, a, b);
        if (r.d <= maxd && (!best || r.d < best.d)) best = { pt: [round1(r.q[0]), round1(r.q[1])], si, ei, d: r.d };
      }
    });
    return best;
  }

  // Where a new or dragged vertex should land. Vertices win over edges. Returns { pt, kind: 'vertex' | 'edge' | null, ... }.
  // vr / er are the snap radii in image pixels; skip is passed to both searches.
  function snap(p, polys, vr, er, skip) {
    const v = nearestVertex(p, polys, vr, skip);
    if (v) return { pt: v.pt, kind: 'vertex', si: v.si, vi: v.vi };
    const e = nearestEdge(p, polys, er, skip && ((si, ei) => skip(si, ei) && skip(si, (ei + 1) % polys[si].length)));
    if (e) {
      const a = polys[e.si][e.ei], b = polys[e.si][(e.ei + 1) % polys[e.si].length];
      if (dist(e.pt, a) > EPS && dist(e.pt, b) > EPS) return { pt: e.pt, kind: 'edge', si: e.si, ei: e.ei };
    }
    return { pt: p, kind: null };
  }

  // Every vertex coincident with pt: [[si, vi], ...]
  function group(polys, pt) {
    const out = [];
    polys.forEach((poly, si) => poly.forEach((v, vi) => { if (dist(v, pt) <= EPS) out.push([si, vi]); }));
    return out;
  }

  // Number of shapes that have a vertex at pt (used to draw shared vertices differently)
  function sharing(polys, pt) { return new Set(group(polys, pt).map(m => m[0])).size; }

  function moveGroup(polys, members, to) { members.forEach(([si, vi]) => { polys[si][vi] = [to[0], to[1]]; }); }

  function onInterior(p, a, b) {
    if (dist(p, a) <= EPS || dist(p, b) <= EPS) return false;
    return nearestOnSegment(p, a, b).d <= ON_EDGE;
  }

  // Make touching shapes share their vertices: wherever a vertex of one shape lies on the interior of another's edge, insert a copy of
  // it into that edge. Repeats until nothing changes (an edge can take several). Returns the number of vertices inserted.
  function weld(polys) {
    let inserted = 0, changed = true, guard = 0;
    while (changed && guard++ < 50) {
      changed = false;
      for (let q = 0; q < polys.length && !changed; q++) {
        const Q = polys[q], n = Q.length;
        for (let ei = 0; ei < n && !changed; ei++) {
          const a = Q[ei], b = Q[(ei + 1) % n];
          for (let p = 0; p < polys.length && !changed; p++) {
            if (p === q) continue;
            for (const v of polys[p]) {
              if (onInterior(v, a, b)) { Q.splice(ei + 1, 0, [v[0], v[1]]); inserted++; changed = true; break; }
            }
          }
        }
      }
    }
    return inserted;
  }

  // What a press at p grabs when no polygon is being drawn: a vertex (with everything coincident to it), else an edge (both of its end
  // vertices, with everything coincident to them). Returns { kind: 'vertex' | 'edge', members, pt } or null.
  function grab(p, polys, vr, er) {
    const v = nearestVertex(p, polys, vr);
    if (v) return { kind: 'vertex', members: group(polys, v.pt), pt: v.pt };
    const e = nearestEdge(p, polys, er);
    if (!e) return null;
    const n = polys[e.si].length, a = polys[e.si][e.ei], b = polys[e.si][(e.ei + 1) % n];
    const seen = new Set(), members = [];
    [...group(polys, a), ...group(polys, b)].forEach(m => { const k = m[0] + ':' + m[1]; if (!seen.has(k)) { seen.add(k); members.push(m); } });
    return { kind: 'edge', members, pt: e.pt };
  }

  g.PenGeom = { EPS, round1, nearestOnSegment, nearestVertex, nearestEdge, snap, group, sharing, moveGroup, weld, grab };
  if (typeof module !== 'undefined' && module.exports) module.exports = g.PenGeom;
})(typeof window !== 'undefined' ? window : globalThis);
