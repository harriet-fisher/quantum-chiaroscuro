// Shared-edge geometry of the pen tool.   node tests/pen_geom.test.js
const assert = require('assert');
const G = require('../src/capture/pen_geom.js');
let n = 0;
const test = (name, fn) => { fn(); n++; console.log('ok  ' + name); };
const clone = p => p.map(q => q.map(v => v.slice()));

// two panels side by side sharing the vertical edge x=100 (A has it as 100,0-100,100; B is taller on that side: a T-junction at (100,50))
const A = () => [[0, 0], [100, 0], [100, 100], [0, 100]];
const B = () => [[100, 0], [200, 0], [200, 100], [100, 100]];

test('nearest point on a segment, clamped to its ends', () => {
  assert.deepStrictEqual(G.nearestOnSegment([5, 5], [0, 0], [10, 0]).q, [5, 0]);
  assert.deepStrictEqual(G.nearestOnSegment([-5, 3], [0, 0], [10, 0]).q, [0, 0]);
  assert.strictEqual(G.nearestOnSegment([5, 5], [0, 0], [0, 0]).d, Math.hypot(5, 5));       // zero-length segment
});

test('snap prefers a vertex over an edge, edges over nothing, and nothing outside the radii', () => {
  const polys = [A()];
  let s = G.snap([98, 3], polys, 8, 5);
  assert.strictEqual(s.kind, 'vertex'); assert.deepStrictEqual(s.pt, [100, 0]);
  s = G.snap([103, 40], polys, 8, 5);
  assert.strictEqual(s.kind, 'edge'); assert.deepStrictEqual(s.pt, [100, 40]);
  s = G.snap([150, 40], polys, 8, 5);
  assert.strictEqual(s.kind, null); assert.deepStrictEqual(s.pt, [150, 40]);
});

test('an edge snap right next to an endpoint becomes a vertex snap, never a near-duplicate vertex', () => {
  const s = G.snap([100.1, 0.2], [A()], 0.1, 5);        // outside the (tiny) vertex radius, but the edge point (100, 0.2) is within EPS of the corner
  assert.strictEqual(s.kind, null);
});

test('weld inserts a vertex where another shape has one on this edge (T-junction)', () => {
  const polys = [A(), [[100, 0], [200, 0], [200, 50], [100, 50]], [[100, 50], [200, 50], [200, 100], [100, 100]]];
  assert.strictEqual(polys[0].length, 4);
  const added = G.weld(polys);
  assert.strictEqual(added, 1);
  assert.deepStrictEqual(polys[0], [[0, 0], [100, 0], [100, 50], [100, 100], [0, 100]]);   // the vertex went into the right edge, in order
  assert.strictEqual(G.weld(polys), 0);                                                      // idempotent
});

test('weld handles several vertices on one edge and leaves unrelated shapes alone', () => {
  const polys = [[[0, 0], [100, 0], [100, 90], [0, 90]], [[100, 10], [150, 10], [150, 30], [100, 30]], [[100, 60], [150, 60], [150, 80], [100, 80]], [[300, 300], [310, 300], [305, 310]]];
  G.weld(polys);
  assert.deepStrictEqual(polys[0].map(p => p.join(',')), ['0,0', '100,0', '100,10', '100,30', '100,60', '100,80', '100,90', '0,90']);
  assert.strictEqual(polys[3].length, 3);
});

test('a vertex drawn exactly on a neighbour edge gets welded both ways', () => {
  const polys = [A(), [[100, 20], [160, 20], [160, 70], [100, 70]]];
  G.weld(polys);
  assert.strictEqual(polys[0].length, 6);                                                    // both of B's corners inserted into A's edge
  assert.ok(polys[0].some(p => p[0] === 100 && p[1] === 20) && polys[0].some(p => p[0] === 100 && p[1] === 70));
});

test('coincident vertices form one group and move together', () => {
  const polys = [A(), B()];
  const grp = G.group(polys, [100, 100]);
  assert.deepStrictEqual(grp.sort(), [[0, 2], [1, 3]]);
  assert.strictEqual(G.sharing(polys, [100, 100]), 2);
  assert.strictEqual(G.sharing(polys, [0, 0]), 1);
  G.moveGroup(polys, grp, [100, 120]);
  assert.deepStrictEqual([polys[0][2], polys[1][3]], [[100, 120], [100, 120]]);
  assert.strictEqual(G.group(polys, [100, 120]).length, 2);                                  // still shared after the move
});

test('grab: a vertex takes its whole group, an edge takes both end vertices and their groups', () => {
  const polys = [A(), B()];
  const v = G.grab([99, 1], polys, 6, 4);
  assert.strictEqual(v.kind, 'vertex'); assert.strictEqual(v.members.length, 2);
  const e = G.grab([100, 50], polys, 6, 4);                                                   // middle of the shared edge
  assert.strictEqual(e.kind, 'edge');
  assert.strictEqual(e.members.length, 4);                                                    // A's two + B's two, each pair coincident
  assert.strictEqual(G.grab([50, 50], polys, 6, 4), null);                                    // inside a shape, away from anything
});

test('snap while dragging ignores the vertices being dragged and snaps onto the other shape', () => {
  const polys = [A(), B()];
  const grp = G.group(polys, [100, 100]);
  const inGroup = new Set(grp.map(m => m[0] + ':' + m[1]));
  const s = G.snap([198, 3], polys, 8, 5, (si, vi) => inGroup.has(si + ':' + vi));
  assert.strictEqual(s.kind, 'vertex'); assert.deepStrictEqual(s.pt, [200, 0]);
  const s2 = G.snap([101, 99], polys, 8, 5, (si, vi) => inGroup.has(si + ':' + vi));          // only the dragged vertex itself is nearby: no snap to itself
  assert.ok(!(s2.kind === 'vertex' && s2.pt[0] === 100 && s2.pt[1] === 100));
});

test('dragging a shared edge keeps both shapes glued', () => {
  const polys = [A(), B()];
  const e = G.grab([100, 50], polys, 6, 4);
  const orig = e.members.map(([si, vi]) => polys[si][vi].slice());
  e.members.forEach(([si, vi], i) => { polys[si][vi] = [orig[i][0] + 15, orig[i][1]]; });
  assert.deepStrictEqual([polys[0][1], polys[0][2], polys[1][0], polys[1][3]], [[115, 0], [115, 100], [115, 0], [115, 100]]);
});

console.log(n + ' pen_geom tests passed');
