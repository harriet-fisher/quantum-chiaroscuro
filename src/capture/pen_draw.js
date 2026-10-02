// Shared by the pen tool page and the projector window, so both draw shapes identically.
// Coordinates are "image" pixels (the photo, or the projector's pixel grid when there is no photo).
(function (g) {
  const hue = id => (id * 67) % 360;
  const MODE_COLOUR = { panel: '#ffd60a', glass: '#6cb0ff', off: '#ff7b7b' };

  function path(ctx, poly, k, close) {
    ctx.beginPath();
    poly.forEach((p, i) => (i ? ctx.lineTo(p[0] * k, p[1] * k) : ctx.moveTo(p[0] * k, p[1] * k)));
    if (close) ctx.closePath();
  }

  // s: {panels, glass, off, cur, mouse, mode, grid:{nx,ny}, cells:[[i,j,panel],...], sel, snap:{pt,kind}|null, hover:[x,y]|null}
  // o: {W, H, k (device px per image px), dpr, fill, grid, cells, labels, border, crosshair, vertices}
  function drawScene(ctx, s, o) {
    const k = o.k, dpr = o.dpr || 1, lw = Math.max(2, 2 * dpr);
    ctx.lineWidth = lw; ctx.lineJoin = 'round';

    if (o.border) {                                   // the edge of the projected frame, to see where it lands on the object
      ctx.setLineDash([12 * dpr, 8 * dpr]); ctx.strokeStyle = 'rgba(255,255,255,0.55)';
      ctx.strokeRect(lw / 2, lw / 2, o.W * k - lw, o.H * k - lw); ctx.setLineDash([]);
    }
    const cw = o.W / s.grid.nx, ch = o.H / s.grid.ny;
    if (o.grid) {
      ctx.strokeStyle = 'rgba(255,255,255,0.22)'; ctx.lineWidth = Math.max(1, dpr);
      for (let i = 1; i < s.grid.nx; i++) { ctx.beginPath(); ctx.moveTo(i * cw * k, 0); ctx.lineTo(i * cw * k, o.H * k); ctx.stroke(); }
      for (let j = 1; j < s.grid.ny; j++) { ctx.beginPath(); ctx.moveTo(0, j * ch * k); ctx.lineTo(o.W * k, j * ch * k); ctx.stroke(); }
      ctx.lineWidth = lw;
    }
    if (o.cells) {                                    // cells that would get a qubit
      ctx.strokeStyle = 'rgba(255,214,10,0.9)';
      s.cells.forEach(c => ctx.strokeRect(c[0] * cw * k + 3 * dpr, c[1] * ch * k + 3 * dpr, cw * k - 6 * dpr, ch * k - 6 * dpr));
    }
    s.panels.forEach((p, i) => {
      path(ctx, p.polygon, k, true);
      if (o.fill) { ctx.fillStyle = `hsla(${hue(p.plane_id)}, 70%, 55%, ${i === s.sel ? 0.45 : 0.28})`; ctx.fill(); }
      ctx.strokeStyle = i === s.sel ? '#ffffff' : `hsl(${hue(p.plane_id)}, 85%, 68%)`; ctx.stroke();
      if (o.labels) {
        const c = p.polygon.reduce((a, q) => [a[0] + q[0] / p.polygon.length, a[1] + q[1] / p.polygon.length], [0, 0]);
        ctx.fillStyle = '#fff'; ctx.font = `${13 * dpr}px system-ui`; ctx.textAlign = 'center';
        ctx.fillText(`${p.name}  plane ${p.plane_id}  ${p.angle_deg}°`, c[0] * k, c[1] * k);
      }
    });
    [[s.glass, 'rgba(70,150,255,0.45)', '#6cb0ff'], [s.off, 'rgba(255,70,70,0.40)', '#ff7b7b']].forEach(([arr, fill, stroke]) =>
      arr.forEach(poly => {
        path(ctx, poly, k, true);
        if (o.fill) { ctx.fillStyle = fill; ctx.fill(); }
        ctx.setLineDash([8 * dpr, 5 * dpr]); ctx.strokeStyle = stroke; ctx.stroke(); ctx.setLineDash([]);
      }));

    if (o.vertices) {                                 // every vertex; a vertex shared by two or more shapes is filled yellow (it moves them together)
      const polys = [...s.panels.map(p => p.polygon), ...s.glass, ...s.off], count = new Map();
      polys.forEach((poly, si) => poly.forEach(p => { const key = p[0].toFixed(1) + ',' + p[1].toFixed(1); (count.get(key) || count.set(key, new Set()).get(key)).add(si); }));
      const r = 4 * dpr;
      polys.forEach(poly => poly.forEach(p => {
        const shared = count.get(p[0].toFixed(1) + ',' + p[1].toFixed(1)).size > 1, x = p[0] * k, y = p[1] * k;
        ctx.lineWidth = Math.max(1, dpr); ctx.strokeStyle = shared ? '#ffd60a' : 'rgba(255,255,255,.85)'; ctx.fillStyle = shared ? '#ffd60a' : 'rgba(0,0,0,.55)';
        ctx.beginPath(); ctx.rect(x - r, y - r, 2 * r, 2 * r); ctx.fill(); ctx.stroke();
      }));
      if (s.hover) { ctx.lineWidth = Math.max(2, 2 * dpr); ctx.strokeStyle = '#fff'; ctx.beginPath(); ctx.arc(s.hover[0] * k, s.hover[1] * k, 9 * dpr, 0, 7); ctx.stroke(); }
    }
    if (s.snap && s.snap.kind) {                      // where the next vertex will land: a ring on a vertex, a diamond on an edge
      const x = s.snap.pt[0] * k, y = s.snap.pt[1] * k, r = 10 * dpr;
      ctx.lineWidth = Math.max(2, 2 * dpr); ctx.strokeStyle = '#ffd60a'; ctx.beginPath();
      if (s.snap.kind === 'vertex') ctx.arc(x, y, r, 0, 7); else { ctx.moveTo(x, y - r); ctx.lineTo(x + r, y); ctx.lineTo(x, y + r); ctx.lineTo(x - r, y); ctx.closePath(); }
      ctx.stroke();
    }

    const colour = MODE_COLOUR[s.mode] || '#fff';
    if (s.cur.length) {                               // polygon in progress, with a rubber band to the cursor
      path(ctx, s.mouse ? s.cur.concat([s.mouse]) : s.cur, k, false); ctx.strokeStyle = colour; ctx.stroke();
      s.cur.forEach((p, i) => { ctx.beginPath(); ctx.arc(p[0] * k, p[1] * k, (i === 0 ? 7 : 4) * dpr, 0, 7); ctx.fillStyle = i === 0 ? colour : '#fff'; ctx.fill(); });
    }
    if (o.crosshair && s.mouse) {                     // where the next vertex will land
      const x = s.mouse[0] * k, y = s.mouse[1] * k, r = 14 * dpr;
      ctx.strokeStyle = colour; ctx.lineWidth = Math.max(1, dpr);
      ctx.beginPath(); ctx.moveTo(x - r, y); ctx.lineTo(x + r, y); ctx.moveTo(x, y - r); ctx.lineTo(x, y + r); ctx.stroke();
      ctx.beginPath(); ctx.arc(x, y, 5 * dpr, 0, 7); ctx.stroke();
    }
  }
  g.PenDraw = { drawScene, hue, MODE_COLOUR };
})(window);
