"use strict";

function spacing(axis, index) {
  const a = Math.max(0, Math.min(axis.length - 2, Math.floor(index)));
  return axis[a + 1] - axis[a];
}

function directionAt(fields, i, j) {
  const jx = interpolate(fields, "Jx", i, j);
  const jy = interpolate(fields, "Jy", i, j);
  const di = jx / spacing(fields.arrays.x, i);
  const dj = jy / spacing(fields.arrays.y, j);
  const size = Math.hypot(di, dj);
  if (!(size > 0) || !isFinite(size)) return null;
  return { di: di / size, dj: dj / size, magnitude: Math.hypot(jx, jy) };
}

function follow(fields, seed, sign, floor) {
  const ny = fields.shape[0], nx = fields.shape[1];
  const points = [];
  const stride = 0.5;
  let i = seed.i, j = seed.j;
  for (let count = 0; count < 600; count++) {
    if (i < 0 || j < 0 || i > nx - 1 || j > ny - 1) break;
    const here = directionAt(fields, i, j);
    if (!here || here.magnitude < floor) break;
    points.push({ i: i, j: j });
    const mid = directionAt(
      fields, i + sign * 0.5 * stride * here.di, j + sign * 0.5 * stride * here.dj
    );
    if (!mid) break;
    i += sign * stride * mid.di;
    j += sign * stride * mid.dj;
  }
  return points;
}

function traceStreamlines(fields, seedsAcross, seedsDown) {
  if (!fields.arrays.Jx || fields.shape.length !== 2) return [];
  const ny = fields.shape[0], nx = fields.shape[1];
  let largest = 0;
  for (let index = 0; index < fields.arrays.Jx.length; index++) {
    const size = Math.hypot(fields.arrays.Jx[index], fields.arrays.Jy[index]);
    if (isFinite(size) && size > largest) largest = size;
  }
  if (!(largest > 0)) return [];
  const floor = 1e-3 * largest;
  const lines = [];
  for (let across = 0; across < seedsAcross; across++) {
    for (let down = 0; down < seedsDown; down++) {
      const seed = {
        i: ((across + 0.5) / seedsAcross) * (nx - 1),
        j: ((down + 0.5) / seedsDown) * (ny - 1),
      };
      const behind = follow(fields, seed, -1, floor).reverse();
      const ahead = follow(fields, seed, 1, floor);
      const traced = behind.concat(ahead.slice(1));
      if (traced.length > 1) lines.push(traced);
    }
  }
  return lines;
}

function drawStreamlines(box, fields, lines) {
  const ny = fields.shape[0], nx = fields.shape[1];
  const pen = box.pen;
  pen.strokeStyle = "rgba(255, 255, 255, 0.75)";
  pen.lineWidth = 1;
  for (const points of lines) {
    pen.beginPath();
    points.forEach((point, index) => {
      const px = (point.i / (nx - 1)) * box.width;
      const py = (1 - point.j / (ny - 1)) * box.height;
      if (index === 0) pen.moveTo(px, py); else pen.lineTo(px, py);
    });
    pen.stroke();
  }
}
