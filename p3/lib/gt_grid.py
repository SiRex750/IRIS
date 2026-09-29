"""Ground-truth flow re-sampled onto frame t's grid by forward splatting.

For display frame t, Sintel gives flow t-1 -> t (SintelSeq.flow_into(t)) and the exclusion
mask (SintelSeq.bad_mask_into(t)), both on frame t-1's pixels. A codec block sits in frame t,
so GT for it must be expressed on frame t's grid:

  each valid, non-occluded t-1 pixel (x, y) with flow (u, v) lands in frame t's 4x4 cell
      (floor((x+u+0.5)/4), floor((y+v+0.5)/4))
  landings outside the image are dropped. Per cell: count, sum u, sum v, sum u^2+v^2.

Cell GT = sums / count, gt_cover = count / 16.
Block GT = sums over the cells the block covers; gt_cover = count / (w*h) with the FULL block
area, so encoder-padding rows (>= image height) count as uncovered.
gt_std = sqrt(E[u^2+v^2] - |E[(u,v)]|^2): spread of GT motion inside the cell/block (large on
motion edges).
"""
import numpy as np

CELL = 4


class GTGrid:
    def __init__(self, flow, bad, cell=CELL):
        H, W = flow.shape[:2]
        self.H, self.W, self.cell = H, W, cell
        self.Hc, self.Wc = -(-H // cell), -(-W // cell)
        yy, xx = np.mgrid[0:H, 0:W]
        u = flow[..., 0].astype(np.float64)
        v = flow[..., 1].astype(np.float64)
        X = np.floor(xx + u + 0.5)
        Y = np.floor(yy + v + 0.5)
        ok = (~bad) & np.isfinite(X) & np.isfinite(Y) & (X >= 0) & (X < W) & (Y >= 0) & (Y < H)
        ci = (Y[ok].astype(np.int64) // cell) * self.Wc + (X[ok].astype(np.int64) // cell)
        n = self.Hc * self.Wc
        uo, vo = u[ok], v[ok]
        shp = (self.Hc, self.Wc)
        self.count = np.bincount(ci, minlength=n).astype(np.float64).reshape(shp)
        self.su = np.bincount(ci, weights=uo, minlength=n).reshape(shp)
        self.sv = np.bincount(ci, weights=vo, minlength=n).reshape(shp)
        self.sq = np.bincount(ci, weights=uo * uo + vo * vo, minlength=n).reshape(shp)
        self._sat = None

    @staticmethod
    def _stats(count, su, sv, sq, area):
        with np.errstate(invalid="ignore", divide="ignore"):
            gx, gy = su / count, sv / count
            var = sq / count - (gx * gx + gy * gy)
        std = np.sqrt(np.clip(var, 0, None))
        return gx, gy, count / area, std

    def cell_gt(self):
        """(gx, gy, gt_cover, gt_std), each (Hc, Wc). NaN GT where count == 0."""
        return self._stats(self.count, self.su, self.sv, self.sq, float(self.cell * self.cell))

    def _build_sat(self):
        # cell-grid summed-area tables, padded with zero rows/cols so blocks reaching into
        # encoder padding (beyond the image) sum only their in-image cells
        pad_r, pad_c = 8, 8
        st = []
        for a in (self.count, self.su, self.sv, self.sq):
            a = np.pad(a, ((0, pad_r), (0, pad_c)))
            s = np.cumsum(np.cumsum(a, 0), 1)
            st.append(np.pad(s, ((1, 0), (1, 0))))
        self._sat = st

    def block_gt(self, x0, y0, w, h):
        """Block GT for blocks with top-left (x0, y0) and size (w, h) in frame t (arrays).

        Returns (gx, gy, gt_cover, gt_std, count). All blocks must sit on the cell grid.
        """
        x0, y0, w, h = (np.asarray(a).astype(np.int64) for a in (x0, y0, w, h))
        c = self.cell
        assert np.all(x0 % c == 0) and np.all(y0 % c == 0) and np.all(w % c == 0) and np.all(h % c == 0), \
            "block not on the 4-px cell grid"
        if self._sat is None:
            self._build_sat()
        cx0, cy0 = x0 // c, y0 // c
        cx1, cy1 = cx0 + w // c, cy0 + h // c
        maxr, maxc = self._sat[0].shape[0] - 1, self._sat[0].shape[1] - 1
        assert np.all(cx0 >= 0) and np.all(cy0 >= 0) and np.all(cx1 <= maxc) and np.all(cy1 <= maxr), \
            "block beyond padded cell grid"
        sums = [s[cy1, cx1] - s[cy0, cx1] - s[cy1, cx0] + s[cy0, cx0] for s in self._sat]
        gx, gy, cover, std = self._stats(*sums, (w * h).astype(np.float64))
        return gx, gy, cover, std, sums[0]
