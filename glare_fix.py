#!/usr/bin/env python3
"""
glare_fix.py - remove the lid-reflection haze from glossy prints scanned in a
camera-in-a-box photo scanner (e.g. QPIX PS970H).

Why the haze exists
  The scanner is a camera looking down into a lit box. A glossy print acts as a
  weak mirror, so the camera sees the lit ceiling of the box reflected over the
  whole photo (the haze) - except where the camera's own dark opening reflects
  back, which shows up as a clear, slightly wavy rectangle near the centre.
  Reflected light *adds* to the image, so it can be subtracted in linear light.

What this script does, per image
  1. Finds the clear rectangle (its position/size shift a little with each
     print's curl) and follows the wobble of each edge.
  2. Measures how much brighter the hazy side is right at the edge, separately
     for R, G and B  ->  haze strength k.
  3. Subtracts k x (haze profile) everywhere outside the rectangle, with a soft
     edge. The haze profile (haze_profile.npy) describes how the haze fades
     towards the left/right of the frame; it was measured from a batch of scans
     from this box (rebuild it with --build-profile for a different box).
  4. Skips prints with no detectable haze (matte/lustre prints) and flags
     uncertain detections as NEEDS-REVIEW in report.csv instead of guessing.

Usage
  python3 glare_fix.py IN_DIR OUT_DIR [--masks] [--preview] [--strength 1.0]
  python3 glare_fix.py IN_DIR OUT_DIR --only IMAG0024.JPG IMAG0029.JPG
  python3 glare_fix.py IN_DIR OUT_DIR --overrides overrides.json
  python3 glare_fix.py IN_DIR --build-profile        (re-measure the haze profile)

overrides.json lets you fix a detection by hand (full-resolution pixels,
x0,y0 = top-left, x1,y1 = bottom-right of the clear rectangle) and/or force k:
  {"IMAG0029.JPG": {"rect": [1700, 1000, 3300, 2150]},
   "IMAG0028.JPG": {"strength": 0.8},
   "IMAG0005.JPG": {"skip": true}}
"""
import argparse, csv, itertools, json, os, sys, glob
from concurrent.futures import ProcessPoolExecutor
import numpy as np, cv2
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
WW = 1228                      # working width (1/4 of a 4912 px scan)
# search windows for the four rectangle edges, as fractions of width/height
RNGF = dict(x0=(0.277, 0.423), x1=(0.619, 0.749), y0=(0.208, 0.415), y1=(0.562, 0.757))
G0, G1 = 3, 8                  # sample band either side of an edge (working px)
OFFS = np.arange(-24, 25)      # profile offsets across an edge (<0 = hazy side)
K3 = np.ones((3, 3), np.uint8)

def lin(a):
    a = np.asarray(a, np.float32); return np.where(a <= 0.04045, a / np.float32(12.92), ((a + np.float32(0.055)) / np.float32(1.055)) ** np.float32(2.4)).astype(np.float32)
def srgb(a):
    a = np.clip(a, 0, 1); return np.where(a <= 0.0031308, a * 12.92, 1.055 * np.power(a, 1 / 2.4) - 0.055)

# ---------------------------------------------------------------- detection
def dstrip(F, key, p, t):
    if key == 'x0': return F[t][:, p - G1:p - G0].mean(1) - F[t][:, p + G0:p + G1].mean(1)
    if key == 'x1': return F[t][:, p + G0:p + G1].mean(1) - F[t][:, p - G1:p - G0].mean(1)
    if key == 'y0': return F[p - G1:p - G0][:, t].mean(0) - F[p + G0:p + G1][:, t].mean(0)
    return F[p + G0:p + G1][:, t].mean(0) - F[p - G1:p - G0][:, t].mean(0)

def consist(d, k): return np.exp(-((d - k) / (0.35 * k + 0.004)) ** 2)

def _peaks(sc, lo, n=4, sep=4):
    out = []
    for i in np.argsort(-sc):
        if all(abs(i - j) >= sep for j in out): out.append(int(i))
        if len(out) == n: break
    return [lo + i for i in out]

def ranges(H, W):
    return {k: (int(a * (W if k[0] == 'x' else H)), int(b * (W if k[0] == 'x' else H))) for k, (a, b) in RNGF.items()}

def fit_rect(F):
    H, W = F.shape; RNG = ranges(H, W)
    rect = dict(x0=int(.334 * W), x1=int(.676 * W), y0=int(.305 * H), y1=int(.665 * H))
    OX0, OX1, OY0, OY1 = RNG['x0'][0], RNG['x1'][1], RNG['y0'][0], RNG['y1'][1]
    best = None
    for it in range(3):
        cands = {}
        for key in RNG:
            lo, hi = RNG[key]
            t = np.arange(rect['y0'] + 20, rect['y1'] - 20) if key[0] == 'x' else np.arange(rect['x0'] + 20, rect['x1'] - 20)
            Dm = np.stack([dstrip(F, key, p, t) for p in range(lo, hi)])
            fr = (Dm > 0.004).mean(1); p30 = np.percentile(Dm, 30, axis=1); med = np.median(Dm, 1)
            ps = sorted(set(_peaks(fr, lo) + _peaks(p30, lo)))
            cands[key] = [(p, float(fr[p - lo]), float(med[p - lo])) for p in ps]
        II = {}
        def dark_ii(kc):
            q = round(kc, 3)
            if q not in II:
                b = np.zeros_like(F, dtype=np.float64); b[OY0:OY1, OX0:OX1] = F[OY0:OY1, OX0:OX1] < 0.5 * q
                II[q] = cv2.integral(b)
            return II[q]
        def rsum(ii, x0, x1, y0, y1): return ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]
        best = None
        for c in itertools.product(*[cands[k] for k in ('x0', 'x1', 'y0', 'y1')]):
            w = c[1][0] - c[0][0]; h = c[3][0] - c[2][0]
            if not (0.244 * W <= w <= 0.423 * W and 0.244 * H <= h <= 0.488 * H): continue
            m = np.array([v[2] for v in c]); kc = np.median(m)
            if kc <= 0.002: continue
            pen = np.abs(m - kc).sum() / (kc + 0.005)
            ii = dark_ii(kc)
            viol = rsum(ii, OX0, OX1, OY0, OY1) - rsum(ii, c[0][0] - 3, c[1][0] + 3, c[2][0] - 3, c[3][0] + 3)
            score = sum(v[1] for v in c) - pen - viol / 150.0
            if best is None or score > best[0]: best = (score, c, kc)
        if best is None: return None, 0.0
        new = dict(x0=best[1][0][0], x1=best[1][1][0], y0=best[1][2][0], y1=best[1][3][0])
        if new == rect: break
        rect = new
    return rect, float(best[2])

def refine(F, rect, k, D=12):
    """let each edge wobble (+-D px, max 1 px per step) to follow the real reflection edge"""
    out = {}
    for key in ('x0', 'x1', 'y0', 'y1'):
        base = rect[key]
        t = np.arange(rect['y0'] - D, rect['y1'] + D) if key[0] == 'x' else np.arange(rect['x0'] - D, rect['x1'] + D)
        offs = np.arange(-D, D + 1)
        C = np.stack([consist(dstrip(F, key, base + o, t), k) for o in offs]).astype(np.float32)
        C = cv2.blur(C, (9, 1)) - 0.004 * np.abs(offs)[:, None]
        n = len(t); acc = C[:, 0].copy(); bp = np.zeros((len(offs), n), int)
        for i in range(1, n):
            cand = np.stack([np.r_[-1e9, acc[:-1]], acc, np.r_[acc[1:], -1e9]])
            a = cand.argmax(0); acc = cand.max(0) + C[:, i]; bp[:, i] = np.arange(len(offs)) + a - 1
        path = np.zeros(n, int); path[-1] = acc.argmax()
        for i in range(n - 1, 0, -1): path[i - 1] = bp[path[i], i]
        pos = cv2.GaussianBlur((base + offs[path]).astype(np.float32).reshape(1, -1), (0, 0), 2).ravel()
        out[key] = (t, pos)
    return out

def edge_profiles(img_lin, edges):
    P = []; H, W = img_lin.shape[:2]
    for key, (t, p) in edges.items():
        t2 = t[15:-15]; p2 = np.round(p).astype(int)[15:-15]
        sgn = -1 if key in ('x0', 'y0') else 1
        q = np.stack([p2 - sgn * o for o in OFFS])
        T = t2[None, :].repeat(len(OFFS), 0)
        P.append(img_lin[T, np.clip(q, 0, W - 1)] if key[0] == 'x' else img_lin[np.clip(q, 0, H - 1), T])
    return P   # list per edge: offs x samples x C

def clear_mask(edges, shape, shift, sigma):
    """1 = hazy, 0 = clear window; soft edge"""
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    def e(key, q):
        t, p = edges[key]; sgn = -1 if key in ('x0', 'y0') else 1
        return np.interp(q, t, p + sgn * shift)
    inside = (xx >= e('x0', yy)) & (xx <= e('x1', yy)) & (yy >= e('y0', xx)) & (yy <= e('y1', xx))
    return cv2.GaussianBlur((~inside).astype(np.float32), (0, 0), sigma)

# ---------------------------------------------------------------- per image
def load_profile(shape):
    p = os.path.join(HERE, 'haze_profile.npy')
    if not os.path.exists(p): return np.ones(shape, np.float32)
    return cv2.resize(np.load(p), (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)

def analyse(path, override=None):
    override = override or {}
    img = cv2.imread(path, cv2.IMREAD_COLOR)[..., ::-1]
    H0, W0 = img.shape[:2]; sc = W0 / WW
    small = cv2.resize(img, (WW, int(round(H0 / sc))), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    Ls = lin(small); L = Ls.mean(2); F = cv2.erode(L, K3)
    res = dict(file=os.path.basename(path), status='', k=[0, 0, 0], rect=None, note='')
    if override.get('skip'):
        res['status'] = 'skipped (override)'; return res, None, img
    if 'rect' in override:
        x0, y0, x1, y1 = [int(round(v / sc)) for v in override['rect']]
        rect = dict(x0=x0, x1=x1, y0=y0, y1=y1)
        t = lambda key: np.arange(y0 + 20, y1 - 20) if key[0] == 'x' else np.arange(x0 + 20, x1 - 20)
        kL = float(np.median(np.concatenate([dstrip(F, k_, rect[k_], t(k_)) for k_ in rect])))
    else:
        rect, kL = fit_rect(F)
        if rect is None:
            res['status'] = 'no haze found'; return res, None, img
    edges = refine(F, rect, max(kL, 0.006))
    P = edge_profiles(Ls, edges)
    o_sel = (OFFS >= -22) & (OFFS <= -14); i_sel = (OFFS >= 12) & (OFFS <= 22)
    per_edge = [float(np.median((p[o_sel].mean(0) - p[i_sel].mean(0)).mean(1))) for p in P]
    Pall = np.concatenate(P, 1)
    k = np.median(Pall[o_sel].mean(0) - Pall[i_sel].mean(0), axis=0)
    prof = np.median((Pall - Pall[-1:]).mean(2), axis=1)
    pk = max(prof[o_sel].mean(), 1e-4)
    def cross(fr):
        idx = np.where((prof[:-1] >= fr * pk) & (prof[1:] < fr * pk))[0]
        return float(OFFS[idx[-1]]) if len(idx) else 0.0
    shift = float(np.clip(-cross(0.5), 0, 8)); sigma = float(np.clip((cross(0.25) - cross(0.75)) / 1.35, 2, 8))
    m = clear_mask(edges, L.shape, shift, sigma)
    g = load_profile(L.shape)
    # physical limit: haze can't be brighter than the darkest hazy pixel
    region = (m > 0.99); region[:int(.06 * L.shape[0])] = False; region[-int(.06 * L.shape[0]):] = False
    region[:, :int(.04 * WW)] = False; region[:, -int(.04 * WW):] = False
    kmax = np.array([np.percentile(Ls[..., c][region] / g[region], 0.3) for c in range(3)])
    med_e = float(np.median(per_edge))
    res.update(rect=[int(rect['x0'] * sc), int(rect['y0'] * sc), int(rect['x1'] * sc), int(rect['y1'] * sc)],
               edges=[round(x, 3) for x in per_edge], shift=shift, sigma=sigma)
    k = np.clip(k, 0, None)
    res['dark_ratio'] = round(float(kmax.mean() / max(k.mean(), 1e-6)), 2)
    k = k * float(override.get('strength', 1.0))
    res['k'] = [round(float(x), 4) for x in k]
    if 'rect' not in override:
        if med_e < 0.012 or float(np.max(k)) < 0.01:
            res['status'] = 'no haze found'; return res, None, img
        weak = sum(e < 0.2 * med_e for e in per_edge)
        if weak >= 2:
            if res['dark_ratio'] < 0.3:
                res['status'] = 'no haze found'; return res, None, img
            res['status'] = 'NEEDS-REVIEW'; res['note'] = 'window edges unclear - copied unchanged'
            return res, None, img
        if weak == 1 and res['dark_ratio'] < 0.3:
            res['status'] = 'no haze found'; return res, None, img
        if k.max() > 0.03 and k.min() < 0.15 * k.max():
            res['status'] = 'NEEDS-REVIEW'; res['note'] = 'haze colour implausible - copied unchanged'
            return res, None, img
        if res['dark_ratio'] < 0.3: res['note'] = 'check: strength may be high'
    res['status'] = 'corrected'
    haze = cv2.resize(m * g, (W0, H0), interpolation=cv2.INTER_LINEAR)
    return res, (haze, k), img

def correct(img, haze, k):
    out = np.empty_like(img)
    for y in range(0, img.shape[0], 512):      # in strips to keep memory low
        out[y:y + 512] = _correct(img[y:y + 512], haze[y:y + 512], k)
    return out

def _correct(img, haze, k):
    li = lin(img.astype(np.float32) / 255)
    H = np.maximum(haze[..., None] * k[None, None, :].astype(np.float32), 1e-6)
    x = li / H
    # subtract the haze; pixels already darker than the predicted haze are eased
    # towards black instead of clipped (keeps shadow detail, no hard black blobs)
    s = x / np.power(1 + x ** 4, 0.25)
    return (srgb(li - H * s) * 255 + 0.5).astype(np.uint8)

def process(args):
    path, out_dir, override, opts = args
    try:
        res, hk, img = analyse(path, override)
        name = os.path.basename(path)
        src = Image.open(path); exif = src.info.get('exif'); icc = src.info.get('icc_profile')
        kw = dict(quality=95, subsampling=0)
        if exif: kw['exif'] = exif
        if icc: kw['icc_profile'] = icc
        if hk is None:
            out = img
            if opts['copy_unchanged']:
                Image.fromarray(out).save(os.path.join(out_dir, name), **kw)
        else:
            haze, k = hk
            out = correct(img, haze, np.array(k) * opts['strength'])
            Image.fromarray(out).save(os.path.join(out_dir, name), **kw)
            if opts['masks']:
                mk = np.clip(haze / max(haze.max(), 1e-6) * 255, 0, 255).astype(np.uint8)
                cv2.imwrite(os.path.join(out_dir, 'masks', os.path.splitext(name)[0] + '_mask.png'), mk)
        if opts['preview']:
            a = cv2.resize(img, (1228, int(1228 * img.shape[0] / img.shape[1])), interpolation=cv2.INTER_AREA)
            b = cv2.resize(out, a.shape[1::-1], interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(out_dir, 'previews', os.path.splitext(name)[0] + '_before_after.jpg'),
                        np.concatenate([a, b], 1)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
        return res
    except Exception as e:
        return dict(file=os.path.basename(path), status='ERROR', note=repr(e), k=[0, 0, 0], rect=None)

# ---------------------------------------------------------------- profile
def build_profile(in_dir, files):
    """measure how the haze fades across the frame, from many hazy scans"""
    A = []
    for f in files:
        res, hk, img = analyse(f)
        if hk is None or res['status'] != 'corrected': continue
        haze, k = hk
        small = cv2.resize(img, (614, int(614 * img.shape[0] / img.shape[1])), interpolation=cv2.INTER_AREA)
        L = lin(small.astype(np.float32) / 255).mean(2)
        m = cv2.resize(haze, small.shape[1::-1], interpolation=cv2.INTER_AREA)
        g_old = load_profile(L.shape); m = m / np.maximum(g_old, 1e-3)
        r = L / max(np.mean(k), 1e-3); r[m < 0.98] = np.nan; A.append(r)
        print('  used', os.path.basename(f), flush=True)
    if len(A) < 20: sys.exit('need at least 20 hazy scans to build a profile (found %d)' % len(A))
    A = np.stack(A); g = np.nanpercentile(A, 4, axis=0)
    w = (~np.isnan(g)).astype(np.float32); v = np.minimum(np.nan_to_num(g), 3).astype(np.float32)
    sm = cv2.GaussianBlur(v * w, (0, 0), 20) / np.maximum(cv2.GaussianBlur(w, (0, 0), 20), 1e-6)
    hole = np.isnan(np.nanmedian(A, axis=0)) | (w.mean() and cv2.GaussianBlur(w, (0, 0), 3) < 0.5)
    ring = (cv2.dilate(hole.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0) & ~(cv2.dilate(hole.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)
    sm /= np.median(sm[ring]); sm[hole] = np.minimum(sm[hole], 1.0)
    sm = np.clip(cv2.GaussianBlur(sm, (0, 0), 4), 0.25, 1.3).astype(np.float32)
    np.save(os.path.join(HERE, 'haze_profile.npy'), sm)
    print('wrote', os.path.join(HERE, 'haze_profile.npy'))

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('in_dir'); ap.add_argument('out_dir', nargs='?')
    ap.add_argument('--only', nargs='*', help='file names to process')
    ap.add_argument('--overrides', help='JSON file of per-photo fixes')
    ap.add_argument('--strength', type=float, default=1.0, help='scale all corrections (default 1.0)')
    ap.add_argument('--masks', action='store_true', help='also save each haze mask as PNG (for Pixelmator/Photoshop)')
    ap.add_argument('--preview', action='store_true', help='also save small before/after images')
    ap.add_argument('--no-copy', action='store_true', help="don't copy photos that needed no correction")
    ap.add_argument('--jobs', type=int, default=2)
    ap.add_argument('--build-profile', action='store_true', help='re-measure haze_profile.npy from IN_DIR')
    a = ap.parse_args()
    exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
    files = sorted(f for f in glob.glob(os.path.join(a.in_dir, '*')) if f.lower().endswith(exts))
    if a.only: files = [f for f in files if os.path.basename(f) in set(a.only)]
    if a.build_profile: return build_profile(a.in_dir, files)
    if not a.out_dir: ap.error('OUT_DIR is required')
    if os.path.abspath(a.out_dir) == os.path.abspath(a.in_dir): ap.error('OUT_DIR must differ from IN_DIR')
    os.makedirs(a.out_dir, exist_ok=True)
    if a.masks: os.makedirs(os.path.join(a.out_dir, 'masks'), exist_ok=True)
    if a.preview: os.makedirs(os.path.join(a.out_dir, 'previews'), exist_ok=True)
    over = json.load(open(a.overrides)) if a.overrides else {}
    opts = dict(strength=a.strength, masks=a.masks, preview=a.preview, copy_unchanged=not a.no_copy)
    jobs = [(f, a.out_dir, over.get(os.path.basename(f)), opts) for f in files]
    rows = []
    with ProcessPoolExecutor(a.jobs) as ex:
        for r in ex.map(process, jobs):
            rows.append(r); print('%-16s %-15s k=%s dark=%s edges=%s %s' % (r['file'], r['status'], r.get('k'), r.get('dark_ratio'), r.get('edges'), r.get('note', '')), flush=True)
    rp = os.path.join(a.out_dir, 'report.csv')
    with open(rp, 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['file', 'status', 'k_r', 'k_g', 'k_b', 'rect_x0_y0_x1_y1', 'note'])
        for r in rows: w.writerow([r['file'], r['status'], *r['k'], r.get('rect'), r.get('note', '')])
    n = lambda s: sum(r['status'] == s for r in rows)
    print('\n%d corrected, %d no haze, %d need review, %d errors  ->  %s' % (n('corrected'), n('no haze found'), n('NEEDS-REVIEW'), n('ERROR'), rp))

if __name__ == '__main__':
    main()
