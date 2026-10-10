# CLAUDE.md — ps970-glare-fix

Notes for Claude (or anyone) working on this repo. User-facing instructions are in README.md.

## What this is

A single-file Python CLI (`glare_fix.py`) that removes the lid-reflection haze from glossy
photo prints scanned in a camera-in-a-box scanner (QPIX PS970H, 4912×3276 JPEG output).
Owner: Mark. The scanner box belongs to someone else, so the physical fix (black flocking
on the lid) isn't an option. The software fix has to carry the load.

## The physics (read this first)

- The camera looks down at the print. A glossy print is a weak mirror, so the camera sees the
  lit lid of the box reflected over the whole photo. That reflection is the haze.
- The camera's own dark opening reflects back as a **clear, wavy-edged rectangle** near the
  centre (about x 0.28–0.75, y 0.21–0.76 of the frame). A scan of the empty box shows it.
- Reflected light is **additive in linear light**: `observed = true + k_c · g(x,y) · m(x,y)`
  - `m` = 1 outside the clear window, 0 inside, with a soft edge (σ ≈ 4–6 px at 1/4 scale; the
    reflection is out of focus)
  - `k_c` = per-print, per-channel haze strength (≈0.02–0.4 linear; depends on gloss and curl)
  - `g` = spatial haze profile, ≈1 next to the window and fading to ≈0.3–0.5 at the left and
    right frame edges. It's measured once per box and stored in `haze_profile.npy`
    (614×410 float32, 1/8 scale).
- The window's position and size **shift by roughly ±10%** from print to print (curl makes the
  print a curved mirror), and its edges wobble. That's why a single fixed mask can't work.
- Matte or lustre prints show little or no haze. Black-and-white prints in this batch
  (IMAG0001–0014) mostly have none.

## Pipeline (`analyse()` → `correct()`)

All detection runs at working width `WW=1228` (1/4 scale), in linear light (`lin()`), on a 3×3
eroded luminance `F`.

1. `fit_rect(F)`: for each edge (x0, x1, y0, y1), score every candidate position inside the
   `RNGF` search window by the step across it (`dstrip`: mean of a band 3–8 px outside minus
   inside). Take the top peaks by two metrics: the fraction of samples with a positive step,
   and the 30th-percentile step. Then choose the combination of four that maximises the sum of
   fractions, minus a penalty for disagreement between the edges' median steps (the haze is the
   same on every edge), minus a penalty for very dark pixels outside the rectangle (haze can't
   coexist with near-black). Three refinement iterations.
2. `refine()`: a Viterbi path lets each edge wobble ±12 px (max 1 px per step) along its
   length, scoring how consistent the local step is with the global `k`.
3. `edge_profiles()`: sample perpendicular profiles (−24…+24 px) across the refined edges.
   - `k_c` = median of (mean over −22…−14) − (mean over +12…+22), per channel. Measuring
     close to the edge underestimates by about 2×, because the edge is soft.
   - The edge 50% point (`shift`, the mask moves outward) and the 25–75% width (`sigma`) come
     from the median luminance profile.
4. Gates (skip/flag rather than guess):
   - `per_edge`: k measured on each edge separately. If 2 or more edges are below 0.2× the
     median, the result is `no haze found` (when `dark_ratio < 0.3`) or otherwise
     `NEEDS-REVIEW`, and the photo is copied unchanged.
   - `dark_ratio` = (0.3rd percentile of `L/g` outside the window) ÷ k. Below 1 means some
     outside pixels are darker than the predicted haze, so k is suspect.
5. `correct()`: subtract `H = k·g·m` in linear light with a soft knee
   (`li − H·s(li/H)`, `s(x)=x/(1+x⁴)^¼`), so pixels already darker than the predicted haze ease
   to black instead of clipping. Processed in 512-row strips to keep memory low.
6. Output: JPEG q95, 4:4:4, with the original EXIF and ICC; optional mask PNG, before/after
   preview, and `report.csv`.

`--build-profile` rebuilds `haze_profile.npy`: run detection on all scans, take the
4th-percentile of `L/k` per pixel outside each window, fill the gaps, smooth (σ 20), and
normalise so the ring around the window = 1.

## Known issues / next steps

1. **False positives on haze-free photos.** IMAG0003, 0015, 0016, 0031 and 0052 got "corrected"
   when a content edge (for example a black cat against a light background) was taken for the
   window. That leaves a faint box. A fix was drafted but not yet applied:
   - if exactly 1 weak edge and `dark_ratio < 0.3` → `no haze found`
   - if `k.max() > 0.03 and k.min() < 0.15·k.max()` (implausible haze colour; real haze is
     close to neutral or bluish) → `NEEDS-REVIEW`
2. **k sometimes too high** on very hazy prints (0028, 0090, 0099): shadows come out crushed
   or contrasty. Content edges on the top/bottom window edges inflate the median. A candidate
   fix is a more robust k (for example the lower-middle of the per-edge values, or a cap from
   `dark_ratio`).
3. **Edge detection fails on busy photos** (content edges near the window, such as 0064 with
   the woman's shoulder). Use `overrides.json` → `rect` for these.
4. Residual seams where the real edge is wavier than ±12 px, plus specular glints along the
   window edge that are in the original scan (0024, right-hand jacket). Not fixable by
   subtraction.
5. The haze profile `g` is assumed fixed, but it probably shifts a little with the window.
   There's a faint content artefact in it just left of the window.
6. A faint horizontal line about half-way down some scans is the tray seam showing through.
   It isn't handled.

## Testing / evaluating changes

There are no unit tests. Evaluate visually:
- Run with `--preview --only …` on the reference set: **0024** (dark jackets crossing the
  window edges, the best seam test), **0039**, **0062**, **0100**, **0026** (strong haze, should
  look good), **0028** and **0090** (very strong haze, watch for over-correction), and
  **0003**, **0015**, **0016** (no haze, must come out unchanged).
- A boosted-shadow view (`clip(img*3)^0.7`) makes the window and any leftover seam obvious.
- Reference full run (before the false-positive fix): 114 corrected, 19 no haze, 3 review.

## Environment

Python ≥ 3.9, numpy, opencv-python-headless, Pillow (`requirements.txt`). Run `./setup.sh`
to create `.venv/`. The `./glare-fix` wrapper runs the script with that venv. Each worker
needs about 1 GB of RAM, so the default is `--jobs 2`.
