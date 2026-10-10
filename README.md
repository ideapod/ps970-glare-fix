# ps970-glare-fix

Removes the hazy reflection that glossy prints pick up in a camera-in-a-box photo
scanner (built for the QPIX PS970H). The scanner's camera sees the lit box lid
reflected in the print, which hazes everything except a clear rectangle in the middle
where the camera's own dark opening reflects back. This tool finds that rectangle on
each scan, measures how strong the haze is, and subtracts it.

Matte or lustre prints with no haze are detected and copied over unchanged.

## Setup (once per machine)

You need Python 3.9 or newer (`python3 --version` to check; on a Mac get it from
python.org or with `brew install python`). Then:

```
git clone <this repo>
cd ps970-glare-fix
./setup.sh
```

This creates a private environment in `.venv/` and installs numpy, OpenCV and Pillow
into it. Nothing is installed system-wide.

## Use

```
./glare-fix "/path/to/scans" "/path/to/output" --preview
```

The output folder gets:

| What | Where |
|---|---|
| Corrected photos (same names, EXIF kept) | `output/` |
| Before/after previews (with `--preview`) | `output/previews/` |
| Haze masks as PNG, for Pixelmator/Photoshop (with `--masks`) | `output/masks/` |
| One line per photo: what was done, haze strength | `output/report.csv` |

Your originals are never modified.

Options:

- `--strength 0.8` scales every correction (below 1 is gentler, above 1 is stronger)
- `--only IMAG0024.JPG IMAG0029.JPG` processes just those files
- `--overrides fixes.json` applies per-photo fixes (see below)
- `--no-copy` skips copying photos that needed no correction
- `--jobs 4` runs 4 photos in parallel (default 2; each one needs about 1 GB of RAM)

## Fixing individual photos

Check the previews. If a photo is over- or under-corrected, or the clear rectangle was
found in the wrong place, make a `fixes.json` file:

```json
{
  "IMAG0028.JPG": {"strength": 0.8},
  "IMAG0029.JPG": {"rect": [1700, 1000, 3300, 2150]},
  "IMAG0003.JPG": {"skip": true}
}
```

`rect` is the clear rectangle in full-size pixels: left, top, right, bottom. Then
re-run with `--overrides fixes.json --only IMAG0028.JPG IMAG0029.JPG IMAG0003.JPG`.

## The haze profile

`haze_profile.npy` records how the haze fades from the middle of the frame towards
the edges. It was measured from about 80 scans made in this particular box. For a
different scanner or box, scan a batch of glossy prints and re-measure it with:

```
./glare-fix "/path/to/those scans" --build-profile
```

## Getting the best scans

The haze is the box lid reflecting in the print. If you can, line the inside of the
lid around the camera opening with matte black flocking paper. With nothing bright
up there to reflect, the haze goes away at the source.
