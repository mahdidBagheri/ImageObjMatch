# Top/bottom object matching

## Dataset layout

Choose a data-root directory containing an `images` folder and one or more
LabelMe annotation folders whose names start with `_`:

```text
data-root/
├── images/
│   └── example.bmp
├── _labels_a/
│   └── example.json
└── _labels_b/
    └── example.json
```

For each image, annotations with the same base filename are loaded from every
`_...` folder and overlapping outlines are aggregated using the merge-IoU
threshold. The visual editor's **Browse…** button selects this data root.

In the visual editor, select either a two-camera layout (top and bottom) or a
four-camera layout. Four-camera images are split at both midpoints, and matches
are constrained to top-left↔bottom-left and top-right↔bottom-right. The canvas
supports button, mouse-wheel, and keyboard zoom with scrollbars for navigation.
The selection is stored per data root in `match_editor.metadata.json`. Missing
or invalid metadata is initialized and saved as a two-camera layout.

`match_objects.py` merges duplicate LabelMe outlines across all `_...`
annotation folders (annotation labels are ignored), divides each composite
image at half its height, and maps objects from the top camera to objects from
the bottom camera. The older `dastchin`/`goo` layout is still accepted when no
underscore-prefixed label folders exist.

Initial matches minimize the total length of the center-to-center lines. The
majority of these matches are then used to robustly estimate a 2-D affine
mapping from top-camera coordinates to bottom-camera coordinates. Calibration
is separate for each capture session and image resolution. The final Hungarian
assignment is refined to minimize error from the predicted transformed
locations, resolving cases where multiple assignments have equally
non-crossing lines. Refinement is constrained to preserve the minimum
intersection count. If one image half has extra objects, it may also change
which objects remain unmatched when that better agrees with the mapping.

Run one image:

```powershell
.\.venv\Scripts\python.exe match_objects.py "1327  20251110 164408.bmp"
```

Run all images:

```powershell
.\.venv\Scripts\python.exe match_objects.py
```

Results are written to `images_dastchin_goo/matches`. Each image gets:

- `*.matches.json`: machine-readable mappings, merged source observations, and
  unmatched objects.
- `*.matches.svg`: the source image with numbered matching lines. White crosses
  mark unmatched objects.

Raster visualizations are also written to the top-level `output` directory as
BMP images. Colored lines join matched centers and white crosses mark unmatched
objects. `output/linear_transforms.json` records the calibrated affine mappings.
Use `--visual-output another_folder` to change that location.

Each matched object pair is exported to the top-level `crops` directory as one
BMP image. The top-camera crop is on the left, the bottom-camera crop is on the
right, and a white divider separates them. Both sides use the same dimensions
and padding. Use `--crop-output another_folder` to change that location.

The method intentionally uses only geometric order. It does not yet compare
the visual appearance of individual objects, so it should be treated as the
requested first-pass assumption rather than ground truth.
