# Matchroom visual editor

Launch the review workspace from this directory:

```powershell
.\.venv\Scripts\python.exe match_editor.py
```

## Review workflow

1. Use **Browse…** to select a data root containing `images/` and one or more
   `_.../` LabelMe annotation folders.
2. Choose **2** or **4** cameras for the selected dataset. Four-camera mode
   splits the image at its horizontal and vertical midpoints; top-left pairs
   only with bottom-left, and top-right pairs only with bottom-right. The choice
   is saved in `match_editor.metadata.json` in the data root and restored the
   next time that folder is opened. A folder without metadata starts in
   two-camera mode and gets a metadata file automatically.
3. Choose an image from the queue or filter by filename.
4. Use **−**, **Fit**, **+**, the mouse wheel, or `Ctrl+-`/`Ctrl++` to zoom.
   Scrollbars navigate images that are larger than the canvas.
5. Click an outlined object in a top camera.
6. Click its partner in the camera directly below it. Any conflicting old pairs are
   removed automatically.
7. Inspect the pair and its live crop preview in the right panel.
8. Use **Unpair selected**, **Undo**, **Redo**, or **Auto-match** as needed.
9. Choose **Save + regenerate crops**.

Saving updates the reviewed match JSON and SVG, refreshes the annotated BMP,
removes stale crops for the current image, and exports the new crop set.

Keyboard shortcuts: `Ctrl+S` saves, `Ctrl+Z` undoes, `Ctrl+Y` redoes,
`Delete` unpairs the current selection, `Esc` clears the selection, and
`Ctrl+0` restores fit-to-window zoom.
