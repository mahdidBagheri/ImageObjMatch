# Matchroom visual editor

Launch the review workspace from this directory:

```powershell
.\.venv\Scripts\python.exe match_editor.py
```

## Review workflow

1. Use **Browse…** to select a data root containing `images/` and one or more
   `_.../` LabelMe annotation folders.
2. Choose an image from the queue or filter by filename.
3. Click an outlined object in either camera.
4. Click its partner in the other camera. Any conflicting old pairs are
   removed automatically.
5. Inspect the pair and its live crop preview in the right panel.
6. Use **Unpair selected**, **Undo**, **Redo**, or **Auto-match** as needed.
7. Choose **Save + regenerate crops**.

Saving updates the reviewed match JSON and SVG, refreshes the annotated BMP,
removes stale crops for the current image, and exports the new crop set.

Keyboard shortcuts: `Ctrl+S` saves, `Ctrl+Z` undoes, `Ctrl+Y` redoes,
`Delete` unpairs the current selection, and `Esc` clears the selection.
