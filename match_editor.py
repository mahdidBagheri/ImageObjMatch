#!/usr/bin/env python3
r"""Visual review and crop regeneration UI for object matches.

The editor intentionally uses only the Python standard library, like the
matching pipeline itself. Run it from this directory with::

    .\.venv\Scripts\python.exe match_editor.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import tkinter as tk
from dataclasses import asdict
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from match_objects import (
    MATCH_COLORS,
    LinearTransform,
    Object,
    annotation_folders,
    calibration_key,
    count_intersections,
    image_files,
    load_image_objects,
    object_json,
    pair_cost,
    render_bmp,
    save_matched_crops,
    svg_visualization,
    transform_guided_match,
)


BG = "#F2F4F7"
PANEL = "#FFFFFF"
INK = "#17212B"
MUTED = "#667085"
LINE = "#DDE2E8"
NAVY = "#142536"
NAVY_2 = "#21384C"
TEAL = "#17A88B"
TEAL_DARK = "#10806B"
AMBER = "#E6A23C"
RED = "#D84A4A"
CANVAS_BG = "#0E1720"

HEX_COLORS = ["#%02x%02x%02x" % color for color in MATCH_COLORS]


def read_bmp_rows(path: Path) -> tuple[int, int, list[bytes]]:
    """Read a 24-bit uncompressed BMP into top-to-bottom RGB rows."""
    data = path.read_bytes()
    if data[:2] != b"BM":
        raise ValueError(f"Not a BMP file: {path.name}")
    pixel_offset = struct.unpack_from("<I", data, 10)[0]
    width, signed_height = struct.unpack_from("<ii", data, 18)
    bpp = struct.unpack_from("<H", data, 28)[0]
    compression = struct.unpack_from("<I", data, 30)[0]
    if bpp != 24 or compression != 0:
        raise ValueError("The editor supports uncompressed 24-bit BMP images")
    height = abs(signed_height)
    stride = (width * 3 + 3) & ~3
    rows: list[bytes] = []
    for y in range(height):
        storage_y = height - 1 - y if signed_height > 0 else y
        raw = data[pixel_offset + storage_y * stride : pixel_offset + storage_y * stride + width * 3]
        rgb = bytearray(width * 3)
        for x in range(width):
            b, g, r = raw[x * 3 : x * 3 + 3]
            rgb[x * 3 : x * 3 + 3] = bytes((r, g, b))
        rows.append(bytes(rgb))
    return width, height, rows


def scaled_photo(
    rows: list[bytes], width: int, height: int, max_width: int, max_height: int
) -> tuple[tk.PhotoImage, float]:
    scale = min(max_width / max(width, 1), max_height / max(height, 1), 1.0)
    out_width = max(1, int(width * scale))
    out_height = max(1, int(height * scale))
    pixels = bytearray(out_width * out_height * 3)
    for out_y in range(out_height):
        source = rows[min(height - 1, int(out_y / scale))]
        row_offset = out_y * out_width * 3
        for out_x in range(out_width):
            source_x = min(width - 1, int(out_x / scale)) * 3
            target_x = row_offset + out_x * 3
            pixels[target_x : target_x + 3] = source[source_x : source_x + 3]
    payload = f"P6\n{out_width} {out_height}\n255\n".encode("ascii") + pixels
    return tk.PhotoImage(data=payload, format="PPM"), scale


class MatchDocument:
    def __init__(self, dataset: Path, image_path: Path, merge_iou: float = 0.5):
        self.dataset = dataset
        self.image_path = image_path
        (
            _,
            self.annotation_paths,
            self.width,
            self.height,
            self.top,
            self.bottom,
        ) = load_image_objects(dataset, image_path, merge_iou)
        self.transform = LinearTransform()
        self.automatic_pairs: list[tuple[Object, Object]] = []
        self.pairs: list[tuple[Object, Object]] = []
        self.history: list[list[tuple[int, int]]] = []
        self.future: list[list[tuple[int, int]]] = []
        self.dirty = False
        self._load_existing()

    @property
    def result_path(self) -> Path:
        return self.dataset / "matches" / f"{self.image_path.stem}.matches.json"

    def _load_existing(self) -> None:
        saved = None
        if self.result_path.exists():
            try:
                saved = json.loads(self.result_path.read_text(encoding="utf-8-sig"))
                values = saved.get("linear_transform", {})
                allowed = LinearTransform.__dataclass_fields__.keys()
                self.transform = LinearTransform(**{key: values[key] for key in allowed if key in values})
            except (OSError, ValueError, TypeError):
                saved = None
        self.automatic_pairs = transform_guided_match(
            self.top, self.bottom, self.width, self.transform
        )
        top_by_id = {item.id: item for item in self.top}
        bottom_by_id = {item.id: item for item in self.bottom}
        loaded: list[tuple[Object, Object]] = []
        if saved:
            for match in saved.get("matches", []):
                top_item = top_by_id.get(match.get("top", {}).get("id"))
                bottom_item = bottom_by_id.get(match.get("bottom", {}).get("id"))
                if top_item is not None and bottom_item is not None:
                    loaded.append((top_item, bottom_item))
        self.pairs = loaded or list(self.automatic_pairs)
        self._sort()

    def _snapshot(self) -> list[tuple[int, int]]:
        return [(top.id, bottom.id) for top, bottom in self.pairs]

    def _restore(self, snapshot: list[tuple[int, int]]) -> None:
        top_by_id = {item.id: item for item in self.top}
        bottom_by_id = {item.id: item for item in self.bottom}
        self.pairs = [
            (top_by_id[top_id], bottom_by_id[bottom_id])
            for top_id, bottom_id in snapshot
            if top_id in top_by_id and bottom_id in bottom_by_id
        ]
        self._sort()

    def _checkpoint(self) -> None:
        self.history.append(self._snapshot())
        self.history = self.history[-50:]
        self.future.clear()

    def _sort(self) -> None:
        self.pairs.sort(key=lambda pair: (pair[0].center_x, pair[0].center_y))

    def pair(self, top: Object, bottom: Object) -> bool:
        if any(a.id == top.id and b.id == bottom.id for a, b in self.pairs):
            return False
        self._checkpoint()
        self.pairs = [
            (a, b) for a, b in self.pairs if a.id != top.id and b.id != bottom.id
        ]
        self.pairs.append((top, bottom))
        self._sort()
        self.dirty = True
        return True

    def unpair(self, top_id: int | None = None, bottom_id: int | None = None) -> bool:
        remaining = [
            (top, bottom)
            for top, bottom in self.pairs
            if not (
                (top_id is not None and top.id == top_id)
                or (bottom_id is not None and bottom.id == bottom_id)
            )
        ]
        if len(remaining) == len(self.pairs):
            return False
        self._checkpoint()
        self.pairs = remaining
        self.dirty = True
        return True

    def reset_automatic(self) -> None:
        if self._snapshot() == [(a.id, b.id) for a, b in self.automatic_pairs]:
            return
        self._checkpoint()
        self.pairs = list(self.automatic_pairs)
        self._sort()
        self.dirty = True

    def undo(self) -> bool:
        if not self.history:
            return False
        self.future.append(self._snapshot())
        self._restore(self.history.pop())
        self.dirty = True
        return True

    def redo(self) -> bool:
        if not self.future:
            return False
        self.history.append(self._snapshot())
        self._restore(self.future.pop())
        self.dirty = True
        return True

    def partner(self, side: str, object_id: int) -> Object | None:
        for top, bottom in self.pairs:
            if side == "top" and top.id == object_id:
                return bottom
            if side == "bottom" and bottom.id == object_id:
                return top
        return None

    def unmatched(self, side: str) -> list[Object]:
        used = {
            (top if side == "top" else bottom).id for top, bottom in self.pairs
        }
        source = self.top if side == "top" else self.bottom
        return [item for item in source if item.id not in used]

    def save(self, output_dir: Path, visual_dir: Path, crop_dir: Path) -> dict:
        paired_ids = {item.id for pair in self.pairs for item in pair}
        result = {
            "image": self.image_path.name,
            "width": self.width,
            "height": self.height,
            "split_y": self.height / 2,
            "annotation_sources": [str(path) for path in self.annotation_paths],
            "calibration_key": calibration_key(self.image_path, self.width, self.height),
            "linear_transform": asdict(self.transform),
            "top_count": len(self.top),
            "bottom_count": len(self.bottom),
            "intersection_count": count_intersections(self.pairs),
            "edited_manually": True,
            "matches": [
                {
                    "match_id": index,
                    "transformation_error_pixels": pair_cost(
                        top, bottom, self.width, self.transform
                    )
                    * self.width,
                    "predicted_bottom_center": list(
                        self.transform.apply(top.center_x, top.center_y)
                    ),
                    "top": object_json(top),
                    "bottom": object_json(bottom),
                }
                for index, (top, bottom) in enumerate(self.pairs)
            ],
            "unmatched_top": [object_json(item) for item in self.top if item.id not in paired_ids],
            "unmatched_bottom": [
                object_json(item) for item in self.bottom if item.id not in paired_ids
            ],
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        visual_dir.mkdir(parents=True, exist_ok=True)
        crop_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / f"{self.image_path.stem}.matches.json"
        temporary = json_path.with_suffix(json_path.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
        os.replace(temporary, json_path)
        try:
            image_href = Path(os.path.relpath(self.image_path, output_dir)).as_posix()
        except ValueError:
            # Windows cannot form a relative path between different drives.
            image_href = self.image_path.resolve().as_uri()
        (output_dir / f"{self.image_path.stem}.matches.svg").write_text(
            svg_visualization(
                image_href,
                self.width,
                self.height,
                self.top,
                self.bottom,
                self.pairs,
            ),
            encoding="utf-8",
        )
        render_bmp(
            self.image_path,
            visual_dir / self.image_path.name,
            self.top,
            self.bottom,
            self.pairs,
        )
        for stale in crop_dir.glob(f"{self.image_path.stem}__match_*.bmp"):
            stale.unlink()
        save_matched_crops(self.image_path, crop_dir, self.pairs)
        self.dirty = False
        return result


class MatchEditor(tk.Tk):
    def __init__(
        self,
        dataset: Path,
        visual_dir: Path,
        crop_dir: Path,
        merge_iou: float,
    ):
        super().__init__()
        self.dataset = dataset
        self.output_dir = dataset / "matches"
        self.visual_dir = visual_dir
        self.crop_dir = crop_dir
        self.merge_iou = merge_iou
        self.paths = list(image_files(dataset, None))
        self.filtered_paths = list(self.paths)
        self.document: MatchDocument | None = None
        self.rows: list[bytes] = []
        self.photo: tk.PhotoImage | None = None
        self.preview_photo: tk.PhotoImage | None = None
        self.image_scale = 1.0
        self.image_origin = (0.0, 0.0)
        self.selected: tuple[str, Object] | None = None
        self.selected_pair: int | None = None
        self._resize_job: str | None = None
        self._refreshing_tree = False

        self.title("Matchroom — Object pairing review")
        self.geometry("1480x900")
        self.minsize(1120, 720)
        self.configure(bg=BG)
        self._configure_styles()
        self._build_ui()
        self._bind_shortcuts()
        self.protocol("WM_DELETE_WINDOW", self._close)
        if self.paths:
            self.after(80, lambda: self.open_image(self.paths[0]))
        else:
            self._set_status("No BMP images were found in the dataset.", error=True)

    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("Header.TFrame", background=NAVY)
        style.configure("TLabel", background=BG, foreground=INK, font=("Segoe UI", 10))
        style.configure("Panel.TLabel", background=PANEL, foreground=INK, font=("Segoe UI", 10))
        style.configure("Muted.TLabel", background=PANEL, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("Title.TLabel", background=NAVY, foreground="white", font=("Segoe UI Semibold", 17))
        style.configure("Subtitle.TLabel", background=NAVY, foreground="#B7C6D4", font=("Segoe UI", 9))
        style.configure("Section.TLabel", background=PANEL, foreground=INK, font=("Segoe UI Semibold", 11))
        style.configure("Count.TLabel", background=PANEL, foreground=TEAL_DARK, font=("Segoe UI Semibold", 10))
        style.configure("TButton", font=("Segoe UI Semibold", 9), padding=(11, 7), borderwidth=0)
        style.configure("Primary.TButton", background=TEAL, foreground="white")
        style.map("Primary.TButton", background=[("active", TEAL_DARK), ("pressed", TEAL_DARK)])
        style.configure("Quiet.TButton", background="#E9EDF1", foreground=INK)
        style.map("Quiet.TButton", background=[("active", "#DDE3E8")])
        style.configure("Danger.TButton", background="#FBE9E9", foreground=RED)
        style.map("Danger.TButton", background=[("active", "#F6D4D4")])
        style.configure("Treeview", rowheight=30, background=PANEL, fieldbackground=PANEL, foreground=INK, borderwidth=0)
        style.configure("Treeview.Heading", background="#EEF1F4", foreground=MUTED, font=("Segoe UI Semibold", 9), relief="flat")
        style.map("Treeview", background=[("selected", "#DDF4EF")], foreground=[("selected", INK)])

    def _build_ui(self) -> None:
        header = ttk.Frame(self, style="Header.TFrame", height=68)
        header.pack(fill="x")
        header.pack_propagate(False)
        brand = ttk.Frame(header, style="Header.TFrame")
        brand.pack(side="left", padx=22, pady=11)
        ttk.Label(brand, text="MATCHROOM", style="Title.TLabel").pack(anchor="w")
        ttk.Label(brand, text="Review object pairs · regenerate trusted crops", style="Subtitle.TLabel").pack(anchor="w")
        self.header_file = ttk.Label(header, text="Select an image", style="Subtitle.TLabel")
        self.header_file.pack(side="right", padx=24)

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True)
        self._build_sidebar(body)
        self._build_workspace(body)
        self._build_inspector(body)

        footer = ttk.Frame(self, style="Panel.TFrame", height=48)
        footer.pack(fill="x", side="bottom")
        footer.pack_propagate(False)
        self.status_dot = tk.Canvas(footer, width=20, height=20, bg=PANEL, highlightthickness=0)
        self.status_dot.pack(side="left", padx=(20, 4))
        self.status_dot.create_oval(6, 6, 14, 14, fill=TEAL, outline="", tags="dot")
        self.status_text = ttk.Label(footer, text="Ready", style="Muted.TLabel")
        self.status_text.pack(side="left")
        ttk.Label(footer, text="Ctrl+S save  ·  Ctrl+Z undo  ·  Delete unpair", style="Muted.TLabel").pack(side="right", padx=20)

    def _build_sidebar(self, parent: ttk.Frame) -> None:
        side = ttk.Frame(parent, style="Panel.TFrame", width=260)
        side.pack(side="left", fill="y", padx=(0, 1))
        side.pack_propagate(False)

        dataset_panel = ttk.Frame(side, style="Panel.TFrame")
        dataset_panel.pack(fill="x", padx=16, pady=(14, 5))
        ttk.Label(dataset_panel, text="DATA ROOT", style="Section.TLabel").pack(side="left")
        ttk.Button(
            dataset_panel,
            text="Browse…",
            style="Quiet.TButton",
            command=self.browse_dataset,
        ).pack(side="right")
        self.dataset_label = ttk.Label(
            side,
            text=str(self.dataset),
            style="Muted.TLabel",
            wraplength=228,
            justify="left",
        )
        self.dataset_label.pack(fill="x", padx=16, pady=(0, 8))

        title = ttk.Frame(side, style="Panel.TFrame")
        title.pack(fill="x", padx=16, pady=(5, 9))
        ttk.Label(title, text="IMAGE QUEUE", style="Section.TLabel").pack(side="left")
        self.queue_count = ttk.Label(title, text="0", style="Count.TLabel")
        self.queue_count.pack(side="right")
        self.search_var = tk.StringVar()
        search = ttk.Entry(side, textvariable=self.search_var, font=("Segoe UI", 10))
        search.pack(fill="x", padx=16, pady=(0, 10), ipady=6)
        search.insert(0, "")
        self.search_var.trace_add("write", lambda *_: self._filter_images())
        list_frame = ttk.Frame(side, style="Panel.TFrame")
        list_frame.pack(fill="both", expand=True, padx=(10, 4), pady=(0, 10))
        self.image_list = tk.Listbox(
            list_frame,
            bg=PANEL,
            fg=INK,
            selectbackground="#DDF4EF",
            selectforeground=INK,
            activestyle="none",
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            font=("Segoe UI", 10),
            exportselection=False,
        )
        scroll = ttk.Scrollbar(list_frame, command=self.image_list.yview)
        self.image_list.configure(yscrollcommand=scroll.set)
        self.image_list.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.image_list.bind("<<ListboxSelect>>", self._queue_select)
        self._refresh_queue()

    def _build_workspace(self, parent: ttk.Frame) -> None:
        workspace = ttk.Frame(parent)
        workspace.pack(side="left", fill="both", expand=True, padx=14, pady=14)
        toolbar = ttk.Frame(workspace, style="Panel.TFrame")
        toolbar.pack(fill="x", pady=(0, 10))
        ttk.Label(toolbar, text="MATCH CANVAS", style="Section.TLabel").pack(side="left", padx=14, pady=11)
        self.summary_label = ttk.Label(toolbar, text="—", style="Muted.TLabel")
        self.summary_label.pack(side="left", padx=4)
        ttk.Button(toolbar, text="↶ Undo", style="Quiet.TButton", command=self.undo).pack(side="right", padx=(2, 10), pady=6)
        ttk.Button(toolbar, text="↷ Redo", style="Quiet.TButton", command=self.redo).pack(side="right", padx=2, pady=6)
        ttk.Button(toolbar, text="Auto-match", style="Quiet.TButton", command=self.reset_automatic).pack(side="right", padx=2, pady=6)

        canvas_frame = ttk.Frame(workspace, style="Panel.TFrame")
        canvas_frame.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(canvas_frame, bg=CANVAS_BG, highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True, padx=8, pady=8)
        self.canvas.bind("<Button-1>", self._canvas_click)
        self.canvas.bind("<Configure>", self._canvas_resized)
        hint = ttk.Frame(workspace, style="Panel.TFrame")
        hint.pack(fill="x", pady=(10, 0))
        self.instruction = ttk.Label(
            hint,
            text="Select an object in either camera, then select its partner.",
            style="Panel.TLabel",
        )
        self.instruction.pack(side="left", padx=14, pady=11)
        self.intersection_badge = ttk.Label(hint, text="0 crossings", style="Muted.TLabel")
        self.intersection_badge.pack(side="right", padx=14)

    def _build_inspector(self, parent: ttk.Frame) -> None:
        panel = ttk.Frame(parent, style="Panel.TFrame", width=340)
        panel.pack(side="right", fill="y")
        panel.pack_propagate(False)
        ttk.Label(panel, text="MATCH INSPECTOR", style="Section.TLabel").pack(anchor="w", padx=16, pady=(17, 3))
        self.selection_text = ttk.Label(
            panel,
            text="Nothing selected",
            style="Muted.TLabel",
            wraplength=300,
            justify="left",
        )
        self.selection_text.pack(anchor="w", padx=16, pady=(0, 12))
        self.preview = tk.Canvas(panel, height=150, bg="#111D27", highlightthickness=0)
        self.preview.pack(fill="x", padx=16)
        self.preview.create_text(150, 75, text="Crop preview", fill="#8393A1", font=("Segoe UI", 10), tags="placeholder")

        match_title = ttk.Frame(panel, style="Panel.TFrame")
        match_title.pack(fill="x", padx=16, pady=(18, 7))
        ttk.Label(match_title, text="CURRENT PAIRS", style="Section.TLabel").pack(side="left")
        self.match_count = ttk.Label(match_title, text="0", style="Count.TLabel")
        self.match_count.pack(side="right")
        tree_frame = ttk.Frame(panel, style="Panel.TFrame")
        tree_frame.pack(fill="both", expand=True, padx=16)
        self.match_tree = ttk.Treeview(tree_frame, columns=("top", "bottom", "error"), show="headings", selectmode="browse")
        self.match_tree.heading("top", text="TOP")
        self.match_tree.heading("bottom", text="BOTTOM")
        self.match_tree.heading("error", text="ERROR")
        self.match_tree.column("top", width=68, anchor="center")
        self.match_tree.column("bottom", width=68, anchor="center")
        self.match_tree.column("error", width=84, anchor="e")
        tree_scroll = ttk.Scrollbar(tree_frame, command=self.match_tree.yview)
        self.match_tree.configure(yscrollcommand=tree_scroll.set)
        self.match_tree.pack(side="left", fill="both", expand=True)
        tree_scroll.pack(side="right", fill="y")
        self.match_tree.bind("<<TreeviewSelect>>", self._tree_select)

        actions = ttk.Frame(panel, style="Panel.TFrame")
        actions.pack(fill="x", padx=16, pady=12)
        ttk.Button(actions, text="Unpair selected", style="Danger.TButton", command=self.unpair_selected).pack(fill="x", pady=(0, 7))
        ttk.Button(actions, text="Save + regenerate crops", style="Primary.TButton", command=self.save).pack(fill="x")

    def _bind_shortcuts(self) -> None:
        self.bind_all("<Control-s>", lambda _: self.save())
        self.bind_all("<Control-z>", lambda _: self.undo())
        self.bind_all("<Control-y>", lambda _: self.redo())
        self.bind_all("<Delete>", lambda _: self.unpair_selected())
        self.bind_all("<Escape>", lambda _: self._clear_selection())

    def browse_dataset(self) -> None:
        selected = filedialog.askdirectory(
            parent=self,
            title="Choose data root (contains images/ and _label folders)",
            initialdir=str(self.dataset.resolve()),
            mustexist=True,
        )
        if not selected:
            return

        dataset = Path(selected)
        image_dir = dataset / "images"
        label_dirs = [
            folder
            for folder in annotation_folders(dataset)
            if folder.name.startswith("_")
        ]
        problems: list[str] = []
        if not image_dir.is_dir():
            problems.append("an images folder")
        if not label_dirs:
            problems.append("at least one label folder whose name starts with '_'")
        if problems:
            messagebox.showerror(
                "Invalid data root",
                "The selected folder must contain " + " and ".join(problems) + ".",
                parent=self,
            )
            return
        if not self._confirm_leave():
            return

        self.dataset = dataset
        self.output_dir = dataset / "matches"
        self.paths = list(image_files(dataset, None))
        self.document = None
        self.selected = None
        self.selected_pair = None
        self.search_var.set("")
        self.filtered_paths = list(self.paths)
        self.dataset_label.configure(text=str(dataset))
        self.header_file.configure(text="Select an image")
        self.canvas.delete("all")
        self.match_tree.delete(*self.match_tree.get_children())
        self.match_count.configure(text="0")
        self.summary_label.configure(text="—")
        self.selection_text.configure(text="Nothing selected")
        self.preview.delete("all")
        self.preview.create_text(
            150,
            75,
            text="Crop preview",
            fill="#8393A1",
            font=("Segoe UI", 10),
            tags="placeholder",
        )
        self._refresh_queue()

        if self.paths:
            self.open_image(self.paths[0])
            self._set_status(
                f"Loaded {len(self.paths)} image(s) and "
                f"{len(label_dirs)} label folder(s)."
            )
        else:
            self._set_status("No BMP images were found in images/.", warning=True)

    def _refresh_queue(self) -> None:
        self.image_list.delete(0, "end")
        for path in self.filtered_paths:
            self.image_list.insert("end", f"  {path.stem}")
        self.queue_count.configure(text=str(len(self.filtered_paths)))

    def _filter_images(self) -> None:
        term = self.search_var.get().strip().lower()
        self.filtered_paths = [path for path in self.paths if term in path.stem.lower()]
        self._refresh_queue()

    def _queue_select(self, _event: tk.Event) -> None:
        selected = self.image_list.curselection()
        if selected:
            self.open_image(self.filtered_paths[selected[0]])

    def _confirm_leave(self) -> bool:
        if not self.document or not self.document.dirty:
            return True
        answer = messagebox.askyesnocancel(
            "Unsaved match edits",
            "Save the current matches and regenerate their crops before leaving?",
            parent=self,
        )
        if answer is None:
            return False
        if answer:
            return self.save()
        return True

    def open_image(self, path: Path) -> None:
        if self.document and path == self.document.image_path:
            return
        if not self._confirm_leave():
            return
        self._set_status(f"Loading {path.name}…")
        self.update_idletasks()
        try:
            document = MatchDocument(self.dataset, path, self.merge_iou)
            width, height, rows = read_bmp_rows(path)
        except Exception as exc:
            messagebox.showerror("Could not open image", str(exc), parent=self)
            self._set_status("Image could not be opened.", error=True)
            return
        self.document = document
        self.rows = rows
        self.source_size = (width, height)
        self.selected = None
        self.selected_pair = None
        self.header_file.configure(text=path.name)
        self._redraw()
        self._set_status("Ready to review. Click one object in each camera to pair them.")

    def _canvas_resized(self, _event: tk.Event) -> None:
        if self._resize_job:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(80, self._redraw)

    def _redraw(self) -> None:
        self._resize_job = None
        self.canvas.delete("all")
        document = self.document
        if not document or not self.rows:
            self.canvas.create_text(
                max(20, self.canvas.winfo_width() / 2),
                max(20, self.canvas.winfo_height() / 2),
                text="Choose an image to begin",
                fill="#81909E",
                font=("Segoe UI", 14),
            )
            return
        margin = 26
        max_width = max(100, self.canvas.winfo_width() - margin * 2)
        max_height = max(100, self.canvas.winfo_height() - margin * 2)
        self.photo, self.image_scale = scaled_photo(
            self.rows, document.width, document.height, max_width, max_height
        )
        draw_w = int(document.width * self.image_scale)
        draw_h = int(document.height * self.image_scale)
        origin_x = (self.canvas.winfo_width() - draw_w) / 2
        origin_y = (self.canvas.winfo_height() - draw_h) / 2
        self.image_origin = (origin_x, origin_y)
        self.canvas.create_image(origin_x, origin_y, image=self.photo, anchor="nw")
        split_y = origin_y + document.height / 2 * self.image_scale
        self.canvas.create_line(origin_x, split_y, origin_x + draw_w, split_y, fill="#FFFFFF", stipple="gray50", width=1)
        self.canvas.create_text(origin_x + 10, origin_y + 10, text="TOP CAMERA", fill="white", anchor="nw", font=("Segoe UI Semibold", 9))
        self.canvas.create_text(origin_x + 10, split_y + 10, text="BOTTOM CAMERA", fill="white", anchor="nw", font=("Segoe UI Semibold", 9))
        for index, (top, bottom) in enumerate(document.pairs):
            color = HEX_COLORS[index % len(HEX_COLORS)]
            x1, y1 = self._screen(top.center_x, top.center_y)
            x2, y2 = self._screen(bottom.center_x, bottom.center_y)
            width = 4 if self.selected_pair == index else 2
            self.canvas.create_line(x1, y1, x2, y2, fill=color, width=width, tags=(f"pair-{index}",))
        matched_top = {top.id for top, _ in document.pairs}
        matched_bottom = {bottom.id for _, bottom in document.pairs}
        for side, objects, matched in (
            ("top", document.top, matched_top),
            ("bottom", document.bottom, matched_bottom),
        ):
            for item in objects:
                is_selected = self.selected and self.selected[0] == side and self.selected[1].id == item.id
                color = AMBER if item.id not in matched else "#E8EEF3"
                if is_selected:
                    color = TEAL
                left, top_y = self._screen(item.bbox[0], item.bbox[1])
                right, bottom_y = self._screen(item.bbox[2], item.bbox[3])
                self.canvas.create_rectangle(left, top_y, right, bottom_y, outline=color, width=3 if is_selected else 1)
                cx, cy = self._screen(item.center_x, item.center_y)
                radius = 10 if is_selected else 7
                self.canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, fill=color, outline=CANVAS_BG, width=2)
                self.canvas.create_text(cx, cy, text=str(item.id), fill=NAVY, font=("Segoe UI Semibold", 7))
        self._refresh_inspector()

    def _screen(self, x: float, y: float) -> tuple[float, float]:
        return (
            self.image_origin[0] + x * self.image_scale,
            self.image_origin[1] + y * self.image_scale,
        )

    def _source(self, x: float, y: float) -> tuple[float, float]:
        return (
            (x - self.image_origin[0]) / max(self.image_scale, 0.001),
            (y - self.image_origin[1]) / max(self.image_scale, 0.001),
        )

    def _hit_object(self, x: float, y: float) -> tuple[str, Object] | None:
        if not self.document:
            return None
        sx, sy = self._source(x, y)
        side = "top" if sy < self.document.height / 2 else "bottom"
        objects = self.document.top if side == "top" else self.document.bottom
        inside = [item for item in objects if item.bbox[0] - 8 <= sx <= item.bbox[2] + 8 and item.bbox[1] - 8 <= sy <= item.bbox[3] + 8]
        candidates = inside or objects
        if not candidates:
            return None
        closest = min(candidates, key=lambda item: math.hypot(sx - item.center_x, sy - item.center_y))
        limit = 48 / max(self.image_scale, 0.1)
        if not inside and math.hypot(sx - closest.center_x, sy - closest.center_y) > limit:
            return None
        return side, closest

    def _canvas_click(self, event: tk.Event) -> None:
        hit = self._hit_object(event.x, event.y)
        if not hit or not self.document:
            self._clear_selection()
            return
        side, item = hit
        if self.selected is None or self.selected[0] == side:
            self.selected = hit
            self.selected_pair = None
            partner = self.document.partner(side, item.id)
            action = "Choose a bottom-camera partner." if side == "top" else "Choose a top-camera partner."
            if partner:
                action += f" Currently paired with object {partner.id}."
            self.instruction.configure(text=action)
            self._redraw()
            return
        first_side, first_item = self.selected
        top = first_item if first_side == "top" else item
        bottom = item if side == "bottom" else first_item
        changed = self.document.pair(top, bottom)
        self.selected = None
        self.selected_pair = next(
            (index for index, pair in enumerate(self.document.pairs) if pair[0].id == top.id and pair[1].id == bottom.id),
            None,
        )
        self.instruction.configure(text=f"Paired top {top.id} with bottom {bottom.id}. Select another object to continue.")
        if changed:
            self._set_status("Match changed — save to regenerate crops.", warning=True)
        self._redraw()

    def _tree_select(self, _event: tk.Event) -> None:
        if self._refreshing_tree:
            return
        chosen = self.match_tree.selection()
        if not chosen or not self.document:
            return
        chosen_index = int(chosen[0])
        if self.selected is None and self.selected_pair == chosen_index:
            return
        self.selected_pair = chosen_index
        self.selected = None
        self._redraw()

    def _refresh_inspector(self) -> None:
        document = self.document
        if not document:
            return
        self._refreshing_tree = True
        try:
            self.match_tree.delete(*self.match_tree.get_children())
            for index, (top, bottom) in enumerate(document.pairs):
                error = pair_cost(top, bottom, document.width, document.transform) * document.width
                self.match_tree.insert("", "end", iid=str(index), values=(top.id, bottom.id, f"{error:.1f}px"))
            if self.selected_pair is not None and self.selected_pair < len(document.pairs):
                self.match_tree.selection_set(str(self.selected_pair))
        finally:
            self._refreshing_tree = False
        if self.selected_pair is not None and self.selected_pair < len(document.pairs):
            top, bottom = document.pairs[self.selected_pair]
            error = pair_cost(top, bottom, document.width, document.transform) * document.width
            self.selection_text.configure(text=f"Top {top.id}  →  Bottom {bottom.id}\nCalibration error: {error:.1f} px")
            self._draw_crop_preview(top, bottom)
        elif self.selected:
            side, item = self.selected
            partner = document.partner(side, item.id)
            partner_text = f"paired with {partner.id}" if partner else "currently unmatched"
            self.selection_text.configure(text=f"{side.title()} object {item.id} · {partner_text}\nChoose an object in the other camera.")
            if side == "top":
                self._draw_crop_preview(item, partner)
            else:
                self._draw_crop_preview(partner, item)
        else:
            self.selection_text.configure(text="Nothing selected\nClick a marker or choose a row below.")
            self.preview.delete("all")
            self.preview.create_text(150, 75, text="Select a pair to preview its crop", fill="#8393A1", font=("Segoe UI", 9))
        unmatched = len(document.unmatched("top")) + len(document.unmatched("bottom"))
        self.match_count.configure(text=str(len(document.pairs)))
        self.summary_label.configure(text=f"{len(document.pairs)} pairs · {unmatched} unmatched")
        crossings = count_intersections(document.pairs)
        self.intersection_badge.configure(text=f"{crossings} crossing{'s' if crossings != 1 else ''}")

    def _draw_crop_preview(self, top: Object | None, bottom: Object | None) -> None:
        self.preview.delete("all")
        if not top or not bottom or not self.rows or not self.document:
            self.preview.create_text(150, 75, text="Complete the pair to preview its crop", fill="#8393A1", font=("Segoe UI", 9))
            return
        object_width = max(top.bbox[2] - top.bbox[0], bottom.bbox[2] - bottom.bbox[0])
        object_height = max(top.bbox[3] - top.bbox[1], bottom.bbox[3] - bottom.bbox[1])
        crop_w = max(1, int(math.ceil(object_width)) + 24)
        crop_h = max(1, int(math.ceil(object_height)) + 24)

        def crop(item: Object) -> list[bytes]:
            left = int(round(item.center_x - crop_w / 2))
            top_y = int(round(item.center_y - crop_h / 2))
            result = []
            black = b"\x00\x00\x00"
            for y in range(crop_h):
                row = bytearray()
                sy = top_y + y
                for x in range(crop_w):
                    sx = left + x
                    row.extend(self.rows[sy][sx * 3 : sx * 3 + 3] if 0 <= sy < self.document.height and 0 <= sx < self.document.width else black)
                result.append(bytes(row))
            return result

        top_rows, bottom_rows = crop(top), crop(bottom)
        combined = [a + b"\xff\xff\xff" * 4 + b for a, b in zip(top_rows, bottom_rows)]
        self.preview_photo, _ = scaled_photo(combined, crop_w * 2 + 4, crop_h, 292, 132)
        self.preview.create_image(150, 75, image=self.preview_photo)
        self.preview.create_text(8, 8, text="TOP", fill="white", anchor="nw", font=("Segoe UI Semibold", 8))
        self.preview.create_text(292, 8, text="BOTTOM", fill="white", anchor="ne", font=("Segoe UI Semibold", 8))

    def _clear_selection(self) -> None:
        self.selected = None
        self.selected_pair = None
        self.instruction.configure(text="Select an object in either camera, then select its partner.")
        self._redraw()

    def unpair_selected(self) -> None:
        if not self.document:
            return
        changed = False
        if self.selected_pair is not None and self.selected_pair < len(self.document.pairs):
            top, bottom = self.document.pairs[self.selected_pair]
            changed = self.document.unpair(top.id, bottom.id)
        elif self.selected:
            side, item = self.selected
            changed = self.document.unpair(top_id=item.id if side == "top" else None, bottom_id=item.id if side == "bottom" else None)
        if changed:
            self.selected = None
            self.selected_pair = None
            self._set_status("Pair removed — save to regenerate crops.", warning=True)
            self._redraw()

    def reset_automatic(self) -> None:
        if self.document:
            self.document.reset_automatic()
            self.selected = None
            self.selected_pair = None
            self.instruction.configure(text="Automatic matches restored. Review any uncertain pairs before saving.")
            self._set_status("Automatic matches restored — save to regenerate crops.", warning=True)
            self._redraw()

    def undo(self) -> None:
        if self.document and self.document.undo():
            self.selected = None
            self.selected_pair = None
            self._set_status("Last edit undone.", warning=True)
            self._redraw()

    def redo(self) -> None:
        if self.document and self.document.redo():
            self.selected = None
            self.selected_pair = None
            self._set_status("Edit restored.", warning=True)
            self._redraw()

    def save(self) -> bool:
        if not self.document:
            return False
        self._set_status("Saving reviewed matches and regenerating crops…")
        self.update_idletasks()
        try:
            result = self.document.save(self.output_dir, self.visual_dir, self.crop_dir)
        except Exception as exc:
            messagebox.showerror("Could not save", str(exc), parent=self)
            self._set_status("Save failed; previous match JSON was preserved.", error=True)
            return False
        self._set_status(f"Saved {len(result['matches'])} matches and regenerated {len(result['matches'])} crops.")
        self.instruction.configure(text="Saved. Continue reviewing or choose another image.")
        return True

    def _set_status(self, text: str, *, warning: bool = False, error: bool = False) -> None:
        color = RED if error else AMBER if warning else TEAL
        self.status_dot.itemconfigure("dot", fill=color)
        self.status_text.configure(text=text)

    def _close(self) -> None:
        if self._confirm_leave():
            self.destroy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("images_dastchin_goo"))
    parser.add_argument("--visual-output", type=Path, default=Path("output"))
    parser.add_argument("--crop-output", type=Path, default=Path("crops"))
    parser.add_argument("--merge-iou", type=float, default=0.5)
    args = parser.parse_args()
    app = MatchEditor(args.dataset, args.visual_output, args.crop_output, args.merge_iou)
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
