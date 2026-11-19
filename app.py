from __future__ import annotations

import os
import re
import tempfile
import time
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import streamlit as st
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window


APP_TITLE = "Raster Band Stack + Pyramids"
SUPPORTED_EXTENSIONS = {
    ".tif",
    ".tiff",
    ".jp2",
    ".img",
    ".vrt",
    ".dat",
    ".bil",
    ".asc",
}
OVERVIEW_CHOICES = [2, 4, 8, 16, 32]
IO_WINDOW_SIZE = 2048

# Recognizes common names such as *_B2.tif, *_B02.jp2, *_BAND_2.tif,
# and Band3.tif. The band marker is removed to create a scene grouping key.
BAND_TOKEN = re.compile(
    r"(?i)(?P<prefix>^|[^A-Za-z0-9])"
    r"(?:BAND[_\-\s]?0?(?P<word>[234])|B0?(?P<short>[234]))"
    r"(?=$|[^0-9])"
)


def _parse_paths(raw: str) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for line in raw.splitlines():
        value = line.strip().strip('"').strip("'")
        if not value:
            continue
        path = Path(value).expanduser()
        key = str(path.resolve()).casefold()
        if key not in seen:
            paths.append(path)
            seen.add(key)
    return paths


def _extract_band_and_group(path: Path) -> tuple[int, str] | None:
    match = BAND_TOKEN.search(path.stem)
    if not match:
        return None
    band_value = match.group("word") or match.group("short")
    band = int(band_value)
    group = BAND_TOKEN.sub(lambda item: f"{item.group('prefix')}BAND", path.stem, count=1)
    group = re.sub(r"[.\-\s]+", "_", group).strip("_").casefold()
    return band, group


def scan_folders(folder_paths: list[Path], recursive: bool) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    seen_files: set[str] = set()
    for root in folder_paths:
        if not root.is_dir():
            continue
        candidates = root.rglob("*") if recursive else root.iterdir()
        for path in candidates:
            if not path.is_file() or path.suffix.casefold() not in SUPPORTED_EXTENSIONS:
                continue
            resolved_path = path.resolve()
            if (
                resolved_path.stem.casefold().endswith("_b2_b3_b4_stack")
                or resolved_path.stem.casefold() == path.parent.name.casefold()
            ):
                continue
            file_key = str(resolved_path).casefold()
            if file_key in seen_files:
                continue
            seen_files.add(file_key)
            parsed = _extract_band_and_group(path)
            if parsed is None:
                continue
            band, group_name = parsed
            folder = str(path.parent.resolve())
            group_key = (folder.casefold(), group_name)
            record = groups.setdefault(
                group_key,
                {
                    "folder": folder,
                    "group_name": group_name,
                    "bands": {2: [], 3: [], 4: []},
                },
            )
            record["bands"][band].append(str(resolved_path))
    return sorted(groups.values(), key=lambda item: (item["folder"].casefold(), item["group_name"]))


def _display_name(record: dict[str, Any]) -> str:
    group = record["group_name"].replace("_band", "").strip("_")
    return group or Path(record["folder"]).name or "raster"


def _output_filename(record: dict[str, Any]) -> str:
    folder = Path(record["folder"])
    name = folder.name or folder.drive.rstrip("\\/: ") or "raster"
    return f"{name}.tif"


def _folders_with_output_conflicts(records: list[dict[str, Any]]) -> set[str]:
    ready_folder_counts = Counter(
        record["folder"].casefold()
        for record in records
        if _scene_status(record) == "Ready"
    )
    return {folder for folder, count in ready_folder_counts.items() if count > 1}


def _scene_status(record: dict[str, Any]) -> str:
    duplicates = [str(band) for band, paths in record["bands"].items() if len(paths) > 1]
    missing = [str(band) for band, paths in record["bands"].items() if not paths]
    if duplicates:
        return "Ambiguous band " + ", ".join(duplicates)
    if missing:
        return "Missing band " + ", ".join(missing)
    return "Ready"


def _grid_matches(source: rasterio.io.DatasetReader, reference: rasterio.io.DatasetReader) -> bool:
    return (
        source.width == reference.width
        and source.height == reference.height
        and source.crs == reference.crs
        and source.transform.almost_equals(reference.transform)
    )


def _convert_nodata(value: float | None, dtype: np.dtype[Any]) -> int | float | None:
    if value is None:
        return None
    numeric_value = float(value)
    if dtype.kind in "iu":
        limits = np.iinfo(dtype)
        if not np.isfinite(numeric_value) or not numeric_value.is_integer():
            raise ValueError("Integer output types require a finite whole-number NoData value.")
        if numeric_value < limits.min or numeric_value > limits.max:
            raise ValueError(f"NoData value {value} cannot be represented by output type {dtype}.")
        return int(numeric_value)
    converted = dtype.type(numeric_value).item()
    if np.isfinite(numeric_value) and not np.isfinite(converted):
        raise ValueError(f"NoData value {value} cannot be represented by output type {dtype}.")
    return converted


def _stack_scene(
    record: dict[str, Any],
    *,
    resampling_name: str,
    overview_resampling_name: str,
    overview_levels: list[int],
    compression: str,
    worker_threads: str,
    overwrite: bool,
    nodata_override: float | None,
) -> tuple[str, str]:
    folder = Path(record["folder"])
    output_path = folder / _output_filename(record)
    if output_path.exists() and not overwrite:
        return "Skipped", f"Already exists: {output_path}"

    temp_path: Path | None = None
    try:
        with rasterio.Env(GDAL_NUM_THREADS=worker_threads):
            with ExitStack() as stack:
                datasets = {
                    band: stack.enter_context(rasterio.open(record["bands"][band][0]))
                    for band in (2, 3, 4)
                }
                reference = datasets[2]
                if any(dataset.count < 1 for dataset in datasets.values()):
                    raise ValueError("A source raster has no readable first band.")

                output_dtype = np.result_type(*[np.dtype(dataset.dtypes[0]) for dataset in datasets.values()])
                output_nodata = _convert_nodata(
                    nodata_override if nodata_override is not None else reference.nodata,
                    output_dtype,
                )
                grid_matches = {band: _grid_matches(datasets[band], reference) for band in (2, 3, 4)}
                for band in (3, 4):
                    source = datasets[band]
                    if not grid_matches[band] and (source.crs is None or reference.crs is None):
                        raise ValueError(
                            f"Band {band} does not share Band 2's grid and one or both rasters lack a CRS. "
                            "Add matching georeferencing or align these bands before stacking."
                        )

                resampling = getattr(Resampling, resampling_name)
                read_sources: dict[int, Any] = {2: reference}
                for band in (3, 4):
                    if grid_matches[band]:
                        read_sources[band] = datasets[band]
                    else:
                        read_sources[band] = stack.enter_context(
                            WarpedVRT(
                                datasets[band],
                                crs=reference.crs,
                                transform=reference.transform,
                                width=reference.width,
                                height=reference.height,
                                resampling=resampling,
                                nodata=output_nodata,
                                dtype=output_dtype.name,
                            )
                        )

                profile: dict[str, Any] = {
                    "driver": "GTiff",
                    "width": reference.width,
                    "height": reference.height,
                    "count": 3,
                    "dtype": output_dtype.name,
                    "crs": reference.crs,
                    "transform": reference.transform,
                    "nodata": output_nodata,
                    "tiled": True,
                    "blockxsize": 512,
                    "blockysize": 512,
                    "bigtiff": "IF_SAFER",
                }
                if compression != "none":
                    profile["compress"] = compression
                    profile["predictor"] = 3 if output_dtype.kind == "f" else 2
                    profile["num_threads"] = worker_threads

                file_descriptor, temp_name = tempfile.mkstemp(
                    prefix=f".{output_path.stem}_",
                    suffix=".tif",
                    dir=str(folder),
                )
                os.close(file_descriptor)
                temp_path = Path(temp_name)
                with rasterio.open(temp_path, "w", **profile) as destination:
                    for row_offset in range(0, reference.height, IO_WINDOW_SIZE):
                        for column_offset in range(0, reference.width, IO_WINDOW_SIZE):
                            window = Window(
                                column_offset,
                                row_offset,
                                min(IO_WINDOW_SIZE, reference.width - column_offset),
                                min(IO_WINDOW_SIZE, reference.height - row_offset),
                            )
                            stack_array = np.empty(
                                (3, int(window.height), int(window.width)), dtype=output_dtype
                            )
                            for output_index, band in enumerate((2, 3, 4)):
                                read_sources[band].read(
                                    1,
                                    window=window,
                                    out=stack_array[output_index],
                                )
                            destination.write(stack_array, window=window)
                    for index, band in enumerate((2, 3, 4), start=1):
                        destination.set_band_description(index, f"Band {band}")
                    destination.update_tags(
                        source_bands="2,3,4",
                        stack_order="Band 2, Band 3, Band 4",
                        pyramid_resampling=overview_resampling_name,
                    )
                    valid_levels = [
                        level for level in overview_levels
                        if max(reference.width, reference.height) >= level
                    ]
                    if valid_levels:
                        destination.build_overviews(
                            valid_levels,
                            getattr(Resampling, overview_resampling_name),
                        )
                        destination.update_tags(ns="rio_overview", resampling=overview_resampling_name)

        os.replace(temp_path, output_path)
        temp_path = None
        overview_text = ", ".join(map(str, valid_levels)) if valid_levels else "none (image too small)"
        return "Completed", f"Saved to {output_path}; overview levels: {overview_text}"
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _choose_folders() -> list[str]:
    root = None
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        common_parent = filedialog.askdirectory(
            parent=root,
            title="Choose the parent folder containing your source folders",
            mustexist=True,
        )
        if not common_parent:
            root.destroy()
            return []

        parent_path = Path(common_parent)
        child_folders = sorted(
            (path for path in parent_path.iterdir() if path.is_dir()),
            key=lambda path: path.name.casefold(),
        )
        parent_has_rasters = any(
            path.is_file() and path.suffix.casefold() in SUPPORTED_EXTENSIONS
            for path in parent_path.iterdir()
        )
        options = ([parent_path] if parent_has_rasters or not child_folders else []) + child_folders
        if not options:
            options = [parent_path]

        root.deiconify()
        root.title("Select source folders")
        root.geometry("780x560")
        root.minsize(580, 360)
        root.attributes("-topmost", True)
        tk.Label(
            root,
            text="Check every folder to include. All listed folders start selected.",
            anchor="w",
            justify="left",
        ).pack(fill="x", padx=14, pady=(14, 4))
        tk.Label(root, text=f"Common parent: {parent_path}", anchor="w", justify="left").pack(
            fill="x", padx=14, pady=(0, 8)
        )

        list_frame = tk.Frame(root)
        list_frame.pack(fill="both", expand=True, padx=14, pady=4)
        canvas = tk.Canvas(list_frame, highlightthickness=0)
        scrollbar = tk.Scrollbar(list_frame, orient="vertical", command=canvas.yview)
        folder_frame = tk.Frame(canvas)
        folder_frame.bind(
            "<Configure>",
            lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        frame_window = canvas.create_window((0, 0), window=folder_frame, anchor="nw")
        canvas.bind(
            "<Configure>",
            lambda event: canvas.itemconfigure(frame_window, width=event.width),
        )
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        folder_checks: list[tuple[Path, tk.BooleanVar]] = []
        for path in options:
            checked = tk.BooleanVar(master=root, value=True)
            tk.Checkbutton(
                folder_frame,
                text=str(path),
                variable=checked,
                anchor="w",
                justify="left",
                wraplength=690,
            ).pack(fill="x", anchor="w", padx=6, pady=2)
            folder_checks.append((path, checked))

        selected_folders: list[str] = []
        button_frame = tk.Frame(root)
        button_frame.pack(fill="x", padx=14, pady=(8, 14))

        def set_all(value: bool) -> None:
            for _path, checked in folder_checks:
                checked.set(value)

        def use_selection() -> None:
            selected_folders.extend(str(path) for path, checked in folder_checks if checked.get())
            root.destroy()

        def cancel_selection() -> None:
            root.destroy()

        tk.Button(button_frame, text="Select all", command=lambda: set_all(True)).pack(side="left")
        tk.Button(button_frame, text="Clear all", command=lambda: set_all(False)).pack(side="left", padx=(8, 0))
        tk.Button(button_frame, text="Use selected folders", command=use_selection).pack(side="right")
        tk.Button(button_frame, text="Cancel", command=cancel_selection).pack(side="right", padx=(0, 8))
        root.protocol("WM_DELETE_WINDOW", cancel_selection)
        root.mainloop()
        return selected_folders
    except Exception:
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass
        return []


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🛰️", layout="wide")
    st.markdown(
        """
        <style>
          .block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem; }
          [data-testid="stAppViewContainer"] { background: #f5f7fa; }
          [data-testid="stHeader"] { background: rgba(0,0,0,0); }
          .hero { padding: 1.35rem 1.55rem; border-radius: 18px; color: #f7fbff;
                  background: linear-gradient(120deg, #12334a, #176b78); margin-bottom: 1.2rem; }
          .hero h1 { margin: 0 0 .3rem 0; font-size: 2rem; }
          .hero p { margin: 0; color: #d9edf1; font-size: 1rem; }
          .step-card { padding: 1rem 1.1rem; background: white; border: 1px solid #e4eaf0;
                       border-radius: 14px; margin-bottom: .8rem; }
          div.stButton > button[kind="primary"] { background: #126b78; border-color: #126b78; }
        </style>
        <div class="hero">
          <h1>🛰️ Raster Band Stack + Pyramids</h1>
          <p>Batch stack bands 2, 3, and 4 in order, then build GeoTIFF overview pyramids.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "source_folders_text" not in st.session_state:
        st.session_state.source_folders_text = ""
    if "scan_records" not in st.session_state:
        st.session_state.scan_records = []
    if "scan_signature" not in st.session_state:
        st.session_state.scan_signature = None

    st.markdown(
        '<div class="step-card"><b>1. Choose source folders</b><br>'
        'Select multiple folders together, or paste a list of folder paths. Each stack is saved beside its bands using the folder name.</div>',
        unsafe_allow_html=True,
    )
    folder_button, folder_hint = st.columns([1, 4])
    with folder_button:
        if st.button("＋ Choose folders", use_container_width=True):
            selected_folders = _choose_folders()
            if selected_folders:
                existing = st.session_state.source_folders_text.strip()
                st.session_state.source_folders_text = "\n".join(
                    [existing, *selected_folders] if existing else selected_folders
                )
                st.rerun()
    with folder_hint:
        st.caption(
            "Choose a common parent, check all folders to include in one window, or paste multiple full paths below, one per line."
        )
    source_text = st.text_area(
        "Source folders — one path per line",
        key="source_folders_text",
        height=145,
        placeholder="C:\\imagery\\scene_01\nD:\\satellite\\scene_02",
    )
    recursive = st.toggle("Also scan subfolders", value=True, help="When enabled, matching rasters in nested folders are included.")

    st.markdown('<div class="step-card"><b>2. Review the bands and processing options</b></div>', unsafe_allow_html=True)
    with st.expander("Processing options", expanded=True):
        option_columns = st.columns(4)
        with option_columns[0]:
            resampling_name = st.selectbox(
                "Resampling if grids differ",
                ["nearest", "bilinear", "cubic"],
                index=0,
                help="Band 2 defines the output grid. Other bands are aligned to it when necessary.",
            )
        with option_columns[1]:
            overview_resampling_name = st.selectbox(
                "Pyramid resampling",
                ["average", "nearest", "bilinear", "cubic"],
                index=0,
            )
        with option_columns[2]:
            compression = st.selectbox(
                "GeoTIFF compression",
                ["deflate", "lzw", "zstd", "none"],
                index=0,
                help="ZSTD is often faster and smaller; DEFLATE is more widely supported by older GIS software.",
            )
        with option_columns[3]:
            thread_choice = st.selectbox(
                "Processing threads",
                ["All CPU cores", "4", "2", "1"],
                index=0,
                help="Uses GDAL multithreading for raster reads, compression, alignment, and pyramid generation.",
            )
            worker_threads = "ALL_CPUS" if thread_choice == "All CPU cores" else thread_choice
        overview_levels = st.multiselect("Overview pyramid levels", OVERVIEW_CHOICES, default=OVERVIEW_CHOICES)
        nodata_input = st.text_input(
            "Output NoData override (optional)",
            placeholder="Leave blank to use Band 2's NoData value",
            help="Set a value such as -9999 when aligned bands may leave pixels outside the Band 2 footprint.",
        )
        overwrite = st.checkbox("Overwrite an existing folder-name .tif", value=False)

    col_scan, col_spacer = st.columns([1, 4])
    with col_scan:
        scan_clicked = st.button("Scan for band sets", type="primary", use_container_width=True)
    folder_paths = _parse_paths(source_text)
    st.caption(f"{len(folder_paths)} source folder(s) listed. Stacks are saved beside each matched band set.")
    signature = (tuple(str(path.resolve()) for path in folder_paths), recursive)
    if scan_clicked:
        invalid = [str(path) for path in folder_paths if not path.is_dir()]
        if not folder_paths:
            st.error("Add at least one source folder first.")
        elif invalid:
            st.error("These folders could not be found: " + ", ".join(invalid))
        else:
            with st.spinner("Scanning folders for matching band files…"):
                st.session_state.scan_records = scan_folders(folder_paths, recursive)
                st.session_state.scan_signature = signature
            st.rerun()

    records = st.session_state.scan_records
    scan_is_current = st.session_state.scan_signature == signature and bool(records)
    if scan_is_current:
        conflicting_folders = _folders_with_output_conflicts(records)
        ready_records = [
            record
            for record in records
            if _scene_status(record) == "Ready"
            and record["folder"].casefold() not in conflicting_folders
        ]
        rows = []
        for record in records:
            status = _scene_status(record)
            if status == "Ready" and record["folder"].casefold() in conflicting_folders:
                status = "Multiple complete scenes in folder; output names would collide"
            rows.append(
                {
                    "Status": status,
                    "Folder": record["folder"],
                    "Scene": _display_name(record),
                    "Output file": _output_filename(record),
                    "Band 2": Path(record["bands"][2][0]).name if record["bands"][2] else "—",
                    "Band 3": Path(record["bands"][3][0]).name if record["bands"][3] else "—",
                    "Band 4": Path(record["bands"][4][0]).name if record["bands"][4] else "—",
                }
            )
        metric_columns = st.columns(3)
        metric_columns[0].metric("Band sets found", len(records))
        metric_columns[1].metric("Ready to stack", len(ready_records))
        metric_columns[2].metric("Need attention", len(records) - len(ready_records))
        st.dataframe(rows, use_container_width=True, hide_index=True)
        if conflicting_folders:
            st.warning(
                "More than one complete band set was found in at least one folder. "
                "Because the output name must match the folder name, split those scenes into separate folders before processing."
            )
        if not ready_records:
            st.info("No complete, unambiguous Band 2 / Band 3 / Band 4 sets were found.")
        if not overview_levels:
            st.warning("Select at least one overview level to build pyramids.")
        if nodata_input.strip():
            try:
                nodata_override = float(nodata_input.strip())
            except ValueError:
                nodata_override = None
                st.error("The NoData override must be numeric.")
        else:
            nodata_override = None

        process_clicked = st.button(
            "Stack bands and build pyramids",
            type="primary",
            disabled=not ready_records or not overview_levels or (bool(nodata_input.strip()) and nodata_override is None),
            use_container_width=True,
        )
        if process_clicked:
            progress = st.progress(0, text="Starting batch…")
            results_area = st.container()
            summaries: list[tuple[str, str, str]] = []
            for index, record in enumerate(ready_records, start=1):
                scene = _display_name(record)
                scene_started = time.perf_counter()
                progress.progress((index - 1) / len(ready_records), text=f"Processing {scene} ({index}/{len(ready_records)})…")
                try:
                    status, details = _stack_scene(
                        record,
                        resampling_name=resampling_name,
                        overview_resampling_name=overview_resampling_name,
                        overview_levels=overview_levels,
                        compression=compression,
                        worker_threads=worker_threads,
                        overwrite=overwrite,
                        nodata_override=nodata_override,
                    )
                except Exception as error:
                    status, details = "Failed", str(error)
                details += f" ({time.perf_counter() - scene_started:.1f}s)"
                summaries.append((scene, status, details))
                progress.progress(index / len(ready_records), text=f"Finished {scene} ({index}/{len(ready_records)})")
            with results_area:
                st.markdown("### Batch results")
                st.dataframe(
                    [{"Scene": scene, "Result": status, "Details": details} for scene, status, details in summaries],
                    use_container_width=True,
                    hide_index=True,
                )
            completed = sum(status == "Completed" for _, status, _ in summaries)
            failed = sum(status == "Failed" for _, status, _ in summaries)
            skipped = sum(status == "Skipped" for _, status, _ in summaries)
            message = (
                f"Batch finished: {completed} stack(s) created, {skipped} skipped, {failed} failed. "
                "Outputs are beside their source bands."
            )
            if failed:
                st.warning(message)
            else:
                st.success(message)
    elif records and not scan_is_current:
        st.info("The folder list or scan setting changed. Scan again to refresh the band list.")
    elif st.session_state.scan_signature is not None and not records:
        st.info("The scan completed, but no files with recognizable Band 2, 3, or 4 names were found.")

    st.caption("Supported naming examples: *_B2.tif, *_B02.jp2, *_BAND_3.tif, and Band4.tif. Raster grids are aligned to Band 2 when needed.")


if __name__ == "__main__":
    main()

