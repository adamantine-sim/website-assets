#!/usr/bin/env python3
"""Render Adamantine VTU output as the demo_316 website images.

The renderer accepts an Adamantine ``.pvd`` file, or one ``.pvtu``/``.vtu``
file, and writes PNG snapshots.  The defaults reproduce the view used by the
demo_316 images: an X-Y top view, the ``temperature`` field, the logarithmic
``plasma`` colour map over 300--450 K, black cell edges, and a 1024 x 838
canvas.

Examples
--------
Render the first six time points from an Adamantine run::

    python scripts/render_demo_316.py output.pvd \
        --output-dir /path/to/website-assets/examples/demo_316

Render every time point instead::

    python scripts/render_demo_316.py output.pvd --all

Render a single ``.pvtu`` file::

    python scripts/render_demo_316.py output.30.pvtu --prefix demo_316

PyVista and VTK are the only runtime dependencies::

    python -m pip install pyvista
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    import pyvista as pv
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
    from vtkmodules.vtkRenderingCore import (
        vtkActor2D,
        vtkCoordinate,
        vtkPolyDataMapper2D,
    )
except ImportError as exc:  # pragma: no cover - exercised by the CLI itself
    raise SystemExit(
        "PyVista is required. Install it with: python -m pip install pyvista"
    ) from exc


DEFAULT_WINDOW_SIZE = (1024, 838)
DEFAULT_CLIM = (300.0, 450.0)
DEFAULT_CAMERA_SCALE = 0.717
DEFAULT_CAMERA_TILT_DEGREES = 10.0
DEFAULT_FRAME_COUNT = 6
DEFAULT_SCALAR_BAR_WIDTH = 0.105
DEFAULT_SCALAR_BAR_HEIGHT = 0.167
DEFAULT_SCALAR_BAR_POSITION = (0.049, 0.640)
DEFAULT_ANNOTATION_FONT_SIZE = 10
DEFAULT_SCALAR_BAR_FONT_SIZE = 18
DEFAULT_AXES_FONT_SIZE = 10


def _reader_for(filename: Path):
    """Return a PyVista reader, with a useful error for unsupported inputs."""

    suffix = filename.suffix.lower()
    if suffix not in {".pvd", ".pvtu", ".vtu"}:
        raise ValueError(
            f"Unsupported input type {filename.suffix!r}; expected .pvd, .pvtu, or .vtu"
        )
    return pv.get_reader(filename)


def _read_as_mesh(reader) -> pv.DataSet:
    """Read a frame and merge PVD/PVTU blocks into one dataset."""

    data = reader.read()
    if not isinstance(data, pv.MultiBlock):
        return data

    blocks = [block for block in data if block is not None and block.n_cells > 0]
    if not blocks:
        raise ValueError("The selected time point contains no cells")
    if len(blocks) == 1:
        return blocks[0]
    return pv.merge(blocks, merge_points=False)


def _time_indices(reader, requested: Sequence[int] | None, render_all: bool) -> list[int]:
    """Resolve CLI time-point selection for a PVD reader."""

    if not isinstance(reader, pv.PVDReader):
        if requested not in (None, [], [0]):
            raise ValueError("--time-index can only select frames from a .pvd input")
        return [0]

    count = reader.number_time_points
    if requested:
        indices = list(requested)
    elif render_all:
        indices = list(range(count))
    else:
        indices = list(range(min(DEFAULT_FRAME_COUNT, count)))

    invalid = [index for index in indices if index < 0 or index >= count]
    if invalid:
        raise IndexError(
            f"Time-point index {invalid[0]} is outside the available range 0..{count - 1}"
        )
    return indices


def _field_range(mesh: pv.DataSet, field: str) -> tuple[float, float]:
    """Return the finite min/max of a scalar field before thresholding."""

    if field in mesh.point_data:
        values = np.asarray(mesh.point_data[field])
    elif field in mesh.cell_data:
        values = np.asarray(mesh.cell_data[field])
    else:
        available = sorted(set(mesh.point_data) | set(mesh.cell_data))
        raise KeyError(f"Field {field!r} was not found. Available fields: {available}")

    if values.ndim != 1:
        raise ValueError(f"Field {field!r} is not a scalar array")
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError(f"Field {field!r} contains no finite values")
    return float(finite.min()), float(finite.max())


def _threshold(mesh: pv.DataSet, field: str, lower_bound: float | None) -> pv.DataSet:
    """Apply the VisIt-style lower threshold used by the documentation."""

    if lower_bound is None:
        return mesh

    preference = "cell" if field in mesh.cell_data else "point"
    filtered = mesh.threshold(
        value=lower_bound,
        scalars=field,
        preference=preference,
        method="upper",
        all_scalars=False,
    )
    if filtered.n_cells == 0:
        raise ValueError(
            f"Threshold {field} >= {lower_bound:g} removed every cell in the selected frame"
        )
    return filtered


def _add_display_lines(
    plotter: pv.Plotter,
    segments: Sequence[tuple[tuple[float, float], tuple[float, float]]],
) -> None:
    """Add black line segments in screenshot/display coordinates.

    ``vtkAxesActor`` is intentionally not used here.  Its orientation widget
    is sized relative to the viewport and is much smaller than the simple
    VisIt triad used by the reference images.
    """

    points = vtkPoints()
    lines = vtkCellArray()
    for start, end in segments:
        first = points.InsertNextPoint(start[0], start[1], 0.0)
        second = points.InsertNextPoint(end[0], end[1], 0.0)
        lines.InsertNextCell(2)
        lines.InsertCellPoint(first)
        lines.InsertCellPoint(second)

    poly_data = vtkPolyData()
    poly_data.SetPoints(points)
    poly_data.SetLines(lines)

    coordinate = vtkCoordinate()
    coordinate.SetCoordinateSystemToDisplay()
    mapper = vtkPolyDataMapper2D()
    mapper.SetInputData(poly_data)
    mapper.SetTransformCoordinate(coordinate)

    actor = vtkActor2D()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.0, 0.0, 0.0)
    actor.GetProperty().SetLineWidth(1.0)
    plotter.renderer.AddViewProp(actor)


def _set_black_axes(
    plotter: pv.Plotter,
    window_size: tuple[int, int],
) -> None:
    """Add the simple black XYZ orientation marker used by the reference view."""

    width, height = window_size
    x_scale = width / DEFAULT_WINDOW_SIZE[0]
    y_scale = height / DEFAULT_WINDOW_SIZE[1]

    def display_point(x: float, y_from_top: float) -> tuple[float, float]:
        return x * x_scale, height - y_from_top * y_scale

    # The VisIt triad has two one-pixel axes.  The Z direction points toward
    # the viewer in this X-Y view, so only its label is visible.
    _add_display_lines(
        plotter,
        [
            (display_point(102, 754), display_point(183, 754)),
            (display_point(102, 753), display_point(102, 687)),
        ],
    )

    label_size = max(1, round(DEFAULT_AXES_FONT_SIZE * y_scale))
    plotter.add_text(
        "Y",
        position=display_point(103, 686),
        font_size=label_size,
        color="black",
        font="arial",
        shadow=False,
        name="axis-y",
    )
    plotter.add_text(
        "Z",
        position=display_point(112, 761),
        font_size=label_size,
        color="black",
        font="arial",
        shadow=False,
        name="axis-z",
    )
    plotter.add_text(
        "X",
        position=display_point(193, 761),
        font_size=label_size,
        color="black",
        font="arial",
        shadow=False,
        name="axis-x",
    )


def _add_annotations(
    plotter: pv.Plotter,
    minimum: float,
    maximum: float,
    window_size: tuple[int, int],
    font_size: int,
) -> None:
    """Place the title and min/max text to the left of the scalar bar."""

    width, height = window_size
    del width  # Positions are intentionally tied to the reference layout.
    text_style = dict(font_size=font_size, color="black", font="arial", shadow=False)

    # add_text uses display pixels with the origin at the lower left.
    # VisIt's displayed maximum is truncated to one decimal place in these
    # images (for example, 523.3504 is shown as 523.3).
    displayed_maximum = f"{np.trunc(maximum * 10.0) / 10.0:.1f}"

    plotter.add_text(
        "Temperature (K)",
        position=(51, height - 151),
        name="temperature-title",
        **text_style,
    )
    plotter.add_text(
        f"Max:  {displayed_maximum}",
        position=(51, height - 327),
        name="temperature-max",
        **text_style,
    )
    plotter.add_text(
        f"Min:  {minimum:.3f}",
        position=(51, height - 349),
        name="temperature-min",
        **text_style,
    )


def _configure_camera(
    plotter: pv.Plotter,
    mesh: pv.DataSet,
    camera_scale: float,
    tilt_degrees: float = 0.0,
) -> None:
    """Use a stable orthographic camera, optionally tilted away from top view."""

    if not 0.0 <= tilt_degrees < 90.0:
        raise ValueError("Camera tilt must be in the range [0, 90) degrees")

    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    x_extent = xmax - xmin
    y_extent = ymax - ymin
    horizontal_extent = max(x_extent, y_extent)
    if horizontal_extent <= 0:
        raise ValueError("The mesh has no non-zero X-Y extent")

    center = ((xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2)
    plotter.view_xy()
    plotter.enable_parallel_projection()
    plotter.camera.focal_point = center
    # Keep the same camera distance as the top view, but move it toward the
    # negative-Y side when a tilt is requested.  This exposes the Z-facing
    # side faces without changing the mesh coordinates or exaggerating Z.
    camera_distance = zmax + horizontal_extent - center[2]
    tilt = np.deg2rad(tilt_degrees)
    plotter.camera.position = (
        center[0],
        center[1] - camera_distance * np.sin(tilt),
        center[2] + camera_distance * np.cos(tilt),
    )
    plotter.camera.up = (0.0, 1.0, 0.0)
    plotter.camera.parallel_scale = horizontal_extent * camera_scale


def render_frame(
    mesh: pv.DataSet,
    output: Path,
    *,
    field: str = "temperature",
    threshold: float | None = 1.0,
    clim: tuple[float, float] = DEFAULT_CLIM,
    cmap: str = "plasma",
    log_scale: bool = True,
    window_size: tuple[int, int] = DEFAULT_WINDOW_SIZE,
    camera_scale: float = DEFAULT_CAMERA_SCALE,
    camera_tilt_degrees: float = 0.0,
    lighting: bool = False,
    annotation_font_size: int = DEFAULT_ANNOTATION_FONT_SIZE,
    scalar_bar_font_size: int = DEFAULT_SCALAR_BAR_FONT_SIZE,
    show_edges: bool = True,
    show_axes: bool = True,
    show_scalar_bar: bool = True,
) -> tuple[float, float]:
    """Render one mesh and return its unfiltered scalar min/max."""

    minimum, maximum = _field_range(mesh, field)
    display_mesh = _threshold(mesh, field, threshold)
    # The VTU stores each cell with its own copy of the corner points.  Merge
    # coincident points so adjacent retained cells share their interfaces
    # when the mesh is rendered from a tilted view.
    display_mesh = display_mesh.clean()
    if log_scale:
        display_minimum, _ = _field_range(display_mesh, field)
        if display_minimum <= 0:
            raise ValueError(
                f"Logarithmic colour scales require positive {field!r} values "
                "after thresholding; use --linear-scale or raise --threshold"
            )
    output.parent.mkdir(parents=True, exist_ok=True)

    plotter = pv.Plotter(
        off_screen=True,
        window_size=window_size,
        border=False,
    )
    # The reference images use crisp, unsmoothed cell edges and annotations.
    plotter.render_window.SetMultiSamples(0)
    plotter.set_background("white")
    _configure_camera(plotter, display_mesh, camera_scale, camera_tilt_degrees)

    scalar_bar_args = dict(
        title="",
        n_labels=5,
        vertical=True,
        width=DEFAULT_SCALAR_BAR_WIDTH,
        height=DEFAULT_SCALAR_BAR_HEIGHT,
        position_x=DEFAULT_SCALAR_BAR_POSITION[0],
        position_y=DEFAULT_SCALAR_BAR_POSITION[1],
        title_font_size=scalar_bar_font_size,
        label_font_size=scalar_bar_font_size,
        color="black",
        font_family="arial",
        fmt="%.1f",
        outline=False,
    )
    plotter.add_mesh(
        display_mesh,
        scalars=field,
        preference="cell" if field in display_mesh.cell_data else "point",
        cmap=cmap,
        clim=clim,
        show_edges=show_edges,
        edge_color="black",
        line_width=1,
        lighting=lighting,
        log_scale=log_scale,
        scalar_bar_args=scalar_bar_args if show_scalar_bar else None,
        show_scalar_bar=show_scalar_bar,
    )
    plotter.reset_camera_clipping_range()

    if show_scalar_bar:
        _add_annotations(
            plotter,
            minimum,
            maximum,
            window_size,
            annotation_font_size,
        )
    if show_axes:
        _set_black_axes(plotter, window_size)

    plotter.screenshot(str(output), transparent_background=False)
    plotter.close()
    return minimum, maximum


def render_input(
    input_path: Path,
    output_dir: Path,
    prefix: str,
    *,
    indices: Sequence[int] | None = None,
    render_all: bool = False,
    **render_options,
) -> list[Path]:
    """Render the selected frames and return the generated file paths."""

    reader = _reader_for(input_path)
    time_indices = _time_indices(reader, indices, render_all)
    generated: list[Path] = []
    for frame_number, time_index in enumerate(time_indices):
        if isinstance(reader, pv.PVDReader):
            reader.set_active_time_point(time_index)
        mesh = _read_as_mesh(reader)
        output = output_dir / f"{prefix}_{frame_number}.png"
        render_frame(mesh, output, **render_options)
        generated.append(output)
        if isinstance(reader, pv.PVDReader):
            time_value = reader.time_point_value(time_index)
            print(f"wrote {output} (time point {time_index}, t={time_value:g})")
        else:
            print(f"wrote {output}")
    return generated


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Adamantine .pvd, .pvtu, or .vtu file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for PNG files (default: current directory)",
    )
    parser.add_argument("--prefix", default="demo_316", help="PNG filename prefix")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--all",
        action="store_true",
        help="Render every PVD time point instead of the first six",
    )
    selection.add_argument(
        "--time-index",
        nargs="+",
        type=int,
        help="PVD time-point indices to render, in output order",
    )
    parser.add_argument("--field", default="temperature", help="Scalar field to render")
    threshold = parser.add_mutually_exclusive_group()
    threshold.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Keep cells with FIELD >= this value (default: 1.0)",
    )
    threshold.add_argument(
        "--no-threshold",
        action="store_true",
        help="Render the full mesh, including inactive cells",
    )
    parser.add_argument(
        "--clim",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=DEFAULT_CLIM,
        help="Colour scale limits (default: 300 450)",
    )
    parser.add_argument("--cmap", default="plasma", help="Matplotlib/PyVista colour map")
    parser.add_argument(
        "--linear-scale",
        action="store_true",
        help="Use a linear colour scale instead of the logarithmic VisIt scale",
    )
    parser.add_argument(
        "--tilt-camera",
        action="store_true",
        help="Tilt the camera 10 degrees from top view and enable lighting",
    )
    parser.add_argument(
        "--annotation-font-size",
        type=int,
        default=DEFAULT_ANNOTATION_FONT_SIZE,
        help="Font size for the title and Max/Min annotations (default: 10)",
    )
    parser.add_argument(
        "--scalar-bar-font-size",
        type=int,
        default=DEFAULT_SCALAR_BAR_FONT_SIZE,
        help="Font size for scalar-bar labels (default: 18)",
    )
    parser.add_argument(
        "--camera-scale",
        type=float,
        default=DEFAULT_CAMERA_SCALE,
        help="Orthographic view height divided by the mesh width (default: 0.717)",
    )
    parser.add_argument(
        "--window-size",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        default=DEFAULT_WINDOW_SIZE,
        help="PNG dimensions (default: 1024 838)",
    )
    parser.add_argument("--no-edges", action="store_true", help="Hide mesh cell edges")
    parser.add_argument("--no-axes", action="store_true", help="Hide the XYZ orientation marker")
    parser.add_argument("--no-scalar-bar", action="store_true", help="Hide the scalar bar and statistics")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise SystemExit(f"Input file does not exist: {input_path}")

    threshold = None if args.no_threshold else args.threshold
    render_input(
        input_path,
        args.output_dir.expanduser().resolve(),
        args.prefix,
        indices=args.time_index,
        render_all=args.all,
        field=args.field,
        threshold=threshold,
        clim=tuple(args.clim),
        cmap=args.cmap,
        log_scale=not args.linear_scale,
        window_size=tuple(args.window_size),
        camera_scale=args.camera_scale,
        camera_tilt_degrees=(
            DEFAULT_CAMERA_TILT_DEGREES if args.tilt_camera else 0.0
        ),
        lighting=args.tilt_camera,
        annotation_font_size=args.annotation_font_size,
        scalar_bar_font_size=args.scalar_bar_font_size,
        show_edges=not args.no_edges,
        show_axes=not args.no_axes,
        show_scalar_bar=not args.no_scalar_bar,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
