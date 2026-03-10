#!/usr/bin/env python3
"""Compare SST-climatology and SST-anomaly experiments with Iris + Matplotlib.

Creates 3-panel figures for each requested diagnostic:
1) climatology
2) anomaly experiment
3) anomaly - climatology

Supports:
- lat-lon maps
- zonal-mean latitude-sigma cross sections
- time selection: single date, range of dates, or time mean over date range

# Single date, u/v/t, both latlon + zonal
python SpeedyWeather/diagscripts/plot_sst_experiment_comparison.py 
  --clim-file SpeedyWeather/experiments/run_primitive_wet_t63_l20_sst_anomaly_0002/output.nc 
  --anom-file SpeedyWeather/experiments/run_primitive_wet_t63_l20_sst_anomaly_0001/output.nc 
  --variables u v t 
  --time-mode single --date 2000-01-12T00:00:00 
  --plot-style contourf 
  --output-dir SpeedyWeather/experiments/figures/figures_sst_compare

# Range of dates, one figure per time
python SpeedyWeather/diagscripts/plot_sst_experiment_comparison.py 
  --clim-file <clim.nc> --anom-file <anom.nc> 
  --variables t 
  --time-mode range --start-date 2000-01-12 --end-date 2000-01-15 
  --plot-style pcolormesh

# Time mean over range, latlon only, plot sigma layer 20 for 3D maps
python SpeedyWeather/experiments/plot_sst_experiment_comparison.py 
  --clim-file <clim.nc> --anom-file <anom.nc> 
  --variables u t 
  --time-mode mean --start-date 2000-01-12 --end-date 2000-01-20 
  --plot-types latlon 
  --sigma-index 20 
  --plot-style contour

"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import List, Sequence, Tuple

import iris
import matplotlib.pyplot as plt
import numpy as np


LAT_NAMES = ("latitude", "lat")
LON_NAMES = ("longitude", "lon")
TIME_NAMES = ("time",)
LAYER_NAMES = ("layer", "sigma", "sigma_layer", "lev", "level")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot climatology vs anomaly experiment diagnostics (and differences)."
    )
    parser.add_argument("--clim-file", required=True, help="NetCDF from SST climatology experiment.")
    parser.add_argument("--anom-file", required=True, help="NetCDF from SST anomaly experiment.")
    parser.add_argument(
        "--variables",
        required=True,
        nargs="+",
        help="Variable names to plot, e.g. u v t sst mslp",
    )
    parser.add_argument(
        "--time-mode",
        choices=("single", "range", "mean"),
        default="single",
        help="single: nearest date, range: plot each selected time, mean: mean over selected range",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Date/time for --time-mode single (e.g. 2000-01-11 or 2000-01-11T12:00:00).",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Start date/time for --time-mode range/mean.",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="End date/time for --time-mode range/mean.",
    )
    parser.add_argument(
        "--plot-style",
        choices=("contour", "pcolormesh", "contourf"),
        default="contourf",
        help="contour: line contours; pcolormesh: color shading; contourf: filled + black contours",
    )
    parser.add_argument(
        "--plot-types",
        nargs="+",
        choices=("latlon", "zonal"),
        default=("latlon", "zonal"),
        help="Which plots to produce.",
    )
    parser.add_argument(
        "--sigma-index",
        type=int,
        default=None,
        help="1-based sigma layer index for lat-lon maps of 3D variables. "
        "Default: bottom-most sigma layer.",
    )
    parser.add_argument("--contour-levels", type=int, default=18, help="Number of contour levels.")
    parser.add_argument("--dpi", type=int, default=140, help="PNG dpi.")
    parser.add_argument(
        "--output-dir",
        default="SpeedyWeather/experiments/figures_sst_compare",
        help="Directory for PNG output.",
    )
    return parser.parse_args()


def parse_datetime(text: str) -> datetime:
    formats = (
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
    )
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    raise ValueError(f"Could not parse datetime '{text}'.")


def to_py_datetime(value) -> datetime:
    return datetime(
        value.year,
        value.month,
        value.day,
        getattr(value, "hour", 0),
        getattr(value, "minute", 0),
        getattr(value, "second", 0),
    )


def get_coord(cube: iris.cube.Cube, names: Sequence[str], axis: str | None = None):
    if axis is not None:
        coords = cube.coords(axis=axis)
        if coords:
            return coords[0]

    for name in names:
        try:
            return cube.coord(name)
        except Exception:
            continue

    for coord in cube.coords():
        label = (coord.var_name or coord.name() or "").lower()
        if label in names:
            return coord

    wanted = ", ".join(names)
    raise ValueError(f"Could not find coordinate ({wanted}) in cube '{cube.var_name or cube.name()}'.")


def coord_dim(cube: iris.cube.Cube, coord) -> int:
    dims = cube.coord_dims(coord)
    if len(dims) != 1:
        raise ValueError(f"Coordinate '{coord.name()}' is not 1D.")
    return dims[0]


def load_var_cube(path: str, varname: str) -> iris.cube.Cube:
    constraint = iris.Constraint(cube_func=lambda c: (c.var_name == varname) or (c.name() == varname))
    cubes = iris.load(path, constraint)
    if not cubes:
        raise ValueError(f"Variable '{varname}' not found in {path}.")
    if len(cubes) > 1:
        exact = [c for c in cubes if c.var_name == varname]
        if exact:
            return exact[0]
    return cubes[0]


def common_time_indices(
    cube_clim: iris.cube.Cube,
    cube_anom: iris.cube.Cube,
) -> Tuple[List[datetime], dict, dict, int, int]:
    tcoord_clim = get_coord(cube_clim, TIME_NAMES, axis="T")
    tcoord_anom = get_coord(cube_anom, TIME_NAMES, axis="T")
    tdim_clim = coord_dim(cube_clim, tcoord_clim)
    tdim_anom = coord_dim(cube_anom, tcoord_anom)

    tvals_clim = [to_py_datetime(v) for v in tcoord_clim.units.num2date(tcoord_clim.points)]
    tvals_anom = [to_py_datetime(v) for v in tcoord_anom.units.num2date(tcoord_anom.points)]
    idx_clim = {t: i for i, t in enumerate(tvals_clim)}
    idx_anom = {t: i for i, t in enumerate(tvals_anom)}

    common = sorted(set(idx_clim).intersection(idx_anom))
    if not common:
        raise ValueError("No overlapping times between climatology and anomaly files.")
    return common, idx_clim, idx_anom, tdim_clim, tdim_anom


def pick_times(common_times: Sequence[datetime], args: argparse.Namespace) -> List[datetime]:
    if args.time_mode == "single":
        if args.date is None:
            raise ValueError("--date is required when --time-mode single.")
        target = parse_datetime(args.date)
        nearest = min(common_times, key=lambda t: abs(t - target))
        return [nearest]

    if args.start_date is None or args.end_date is None:
        raise ValueError("--start-date and --end-date are required for --time-mode range/mean.")

    start = parse_datetime(args.start_date)
    end = parse_datetime(args.end_date)
    if end < start:
        raise ValueError("--end-date must be >= --start-date.")

    chosen = [t for t in common_times if start <= t <= end]
    if not chosen:
        raise ValueError("No common times fall inside the requested date range.")
    return chosen


def take_along_axis(data: np.ndarray, indices: Sequence[int], axis: int) -> np.ndarray:
    return np.take(data, indices, axis=axis)


def remove_axis_index(axis: int, removed_axis: int) -> int:
    return axis - 1 if axis > removed_axis else axis


def sanitize_label(text: str) -> str:
    return text.replace(":", "").replace("-", "").replace(" ", "_")


def draw_field(
    ax: plt.Axes,
    x2d: np.ndarray,
    y2d: np.ndarray,
    field2d: np.ndarray,
    style: str,
    cmap: str,
    vmin: float,
    vmax: float,
    nlevels: int,
):
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0e-12

    if style == "pcolormesh":
        mappable = ax.pcolormesh(x2d, y2d, field2d, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        return mappable

    levels = np.linspace(vmin, vmax, nlevels)
    if style == "contour":
        mappable = ax.contour(x2d, y2d, field2d, levels=levels, cmap=cmap, linewidths=0.9)
        return mappable

    mappable = ax.contourf(x2d, y2d, field2d, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.contour(x2d, y2d, field2d, levels=levels, colors="k", linewidths=0.35)
    return mappable


def finite_minmax(*arrays: np.ndarray) -> Tuple[float, float]:
    vals = []
    for arr in arrays:
        data = np.asarray(np.ma.filled(arr, np.nan))
        good = data[np.isfinite(data)]
        if good.size:
            vals.append(good)
    if not vals:
        return -1.0, 1.0
    merged = np.concatenate(vals)
    return float(np.min(merged)), float(np.max(merged))


def select_sigma_index(layer_values: np.ndarray, sigma_index_1based: int | None) -> int:
    if sigma_index_1based is None:
        return int(np.nanargmax(layer_values))

    idx = sigma_index_1based - 1
    if idx < 0 or idx >= len(layer_values):
        raise ValueError(f"--sigma-index must be within 1..{len(layer_values)} for this variable.")
    return idx


def to_lat_lon_map(
    field_no_time: np.ndarray,
    lat_dim: int,
    lon_dim: int,
    layer_dim: int | None,
    sigma_idx: int | None,
) -> np.ndarray:
    arr = field_no_time
    if layer_dim is not None:
        if sigma_idx is None:
            raise ValueError("Internal error: sigma_idx not set for layered variable.")
        arr = np.take(arr, sigma_idx, axis=layer_dim)
    return np.moveaxis(arr, (lat_dim, lon_dim), (0, 1))


def to_zonal_cross_section(
    field_no_time: np.ndarray,
    lat_dim: int,
    lon_dim: int,
    layer_dim: int,
) -> np.ndarray:
    arr = np.moveaxis(field_no_time, (layer_dim, lat_dim, lon_dim), (0, 1, 2))
    return np.ma.mean(arr, axis=2)


def plot_three_panel_latlon(
    var: str,
    units: str,
    timestamp_label: str,
    lat: np.ndarray,
    lon: np.ndarray,
    clim_field: np.ndarray,
    anom_field: np.ndarray,
    diff_field: np.ndarray,
    style: str,
    levels: int,
    out_png: str,
    dpi: int,
    sigma_value: float | None = None,
):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    lon2d, lat2d = np.meshgrid(lon, lat)

    main_min, main_max = finite_minmax(clim_field, anom_field)
    diff_abs = max(abs(finite_minmax(diff_field)[0]), abs(finite_minmax(diff_field)[1]))
    diff_abs = diff_abs if diff_abs > 0 else 1.0e-12

    panels = [
        ("Climatology", clim_field, "viridis", main_min, main_max),
        ("Anomaly", anom_field, "viridis", main_min, main_max),
        ("Anomaly - Climatology", diff_field, "RdBu_r", -diff_abs, diff_abs),
    ]

    for ax, (title, field, cmap, vmin, vmax) in zip(axes, panels):
        m = draw_field(ax, lon2d, lat2d, field, style, cmap, vmin, vmax, levels)
        ax.set_title(title)
        ax.set_xlabel("Longitude (degE)")
        ax.set_ylabel("Latitude (degN)")
        cb = fig.colorbar(m, ax=ax, shrink=0.86, pad=0.03)
        cb.set_label(units)

    level_text = "" if sigma_value is None else f", sigma={sigma_value:.3f}"
    fig.suptitle(f"{var} [{units}] at {timestamp_label}{level_text}", fontsize=12)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def plot_three_panel_zonal(
    var: str,
    units: str,
    timestamp_label: str,
    lat: np.ndarray,
    sigma: np.ndarray,
    clim_xsec: np.ndarray,
    anom_xsec: np.ndarray,
    diff_xsec: np.ndarray,
    style: str,
    levels: int,
    out_png: str,
    dpi: int,
):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    lat2d, sigma2d = np.meshgrid(lat, sigma)

    main_min, main_max = finite_minmax(clim_xsec, anom_xsec)
    diff_abs = max(abs(finite_minmax(diff_xsec)[0]), abs(finite_minmax(diff_xsec)[1]))
    diff_abs = diff_abs if diff_abs > 0 else 1.0e-12

    panels = [
        ("Climatology", clim_xsec, "viridis", main_min, main_max),
        ("Anomaly", anom_xsec, "viridis", main_min, main_max),
        ("Anomaly - Climatology", diff_xsec, "RdBu_r", -diff_abs, diff_abs),
    ]

    for ax, (title, field, cmap, vmin, vmax) in zip(axes, panels):
        m = draw_field(ax, lat2d, sigma2d, field, style, cmap, vmin, vmax, levels)
        ax.set_title(title)
        ax.set_xlabel("Latitude (degN)")
        ax.set_ylabel("Sigma")
        ax.set_ylim(1.0, 0.0)
        cb = fig.colorbar(m, ax=ax, shrink=0.86, pad=0.03)
        cb.set_label(units)

    fig.suptitle(f"{var} zonal mean [{units}] at {timestamp_label}", fontsize=12)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def run_for_variable(var: str, args: argparse.Namespace) -> None:
    cube_clim = load_var_cube(args.clim_file, var)
    cube_anom = load_var_cube(args.anom_file, var)

    lat_coord = get_coord(cube_clim, LAT_NAMES, axis="Y")
    lon_coord = get_coord(cube_clim, LON_NAMES, axis="X")
    time_coord = get_coord(cube_clim, TIME_NAMES, axis="T")
    lat_dim0 = coord_dim(cube_clim, lat_coord)
    lon_dim0 = coord_dim(cube_clim, lon_coord)
    time_dim0 = coord_dim(cube_clim, time_coord)

    try:
        layer_coord = get_coord(cube_clim, LAYER_NAMES, axis="Z")
        layer_dim0 = coord_dim(cube_clim, layer_coord)
        layer_values = np.asarray(layer_coord.points, dtype=float)
    except Exception:
        layer_coord = None
        layer_dim0 = None
        layer_values = None

    # Basic structural consistency checks between the two experiments.
    lat_coord_anom = get_coord(cube_anom, LAT_NAMES, axis="Y")
    lon_coord_anom = get_coord(cube_anom, LON_NAMES, axis="X")
    lat_anom = np.asarray(lat_coord_anom.points, dtype=float)
    lon_anom = np.asarray(lon_coord_anom.points, dtype=float)
    lat_clim = np.asarray(lat_coord.points, dtype=float)
    lon_clim = np.asarray(lon_coord.points, dtype=float)
    if lat_clim.shape != lat_anom.shape or not np.allclose(lat_clim, lat_anom):
        raise ValueError(f"Latitude coordinates differ between files for '{var}'.")
    if lon_clim.shape != lon_anom.shape or not np.allclose(lon_clim, lon_anom):
        raise ValueError(f"Longitude coordinates differ between files for '{var}'.")

    common_times, idx_clim, idx_anom, tdim_clim, tdim_anom = common_time_indices(cube_clim, cube_anom)
    if tdim_clim != time_dim0 or tdim_anom != time_dim0:
        raise ValueError(f"Unexpected time dimension mismatch for '{var}'.")

    selected_times = pick_times(common_times, args)

    os.makedirs(args.output_dir, exist_ok=True)
    units = str(cube_clim.units)
    lat_vals = np.asarray(lat_coord.points, dtype=float)
    lon_vals = np.asarray(lon_coord.points, dtype=float)

    clim_data = np.ma.array(cube_clim.data)
    anom_data = np.ma.array(cube_anom.data)

    if args.time_mode == "mean":
        indices_clim = [idx_clim[t] for t in selected_times]
        indices_anom = [idx_anom[t] for t in selected_times]

        clim_sel = take_along_axis(clim_data, indices_clim, axis=time_dim0)
        anom_sel = take_along_axis(anom_data, indices_anom, axis=time_dim0)

        clim_no_time = np.ma.mean(clim_sel, axis=time_dim0)
        anom_no_time = np.ma.mean(anom_sel, axis=time_dim0)

        label_time = (
            f"mean_{selected_times[0].strftime('%Y-%m-%dT%H%M')}"
            f"_to_{selected_times[-1].strftime('%Y-%m-%dT%H%M')}"
        )
        time_title = (
            f"time-mean {selected_times[0].strftime('%Y-%m-%d %H:%M')} "
            f"to {selected_times[-1].strftime('%Y-%m-%d %H:%M')}"
        )

        produce_plots_for_time_slice(
            var,
            units,
            time_title,
            label_time,
            lat_vals,
            lon_vals,
            layer_values,
            clim_no_time,
            anom_no_time,
            time_dim0,
            lat_dim0,
            lon_dim0,
            layer_dim0,
            args,
        )
        return

    for t in selected_times:
        i_clim = idx_clim[t]
        i_anom = idx_anom[t]
        clim_no_time = np.take(clim_data, i_clim, axis=time_dim0)
        anom_no_time = np.take(anom_data, i_anom, axis=time_dim0)

        label_time = t.strftime("%Y%m%dT%H%M")
        time_title = t.strftime("%Y-%m-%d %H:%M")
        produce_plots_for_time_slice(
            var,
            units,
            time_title,
            label_time,
            lat_vals,
            lon_vals,
            layer_values,
            clim_no_time,
            anom_no_time,
            time_dim0,
            lat_dim0,
            lon_dim0,
            layer_dim0,
            args,
        )


def produce_plots_for_time_slice(
    var: str,
    units: str,
    time_title: str,
    label_time: str,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    layer_values: np.ndarray | None,
    clim_no_time: np.ndarray,
    anom_no_time: np.ndarray,
    time_dim0: int,
    lat_dim0: int,
    lon_dim0: int,
    layer_dim0: int | None,
    args: argparse.Namespace,
) -> None:
    lat_dim = remove_axis_index(lat_dim0, time_dim0)
    lon_dim = remove_axis_index(lon_dim0, time_dim0)
    layer_dim = None if layer_dim0 is None else remove_axis_index(layer_dim0, time_dim0)

    n_dims_no_time = clim_no_time.ndim
    if n_dims_no_time < 2:
        raise ValueError(f"Variable '{var}' has <2 dimensions after time selection; cannot plot maps.")

    if layer_dim is None and n_dims_no_time > 2:
        raise ValueError(
            f"Variable '{var}' has {n_dims_no_time} dimensions after time selection without a sigma/layer "
            "coordinate. This script supports 2D (lat,lon) or layered 3D fields."
        )

    diff_no_time = anom_no_time - clim_no_time

    if "latlon" in args.plot_types:
        sigma_idx = None
        sigma_value = None
        if layer_dim is not None and layer_values is not None:
            sigma_idx = select_sigma_index(layer_values, args.sigma_index)
            sigma_value = float(layer_values[sigma_idx])

        clim_map = to_lat_lon_map(clim_no_time, lat_dim, lon_dim, layer_dim, sigma_idx)
        anom_map = to_lat_lon_map(anom_no_time, lat_dim, lon_dim, layer_dim, sigma_idx)
        diff_map = to_lat_lon_map(diff_no_time, lat_dim, lon_dim, layer_dim, sigma_idx)

        out_png = os.path.join(args.output_dir, f"{var}_latlon_{sanitize_label(label_time)}.png")
        plot_three_panel_latlon(
            var=var,
            units=units,
            timestamp_label=time_title,
            lat=lat_vals,
            lon=lon_vals,
            clim_field=clim_map,
            anom_field=anom_map,
            diff_field=diff_map,
            style=args.plot_style,
            levels=args.contour_levels,
            out_png=out_png,
            dpi=args.dpi,
            sigma_value=sigma_value,
        )
        print(f"Wrote {out_png}")

    if "zonal" in args.plot_types:
        if layer_dim is None or layer_values is None:
            print(f"Skipping zonal cross-section for '{var}' (no sigma/layer dimension).")
            return

        clim_xsec = to_zonal_cross_section(clim_no_time, lat_dim, lon_dim, layer_dim)
        anom_xsec = to_zonal_cross_section(anom_no_time, lat_dim, lon_dim, layer_dim)
        diff_xsec = to_zonal_cross_section(diff_no_time, lat_dim, lon_dim, layer_dim)

        out_png = os.path.join(args.output_dir, f"{var}_zonal_{sanitize_label(label_time)}.png")
        plot_three_panel_zonal(
            var=var,
            units=units,
            timestamp_label=time_title,
            lat=lat_vals,
            sigma=np.asarray(layer_values, dtype=float),
            clim_xsec=clim_xsec,
            anom_xsec=anom_xsec,
            diff_xsec=diff_xsec,
            style=args.plot_style,
            levels=args.contour_levels,
            out_png=out_png,
            dpi=args.dpi,
        )
        print(f"Wrote {out_png}")


def main() -> None:
    args = parse_args()
    for var in args.variables:
        print(f"Processing {var} ...")
        run_for_variable(var, args)


if __name__ == "__main__":
    main()
