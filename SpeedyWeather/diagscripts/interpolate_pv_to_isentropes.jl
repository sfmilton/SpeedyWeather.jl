#!/usr/bin/env julia

import Pkg

project_dir = joinpath(@__DIR__, "..")
Pkg.activate(project_dir)
Pkg.instantiate()

using Dates
using NCDatasets
using Printf

const CAIROMAKIE = let
    try
        @eval import CairoMakie
        getfield(Main, :CairoMakie)
    catch
        nothing
    end
end

const DEFAULT_RUN_ID = "primitive_wet_t63_l20_full_physics"
const DEFAULT_THETA_LEVELS_K = [315.0, 350.0]
const DEFAULT_OUTPUT_FILENAME = "output_isentropic_pv.nc"
const DEFAULT_PLOT_SUBDIR = "pv_isentropic_plots"
const KAPPA = 0.2857142857142857          # R_d / c_p
const P0_HPA = 1000.0                     # reference pressure for potential temperature
const OMEGA = 7.292115e-5                 # Earth rotation rate [s^-1]
const GRAVITY = 9.80665                   # [m/s^2]

Base.@kwdef mutable struct Config
    input_path::Union{Nothing, String} = nothing
    run_id::String = DEFAULT_RUN_ID
    output_path::Union{Nothing, String} = nothing
    save_netcdf::Bool = true
    theta_levels_k::Vector{Float64} = copy(DEFAULT_THETA_LEVELS_K)
    plot_theta_k::Union{Nothing, Float64} = nothing
    date_single::Union{Nothing, DateTime} = nothing
    date_start::Union{Nothing, DateTime} = nothing
    date_end::Union{Nothing, DateTime} = nothing
    plot_dir::Union{Nothing, String} = nothing
end

function usage()
    println("""
Usage:
  julia SpeedyWeather/diagscripts/interpolate_pv_to_isentropes.jl [OPTIONS]
  julia SpeedyWeather/diagscripts/interpolate_pv_to_isentropes.jl [input.nc] [output.nc|-] [theta_csv]

Options:
  --input PATH             Input NetCDF file. Defaults to latest run output for --run-id.
  --run-id ID              Run ID used for latest output auto-discovery (default: $(DEFAULT_RUN_ID)).
  --output PATH            Output NetCDF path for interpolated isentropic PV.
  --theta-levels CSV       Isentropic levels for interpolation, e.g. 315,350.
  --save-netcdf            Save interpolated isentropic PV NetCDF (default).
  --no-save-netcdf         Do not save a NetCDF file unless required for plotting.
  --plot-theta LEVEL       Isentropic level (K) to plot from ipv_pvu.
  --date DATETIME          Plot nearest available time to a single datetime.
  --start-date DATETIME    Plot a date range start (inclusive).
  --end-date DATETIME      Plot a date range end (inclusive).
  --plot-dir DIR           Directory for PNG plots (default: <input_dir>/$(DEFAULT_PLOT_SUBDIR)).
  --help, -h               Show this help text.

Note:
  Plotting requires CairoMakie in the active Julia environment.

Datetime formats accepted:
  yyyy-mm-dd
  yyyy-mm-ddTHH:MM
  yyyy-mm-ddTHH:MM:SS
  yyyy-mm-dd HH:MM
  yyyy-mm-dd HH:MM:SS

Examples:
  # Save interpolated PV on 315 and 350 K
  julia SpeedyWeather/diagscripts/interpolate_pv_to_isentropes.jl \\
    --input /path/to/output.nc --output /path/to/output_isentropic_pv.nc --theta-levels 315,350

  # Plot single date at 350 K, without keeping an intermediate NetCDF
  julia SpeedyWeather/diagscripts/interpolate_pv_to_isentropes.jl \\
    --input /path/to/output.nc --no-save-netcdf --plot-theta 350 --date 1980-01-15T00:00:00

  # Plot all dates in a range at 315 K and save NetCDF
  julia SpeedyWeather/diagscripts/interpolate_pv_to_isentropes.jl \\
    --input /path/to/output.nc --save-netcdf --plot-theta 315 \\
    --start-date 1980-01-01 --end-date 1980-01-31
""")
end

function latest_output_path(experiments_dir::AbstractString; run_id::AbstractString = DEFAULT_RUN_ID)
    prefix = "run_$(run_id)_"
    candidates = Tuple{Int, String}[]

    for name in readdir(experiments_dir)
        run_path = joinpath(experiments_dir, name)
        startswith(name, prefix) || continue
        isdir(run_path) || continue
        run_number = tryparse(Int, split(name, "_")[end])
        isnothing(run_number) && continue
        push!(candidates, (run_number, run_path))
    end

    isempty(candidates) && error(
        "No run folder found for id '$(run_id)' in $(experiments_dir). " *
        "Pass an explicit NetCDF path as first argument."
    )

    sort!(candidates, by = first)
    latest_path = last(candidates)[2]
    output_nc = joinpath(latest_path, "output.nc")
    isfile(output_nc) || error("Could not find NetCDF file at $(output_nc).")
    return output_nc
end

function parse_csv_floats(s::AbstractString; name::AbstractString)
    vals = Float64[]
    for token in split(s, ",")
        stripped = strip(token)
        isempty(stripped) && continue
        parsed = tryparse(Float64, stripped)
        isnothing(parsed) && error("Could not parse $(name) value: '$(stripped)'")
        push!(vals, parsed)
    end
    isempty(vals) && error("No valid $(name) values provided.")
    return vals
end

function copy_attrs_as_dict(var)
    d = Dict{String, Any}()
    for key in keys(var.attrib)
        d[string(key)] = var.attrib[key]
    end
    return d
end

function parse_float_arg(arg::AbstractString; name::AbstractString)
    parsed = tryparse(Float64, strip(arg))
    isnothing(parsed) && error("Could not parse $(name): '$(arg)'.")
    return parsed
end

const DATETIME_FORMATS = (
    dateformat"yyyy-mm-ddTHH:MM:SS",
    dateformat"yyyy-mm-ddTHH:MM",
    dateformat"yyyy-mm-dd HH:MM:SS",
    dateformat"yyyy-mm-dd HH:MM",
    dateformat"yyyy-mm-dd",
)

function parse_datetime_arg(arg::AbstractString; name::AbstractString)
    s = strip(arg)
    isempty(s) && error("Empty datetime string for $(name).")
    for fmt in DATETIME_FORMATS
        dt = try
            DateTime(s, fmt)
        catch
            nothing
        end
        !isnothing(dt) && return dt
    end
    error("Could not parse $(name): '$(arg)'.")
end

function parse_args(args::Vector{String})
    cfg = Config()
    positional = String[]
    i = 1

    while i <= length(args)
        arg = args[i]
        if arg == "--help" || arg == "-h"
            usage()
            exit(0)
        elseif arg == "--input"
            i < length(args) || error("Option $(arg) requires a value.")
            cfg.input_path = args[i + 1]
            i += 2
        elseif arg == "--run-id"
            i < length(args) || error("Option $(arg) requires a value.")
            cfg.run_id = args[i + 1]
            i += 2
        elseif arg == "--output"
            i < length(args) || error("Option $(arg) requires a value.")
            cfg.output_path = args[i + 1]
            i += 2
        elseif arg == "--theta-levels"
            i < length(args) || error("Option $(arg) requires a CSV list.")
            cfg.theta_levels_k = parse_csv_floats(args[i + 1], name = "theta level")
            i += 2
        elseif arg == "--plot-theta"
            i < length(args) || error("Option $(arg) requires a value.")
            cfg.plot_theta_k = parse_float_arg(args[i + 1], name = "plot theta level")
            i += 2
        elseif arg == "--date"
            i < length(args) || error("Option $(arg) requires a datetime.")
            cfg.date_single = parse_datetime_arg(args[i + 1], name = "--date")
            i += 2
        elseif arg == "--start-date"
            i < length(args) || error("Option $(arg) requires a datetime.")
            cfg.date_start = parse_datetime_arg(args[i + 1], name = "--start-date")
            i += 2
        elseif arg == "--end-date"
            i < length(args) || error("Option $(arg) requires a datetime.")
            cfg.date_end = parse_datetime_arg(args[i + 1], name = "--end-date")
            i += 2
        elseif arg == "--plot-dir"
            i < length(args) || error("Option $(arg) requires a path.")
            cfg.plot_dir = args[i + 1]
            i += 2
        elseif arg == "--save-netcdf"
            cfg.save_netcdf = true
            i += 1
        elseif arg == "--no-save-netcdf"
            cfg.save_netcdf = false
            i += 1
        elseif startswith(arg, "--")
            error("Unknown option: $(arg). Use --help for usage.")
        else
            push!(positional, arg)
            i += 1
        end
    end

    if length(positional) >= 1 && isnothing(cfg.input_path)
        cfg.input_path = positional[1] == "-" ? nothing : positional[1]
    end
    if length(positional) >= 2 && isnothing(cfg.output_path)
        cfg.output_path = positional[2] == "-" ? nothing : positional[2]
    end
    if length(positional) >= 3 && cfg.theta_levels_k == DEFAULT_THETA_LEVELS_K
        cfg.theta_levels_k = parse_csv_floats(positional[3], name = "theta level")
    end
    length(positional) <= 3 || error("Too many positional arguments. Use --help.")

    if !isnothing(cfg.output_path)
        cfg.save_netcdf = true
    end

    isempty(cfg.theta_levels_k) && error("At least one theta level must be specified.")

    if !isnothing(cfg.date_single) && (!isnothing(cfg.date_start) || !isnothing(cfg.date_end))
        error("Use either --date OR (--start-date and --end-date), not both.")
    end
    if xor(isnothing(cfg.date_start), isnothing(cfg.date_end))
        error("Both --start-date and --end-date must be set for date-range plotting.")
    end
    if !isnothing(cfg.date_start) && !isnothing(cfg.date_end) && cfg.date_end < cfg.date_start
        error("--end-date must be >= --start-date.")
    end

    if !cfg.save_netcdf &&
       isnothing(cfg.plot_theta_k) &&
       isnothing(cfg.date_single) &&
       isnothing(cfg.date_start) &&
       isnothing(cfg.date_end)
        error("No outputs requested. Use --save-netcdf and/or plotting options.")
    end

    return cfg
end

function sigma_order(sigma::AbstractVector{Float64})
    if issorted(sigma)
        return collect(1:length(sigma))
    elseif issorted(sigma; rev = true)
        return collect(length(sigma):-1:1)
    else
        return sortperm(sigma)
    end
end

function select_surface_pressure_var(ds_in::NCDataset)
    if haskey(ds_in, "pres")
        return "pres"
    elseif haskey(ds_in, "mslp")
        @warn "Variable 'pres' not found. Falling back to 'mslp' as a pressure proxy."
        return "mslp"
    else
        error("Input file needs 'pres' (preferred) or 'mslp' to infer full-level pressure.")
    end
end

@inline function find_theta_bracket(theta_col::Vector{Float64}, theta_target::Float64)
    n = length(theta_col)
    @inbounds for k in 1:n-1
        theta1 = theta_col[k]
        theta2 = theta_col[k + 1]
        if !isfinite(theta1) || !isfinite(theta2) || theta1 == theta2
            continue
        end
        in_interval = (theta1 <= theta_target <= theta2) || (theta2 <= theta_target <= theta1)
        in_interval && return k
    end
    return 0
end

function interpolate_pv_to_isentropes!(
        ds_out::NCDataset,
        ds_in::NCDataset,
        sigma::Vector{Float64},
        theta_levels_k::Vector{Float64},
        surface_pressure_var::AbstractString,
    )
    for req in ("temp", "vor")
        haskey(ds_in, req) || error("Input file does not contain required variable '$(req)'.")
    end
    haskey(ds_in, surface_pressure_var) || error("Input file does not contain '$(surface_pressure_var)'.")

    temp_raw = ds_in["temp"]
    vor_raw = ds_in["vor"]
    sz = size(temp_raw)
    length(sz) == 4 || error("Variable 'temp' must have dims (lon,lat,layer,time), got size=$(sz).")
    nlon, nlat, nlayer, ntime = sz
    size(vor_raw) == sz || error("Variable 'vor' must have the same shape as 'temp'.")
    nlayer == length(sigma) || error("Layer mismatch: variable has $(nlayer), sigma has $(length(sigma)).")

    order = sigma_order(sigma)
    sigma_sorted = sigma[order]
    ntheta = length(theta_levels_k)

    lat = Float64.(ds_in["lat"][:])
    coriolis = @. 2 * OMEGA * sind(lat)

    p_col = Vector{Float64}(undef, nlayer)   # Pa
    theta_col = Vector{Float64}(undef, nlayer)   # K
    zeta_col = Vector{Float64}(undef, nlayer)   # s^-1

    pv_slice = fill(Float32(NaN), nlon, nlat, ntheta)
    pv_pvu_slice = fill(Float32(NaN), nlon, nlat, ntheta)
    p_theta_slice = fill(Float32(NaN), nlon, nlat, ntheta)

    for it in 1:ntime
        pres_t = Float64.(coalesce.(ds_in[surface_pressure_var][:, :, it], NaN))  # hPa
        temp_t = Float64.(coalesce.(temp_raw[:, :, :, it], NaN))                   # degC
        vor_t = Float64.(coalesce.(vor_raw[:, :, :, it], NaN))                     # s^-1

        fill!(pv_slice, Float32(NaN))
        fill!(pv_pvu_slice, Float32(NaN))
        fill!(p_theta_slice, Float32(NaN))

        @inbounds for j in 1:nlat
            f = coriolis[j]
            for i in 1:nlon
                ps_hpa = pres_t[i, j]
                if !isfinite(ps_hpa) || ps_hpa <= 0
                    continue
                end

                for k in 1:nlayer
                    ks = order[k]
                    p_hpa = sigma_sorted[k] * ps_hpa
                    t_c = temp_t[i, j, ks]
                    zeta = vor_t[i, j, ks]

                    if !isfinite(p_hpa) || p_hpa <= 0 || !isfinite(t_c) || !isfinite(zeta)
                        p_col[k] = NaN
                        theta_col[k] = NaN
                        zeta_col[k] = NaN
                        continue
                    end

                    p_col[k] = 100 * p_hpa                  # Pa
                    t_k = t_c + 273.15
                    theta_col[k] = t_k * (P0_HPA / p_hpa)^KAPPA
                    zeta_col[k] = zeta
                end

                for n in 1:ntheta
                    theta_target = theta_levels_k[n]
                    k = find_theta_bracket(theta_col, theta_target)
                    k == 0 && continue

                    k2 = k + 1
                    theta1 = theta_col[k]
                    theta2 = theta_col[k2]
                    p1 = p_col[k]
                    p2 = p_col[k2]
                    zeta1 = zeta_col[k]
                    zeta2 = zeta_col[k2]

                    if !isfinite(theta1) || !isfinite(theta2) || !isfinite(p1) || !isfinite(p2) || p1 == p2
                        continue
                    end

                    alpha = (theta_target - theta1) / (theta2 - theta1)
                    p_interp = muladd(alpha, p2 - p1, p1)       # Pa
                    zeta_interp = muladd(alpha, zeta2 - zeta1, zeta1)       # s^-1
                    dthetadp = (theta2 - theta1) / (p2 - p1)            # K/Pa

                    pv = -GRAVITY * (zeta_interp + f) * dthetadp   # K m^2 kg^-1 s^-1
                    pv_slice[i, j, n] = Float32(pv)
                    pv_pvu_slice[i, j, n] = Float32(1e6 * pv)
                    p_theta_slice[i, j, n] = Float32(p_interp / 100)  # hPa
                end
            end
        end

        ds_out["ipv"][:, :, :, it] = pv_slice
        ds_out["ipv_pvu"][:, :, :, it] = pv_pvu_slice
        ds_out["p_on_theta"][:, :, :, it] = p_theta_slice
        @printf("  wrote time %d/%d\n", it, ntime)
    end

    return nothing
end

function has_isentropic_pv_fields(ds::NCDataset)
    return all(name -> haskey(ds, name), ("lon", "lat", "theta_level", "time", "ipv_pvu"))
end

function dataset_has_isentropic_pv(input_path::AbstractString)
    ds = NCDataset(input_path, "r")
    try
        return has_isentropic_pv_fields(ds)
    finally
        close(ds)
    end
end

function needs_plotting(cfg::Config)
    return !isnothing(cfg.plot_theta_k) ||
           !isnothing(cfg.date_single) ||
           !isnothing(cfg.date_start) ||
           !isnothing(cfg.date_end)
end

function default_output_path(input_path::AbstractString)
    return joinpath(dirname(input_path), DEFAULT_OUTPUT_FILENAME)
end

function ensure_cairomakie_loaded()
    isnothing(CAIROMAKIE) || return CAIROMAKIE
    error(
        "Plotting requested but CairoMakie is not installed in the active Julia environment.\n" *
        "Install it with: julia -e 'import Pkg; Pkg.activate(\"SpeedyWeather\"); Pkg.add(\"CairoMakie\")'"
    )
end

function as_datetime_vector(times_raw)
    out = DateTime[]
    sizehint!(out, length(times_raw))
    for t in times_raw
        if t isa DateTime
            push!(out, t)
        else
            converted = try
                DateTime(t)
            catch
                nothing
            end
            isnothing(converted) && error(
                "Could not decode 'time' coordinate to DateTime. " *
                "Found element type $(typeof(t))."
            )
            push!(out, converted)
        end
    end
    return out
end

function nearest_time_index(times::Vector{DateTime}, target::DateTime)
    isempty(times) && error("No times available in dataset.")
    diffs = [abs(Dates.value(t - target)) for t in times]
    return argmin(diffs)
end

function select_time_indices_for_plot(
        times::Vector{DateTime};
        date_single::Union{Nothing, DateTime} = nothing,
        date_start::Union{Nothing, DateTime} = nothing,
        date_end::Union{Nothing, DateTime} = nothing,
    )
    if !isnothing(date_single)
        idx = nearest_time_index(times, date_single)
        return [idx]
    elseif !isnothing(date_start) && !isnothing(date_end)
        idx = findall(t -> (t >= date_start) && (t <= date_end), times)
        isempty(idx) && error("No dataset times fall in [$(date_start), $(date_end)].")
        return idx
    else
        return collect(eachindex(times))
    end
end

function finite_absmax(A::AbstractArray{<:Real})
    vals = A[isfinite.(A)]
    isempty(vals) && return 0.0
    return maximum(abs, vals)
end

theta_tag(theta_k::Real) = replace(@sprintf("%.1f", theta_k), "." => "p")
format_time_label(t::DateTime) = Dates.format(t, "yyyy-mm-dd HH:MM") * " UTC"

function plot_isentropic_pv_maps(
        pv_dataset_path::AbstractString;
        plot_dir::AbstractString,
        plot_theta_k::Union{Nothing, Float64} = nothing,
        date_single::Union{Nothing, DateTime} = nothing,
        date_start::Union{Nothing, DateTime} = nothing,
        date_end::Union{Nothing, DateTime} = nothing,
    )
    cm = ensure_cairomakie_loaded()
    mkpath(plot_dir)

    ds = NCDataset(pv_dataset_path, "r")
    try
        has_isentropic_pv_fields(ds) || error(
            "Dataset does not contain required isentropic PV fields for plotting. " *
            "Expected at least lon, lat, theta_level, time, ipv_pvu."
        )

        lon_raw = Float64.(ds["lon"][:])
        lat_raw = Float64.(ds["lat"][:])
        theta_levels = Float64.(ds["theta_level"][:])
        times = as_datetime_vector(ds["time"][:])
        ipv_pvu = ds["ipv_pvu"]

        isempty(theta_levels) && error("theta_level coordinate is empty.")
        theta_target = isnothing(plot_theta_k) ? theta_levels[1] : plot_theta_k
        i_theta = argmin(abs.(theta_levels .- theta_target))
        theta_selected = theta_levels[i_theta]

        if !isnothing(plot_theta_k) && !isapprox(theta_selected, plot_theta_k; atol = 1e-6)
            println(
                "Requested plot theta $(plot_theta_k) K not present; " *
                "using nearest available $(theta_selected) K."
            )
        end

        time_indices = select_time_indices_for_plot(
            times;
            date_single,
            date_start,
            date_end,
        )

        if !isnothing(date_single)
            requested = date_single
            chosen = times[only(time_indices)]
            println("Single-date mode: requested $(requested), plotting nearest $(chosen).")
        end

        lon_order = sortperm(lon_raw)
        lat_order = sortperm(lat_raw)
        lon = lon_raw[lon_order]
        lat = lat_raw[lat_order]

        absmax = 0.0
        for it in time_indices
            field = Float64.(coalesce.(ipv_pvu[:, :, i_theta, it], NaN))
            field = field[lon_order, lat_order]
            absmax = max(absmax, finite_absmax(field))
        end
        absmax = absmax == 0 ? 1.0 : absmax

        for it in time_indices
            valid_time = times[it]
            time_tag = Dates.format(valid_time, "yyyymmddTHHMMSS")
            field = Float64.(coalesce.(ipv_pvu[:, :, i_theta, it], NaN))
            field = field[lon_order, lat_order]

            fig = cm.Figure(size = (1120, 680), fontsize = 20)
            ax = cm.Axis(
                fig[1, 1],
                title = "Isentropic Ertel PV at $(theta_selected) K, $(format_time_label(valid_time))",
                xlabel = "Longitude [deg]",
                ylabel = "Latitude [deg]",
            )
            hm = cm.heatmap!(
                ax,
                lon,
                lat,
                field;
                colormap = :balance,
                colorrange = (-absmax, absmax),
            )
            cm.Colorbar(fig[1, 2], hm, label = "PVU")

            outfile = joinpath(plot_dir, "ipv_pvu_theta$(theta_tag(theta_selected))K_$(time_tag).png")
            cm.save(outfile, fig)
            println("Wrote $(outfile)")
        end
    finally
        close(ds)
    end

    return nothing
end

function resolve_input_path(cfg::Config)
    if !isnothing(cfg.input_path)
        isfile(cfg.input_path) || error("Input file not found: $(cfg.input_path)")
        return cfg.input_path
    end

    experiments_dir = joinpath(@__DIR__, "..", "experiments")
    return latest_output_path(experiments_dir; run_id = cfg.run_id)
end

function create_output_dataset(
        input_path::AbstractString,
        output_path::AbstractString,
        theta_levels_k::Vector{Float64},
    )
    isfile(output_path) && error("Output file already exists: $(output_path)")

    ds_in = NCDataset(input_path, "r")
    try
        for req in ("lon", "lat", "layer", "time", "temp", "vor")
            haskey(ds_in, req) || error("Input file is missing required variable '$(req)'.")
        end
        surface_pressure_var = select_surface_pressure_var(ds_in)
        println("Using surface pressure variable: $(surface_pressure_var)")

        lon = Float64.(ds_in["lon"][:])
        lat = Float64.(ds_in["lat"][:])
        sigma = Float64.(ds_in["layer"][:])
        time_raw = Float64.(ds_in["time"].var[:])
        ntime = length(time_raw)

        ds_out = NCDataset(output_path, "c")
        try
            defDim(ds_out, "lon", length(lon))
            defDim(ds_out, "lat", length(lat))
            defDim(ds_out, "theta_level", length(theta_levels_k))
            defDim(ds_out, "time", ntime)

            lon_attrs = copy_attrs_as_dict(ds_in["lon"])
            lat_attrs = copy_attrs_as_dict(ds_in["lat"])
            time_attrs = copy_attrs_as_dict(ds_in["time"])

            defVar(ds_out, "lon", lon, ("lon",), attrib = lon_attrs)
            defVar(ds_out, "lat", lat, ("lat",), attrib = lat_attrs)
            defVar(ds_out, "theta_level", theta_levels_k, ("theta_level",), attrib = Dict(
                "units" => "K",
                "long_name" => "potential temperature level",
                "standard_name" => "air_potential_temperature",
            ))
            defVar(ds_out, "time", Float64, ("time",), attrib = time_attrs)
            ds_out["time"][:] = time_raw

            defVar(ds_out, "ipv", Float32, ("lon", "lat", "theta_level", "time"), attrib = Dict(
                "units" => "K m^2 kg^-1 s^-1",
                "long_name" => "isentropic Ertel potential vorticity",
                "coordinates" => "lon lat theta_level time",
            ))
            defVar(ds_out, "ipv_pvu", Float32, ("lon", "lat", "theta_level", "time"), attrib = Dict(
                "units" => "PVU",
                "long_name" => "isentropic Ertel potential vorticity",
                "coordinates" => "lon lat theta_level time",
            ))
            defVar(ds_out, "p_on_theta", Float32, ("lon", "lat", "theta_level", "time"), attrib = Dict(
                "units" => "hPa",
                "long_name" => "pressure on potential temperature surfaces",
                "coordinates" => "lon lat theta_level time",
            ))

            interpolate_pv_to_isentropes!(
                ds_out,
                ds_in,
                sigma,
                theta_levels_k,
                surface_pressure_var,
            )
        finally
            close(ds_out)
        end
    finally
        close(ds_in)
    end

    return nothing
end

function main(args::Vector{String} = ARGS)
    cfg = parse_args(args)
    resolved_input = resolve_input_path(cfg)
    plot_requested = needs_plotting(cfg)

    pv_dataset_path = resolved_input
    cleanup_temp_pv_file = false

    try
        if dataset_has_isentropic_pv(resolved_input)
            println("Input already contains isentropic PV diagnostics; reusing it directly.")
            if cfg.save_netcdf && !isnothing(cfg.output_path) && cfg.output_path != resolved_input
                error(
                    "Input already has isentropic PV fields. " *
                    "Refusing to overwrite/copy automatically to $(cfg.output_path)."
                )
            end
        else
            target_output = if cfg.save_netcdf
                isnothing(cfg.output_path) ? default_output_path(resolved_input) : cfg.output_path
            else
                tempname() * ".nc"
            end

            println("Input:  $(resolved_input)")
            println("Output: $(target_output)")
            println("Theta levels [K]: $(join(cfg.theta_levels_k, ", "))")
            create_output_dataset(resolved_input, target_output, cfg.theta_levels_k)
            println("Interpolated isentropic PV written: $(target_output)")

            pv_dataset_path = target_output
            cleanup_temp_pv_file = !cfg.save_netcdf
        end

        if plot_requested
            plot_dir = isnothing(cfg.plot_dir) ?
                joinpath(dirname(resolved_input), DEFAULT_PLOT_SUBDIR) :
                cfg.plot_dir
            println("Plot directory: $(plot_dir)")
            Base.invokelatest(
                plot_isentropic_pv_maps,
                pv_dataset_path;
                plot_dir,
                plot_theta_k = cfg.plot_theta_k,
                date_single = cfg.date_single,
                date_start = cfg.date_start,
                date_end = cfg.date_end,
            )
        end
    finally
        if cleanup_temp_pv_file && isfile(pv_dataset_path)
            rm(pv_dataset_path; force = true)
        end
    end

    println("Done.")
    return nothing
end

main()
