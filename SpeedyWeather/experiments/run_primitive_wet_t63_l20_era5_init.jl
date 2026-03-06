#!/usr/bin/env julia

import Pkg

project_dir = joinpath(@__DIR__, "..")
Pkg.activate(project_dir)
Pkg.instantiate()

using Dates
using NCDatasets
using Printf
using SpeedyWeather
using Statistics

const DEFAULT_OUTPUT_ID = "primitive_wet_t63_l20_era5_ic"

Base.@kwdef mutable struct Config
    pressure_file::String = ""
    surface_file::String = ""
    time_index::Int = 1
    period_days::Int = 10
    trunc::Int = 63
    nlayers::Int = 20
    output_dt_hours::Int = 6
    output_id::String = DEFAULT_OUTPUT_ID
    # variable names
    lon_name::String = "longitude"
    lat_name::String = "latitude"
    level_name::String = "level"
    time_name::String = "time"
    u_name::String = "u"
    v_name::String = "v"
    t_name::String = "t"
    q_name::String = "q"
    sp_name::String = "sp"
end

function print_usage()
    println(
        """
Usage:
  julia experiments/run_primitive_wet_t63_l20_era5_init.jl <era5_pressure_levels.nc> [options]

Options:
  --surface-file PATH   NetCDF file with surface pressure variable (default: same as pressure file)
  --time-index N        Time index (1-based, default: 1)
  --period-days N       Run length in days (default: 10)
  --output-id ID        Run/output id (default: $(DEFAULT_OUTPUT_ID))
  --output-dt-hours N   Output interval in hours (default: 6)
  --trunc N             Spectral truncation (default: 63)
  --nlayers N           Number of sigma layers (default: 20)

  --lon-name NAME       Longitude coordinate name (default: longitude)
  --lat-name NAME       Latitude coordinate name (default: latitude)
  --level-name NAME     Pressure-level coordinate name (default: level)
  --time-name NAME      Time coordinate name (default: time)
  --u-name NAME         Zonal wind variable name (default: u)
  --v-name NAME         Meridional wind variable name (default: v)
  --t-name NAME         Temperature variable name (default: t)
  --q-name NAME         Specific humidity variable name (default: q)
  --sp-name NAME        Surface pressure variable name (default: sp)

Notes:
  - Pressure-level file should contain u/v/t/q on pressure levels.
  - Surface pressure is expected in Pa and is converted to ln(Pa) for model 'pres'.
  - Pressure levels can be in hPa or Pa; units are auto-detected.
"""
    )
end

function parse_args(args)
    isempty(args) && (print_usage(); error("Missing required ERA5 pressure-level file argument."))

    cfg = Config()
    i = 1

    while i <= length(args)
        arg = args[i]
        if arg == "-h" || arg == "--help"
            print_usage()
            exit(0)
        elseif startswith(arg, "--")
            i == length(args) && error("Option $(arg) requires a value.")
            val = args[i + 1]

            if arg == "--surface-file"
                cfg.surface_file = val
            elseif arg == "--time-index"
                cfg.time_index = parse(Int, val)
            elseif arg == "--period-days"
                cfg.period_days = parse(Int, val)
            elseif arg == "--output-id"
                cfg.output_id = val
            elseif arg == "--output-dt-hours"
                cfg.output_dt_hours = parse(Int, val)
            elseif arg == "--trunc"
                cfg.trunc = parse(Int, val)
            elseif arg == "--nlayers"
                cfg.nlayers = parse(Int, val)
            elseif arg == "--lon-name"
                cfg.lon_name = val
            elseif arg == "--lat-name"
                cfg.lat_name = val
            elseif arg == "--level-name"
                cfg.level_name = val
            elseif arg == "--time-name"
                cfg.time_name = val
            elseif arg == "--u-name"
                cfg.u_name = val
            elseif arg == "--v-name"
                cfg.v_name = val
            elseif arg == "--t-name"
                cfg.t_name = val
            elseif arg == "--q-name"
                cfg.q_name = val
            elseif arg == "--sp-name"
                cfg.sp_name = val
            else
                error("Unknown option: $(arg)")
            end
            i += 2
        else
            if isempty(cfg.pressure_file)
                cfg.pressure_file = arg
            else
                error("Unexpected positional argument: $(arg)")
            end
            i += 1
        end
    end

    isempty(cfg.pressure_file) && error("Missing ERA5 pressure-level file.")
    isempty(cfg.surface_file) && (cfg.surface_file = cfg.pressure_file)

    cfg.time_index >= 1 || error("--time-index must be >= 1")
    cfg.period_days >= 0 || error("--period-days must be >= 0")
    cfg.trunc > 0 || error("--trunc must be > 0")
    cfg.nlayers > 0 || error("--nlayers must be > 0")
    cfg.output_dt_hours > 0 || error("--output-dt-hours must be > 0")

    isfile(cfg.pressure_file) || error("Pressure-level file does not exist: $(cfg.pressure_file)")
    isfile(cfg.surface_file) || error("Surface file does not exist: $(cfg.surface_file)")

    return cfg
end

@inline function _to_float_nan(x)
    x isa Missing && return NaN
    return Float64(x)
end

function to_float_nan_array(A)
    out = Array{Float64}(undef, size(A))
    @inbounds for idx in eachindex(A)
        out[idx] = _to_float_nan(A[idx])
    end
    return out
end

function dim_index(v, name::AbstractString)
    dims = String.(dimnames(v))
    idx = findfirst(==(name), dims)
    isnothing(idx) && error("Variable does not have dimension '$(name)'. Dimensions are $(dims).")
    return idx
end

function read_3d_at_time(ds::NCDataset, varname::String, cfg::Config)
    haskey(ds, varname) || error("Variable '$(varname)' not found in $(ds.path).")
    v = ds[varname]
    raw = to_float_nan_array(v.var[:])

    i_lon = dim_index(v, cfg.lon_name)
    i_lat = dim_index(v, cfg.lat_name)
    i_lev = dim_index(v, cfg.level_name)

    dims = String.(dimnames(v))
    i_time = findfirst(==(cfg.time_name), dims)

    if isnothing(i_time)
        data = raw
    else
        nt = size(raw, i_time)
        cfg.time_index <= nt || error("Requested time index $(cfg.time_index), but '$(varname)' has only $(nt) time steps.")
        data = selectdim(raw, i_time, cfg.time_index)
        kept = collect(1:length(dims))
        deleteat!(kept, i_time)
        i_lon = findfirst(==(i_lon), kept)
        i_lat = findfirst(==(i_lat), kept)
        i_lev = findfirst(==(i_lev), kept)
    end

    perm = (i_lon, i_lat, i_lev)
    return Array(permutedims(data, perm))
end

function read_2d_at_time(ds::NCDataset, varname::String, cfg::Config)
    haskey(ds, varname) || error("Variable '$(varname)' not found in $(ds.path).")
    v = ds[varname]
    raw = to_float_nan_array(v.var[:])

    i_lon = dim_index(v, cfg.lon_name)
    i_lat = dim_index(v, cfg.lat_name)

    dims = String.(dimnames(v))
    i_time = findfirst(==(cfg.time_name), dims)

    if isnothing(i_time)
        data = raw
    else
        nt = size(raw, i_time)
        cfg.time_index <= nt || error("Requested time index $(cfg.time_index), but '$(varname)' has only $(nt) time steps.")
        data = selectdim(raw, i_time, cfg.time_index)
        kept = collect(1:length(dims))
        deleteat!(kept, i_time)
        i_lon = findfirst(==(i_lon), kept)
        i_lat = findfirst(==(i_lat), kept)
    end

    perm = (i_lon, i_lat)
    return Array(permutedims(data, perm))
end

function unique_sorted_indices(vals::Vector{Float64}; atol::Float64 = 1e-10)
    perm = sortperm(vals)
    keep = trues(length(perm))
    if !isempty(perm)
        prev = vals[perm[1]]
        for i in 2:length(perm)
            cur = vals[perm[i]]
            if abs(cur - prev) <= atol
                keep[i] = false
            else
                prev = cur
            end
        end
    end
    return perm[keep]
end

function pressure_levels_to_pa(levels::Vector{Float64})
    maxlev = maximum(levels)
    if maxlev < 2_000
        return 100.0 .* levels
    elseif maxlev < 200_000
        return levels
    else
        error("Pressure levels look invalid (max=$(maxlev)). Expected hPa or Pa.")
    end
end

function surface_pressure_to_pa(ps::Array{Float64, 2})
    finite_vals = ps[isfinite.(ps)]
    isempty(finite_vals) && error("Surface pressure field is all missing/NaN.")
    mean_ps = mean(finite_vals)

    if mean_ps < 2_000
        return 100.0 .* ps
    elseif mean_ps < 200_000
        return ps
    else
        error("Surface pressure looks invalid (mean=$(mean_ps)). Expected hPa or Pa.")
    end
end

struct ERA5IC
    lon::Vector{Float64}
    lat::Vector{Float64}
    p::Vector{Float64}
    u::Array{Float64, 3}
    v::Array{Float64, 3}
    t::Array{Float64, 3}
    q::Array{Float64, 3}
    ps::Array{Float64, 2}
end

function load_era5_ic(cfg::Config)
    ds_pl = NCDataset(cfg.pressure_file)
    ds_sf = NCDataset(cfg.surface_file)

    try
        haskey(ds_pl, cfg.lon_name) || error("Missing coordinate '$(cfg.lon_name)' in $(cfg.pressure_file)")
        haskey(ds_pl, cfg.lat_name) || error("Missing coordinate '$(cfg.lat_name)' in $(cfg.pressure_file)")
        haskey(ds_pl, cfg.level_name) || error("Missing coordinate '$(cfg.level_name)' in $(cfg.pressure_file)")
        haskey(ds_sf, cfg.lon_name) || error("Missing coordinate '$(cfg.lon_name)' in $(cfg.surface_file)")
        haskey(ds_sf, cfg.lat_name) || error("Missing coordinate '$(cfg.lat_name)' in $(cfg.surface_file)")

        lon_raw = Float64.(ds_pl[cfg.lon_name][:])
        lat_raw = Float64.(ds_pl[cfg.lat_name][:])
        p_raw = Float64.(ds_pl[cfg.level_name][:])
        lon_raw_sf = Float64.(ds_sf[cfg.lon_name][:])
        lat_raw_sf = Float64.(ds_sf[cfg.lat_name][:])

        u = read_3d_at_time(ds_pl, cfg.u_name, cfg)
        v = read_3d_at_time(ds_pl, cfg.v_name, cfg)
        t = read_3d_at_time(ds_pl, cfg.t_name, cfg)
        q = read_3d_at_time(ds_pl, cfg.q_name, cfg)

        ps = read_2d_at_time(ds_sf, cfg.sp_name, cfg)

        size(u) == size(v) == size(t) == size(q) || error("u/v/t/q must all have identical dimensions.")
        nlon, nlat, nlev = size(u)
        size(ps) == (nlon, nlat) || error("Surface pressure size $(size(ps)) does not match horizontal size ($(nlon), $(nlat)).")

        # Normalize and sort longitudes to [0, 360), remove duplicates if needed.
        lon_norm = mod.(lon_raw, 360.0)
        lon_perm = unique_sorted_indices(lon_norm)
        lon = lon_norm[lon_perm]

        # Sort latitude ascending.
        lat_perm = sortperm(lat_raw)
        lat = lat_raw[lat_perm]

        # Surface file coordinates must match pressure-level file coordinates.
        lon_norm_sf = mod.(lon_raw_sf, 360.0)
        lon_perm_sf = unique_sorted_indices(lon_norm_sf)
        lon_sf = lon_norm_sf[lon_perm_sf]

        lat_perm_sf = sortperm(lat_raw_sf)
        lat_sf = lat_raw_sf[lat_perm_sf]

        length(lon_sf) == length(lon) || error(
            "Surface longitude length $(length(lon_sf)) does not match pressure-level longitude length $(length(lon))."
        )
        length(lat_sf) == length(lat) || error(
            "Surface latitude length $(length(lat_sf)) does not match pressure-level latitude length $(length(lat))."
        )
        maximum(abs.(lon_sf .- lon)) < 1e-6 || error(
            "Surface and pressure-level longitude coordinates differ. Provide files on the same horizontal grid."
        )
        maximum(abs.(lat_sf .- lat)) < 1e-6 || error(
            "Surface and pressure-level latitude coordinates differ. Provide files on the same horizontal grid."
        )

        # Sort pressure ascending in Pa.
        p_pa = pressure_levels_to_pa(p_raw)
        p_perm = sortperm(p_pa)
        p = p_pa[p_perm]

        u = u[lon_perm, lat_perm, p_perm]
        v = v[lon_perm, lat_perm, p_perm]
        t = t[lon_perm, lat_perm, p_perm]
        q = q[lon_perm, lat_perm, p_perm]
        ps = ps[lon_perm_sf, lat_perm_sf]
        ps = surface_pressure_to_pa(ps)

        # Basic range checks.
        minimum(p) > 0 || error("Pressure levels must be positive.")
        minimum(ps) > 0 || error("Surface pressure contains non-positive values.")

        return ERA5IC(lon, lat, p, u, v, t, q, ps)
    finally
        close(ds_pl)
        close(ds_sf)
    end
end

@inline function lon_bracket(lon::Vector{Float64}, λ::Float64)
    λn = mod(λ, 360.0)
    n = length(lon)

    if λn < lon[1]
        i1 = n
        i2 = 1
        w = (λn + 360.0 - lon[n]) / (lon[1] + 360.0 - lon[n])
        return i1, i2, w
    elseif λn > lon[n]
        i1 = n
        i2 = 1
        w = (λn - lon[n]) / (lon[1] + 360.0 - lon[n])
        return i1, i2, w
    else
        lo = searchsortedlast(lon, λn)
        if lo == n
            return n, 1, 0.0
        end
        hi = lo + 1
        w = (λn - lon[lo]) / (lon[hi] - lon[lo])
        return lo, hi, w
    end
end

@inline function bracket_clamped(x::Vector{Float64}, xq::Float64)
    n = length(x)
    if xq <= x[1]
        return 1, 1, 0.0
    elseif xq >= x[n]
        return n, n, 0.0
    else
        lo = clamp(searchsortedlast(x, xq), 1, n - 1)
        hi = lo + 1
        w = (xq - x[lo]) / (x[hi] - x[lo])
        return lo, hi, w
    end
end

@inline function weighted_value(values::NTuple{4, Float64}, weights::NTuple{4, Float64})
    v11, v21, v12, v22 = values
    w11, w21, w12, w22 = weights

    if isfinite(v11) && isfinite(v21) && isfinite(v12) && isfinite(v22)
        return w11 * v11 + w21 * v21 + w12 * v12 + w22 * v22
    end

    s = 0.0
    wsum = 0.0

    if isfinite(v11)
        s += w11 * v11
        wsum += w11
    end
    if isfinite(v21)
        s += w21 * v21
        wsum += w21
    end
    if isfinite(v12)
        s += w12 * v12
        wsum += w12
    end
    if isfinite(v22)
        s += w22 * v22
        wsum += w22
    end

    return wsum > 0 ? s / wsum : NaN
end

@inline function bilinear_2d(field::AbstractMatrix{Float64}, i1::Int, i2::Int, wx::Float64, j1::Int, j2::Int, wy::Float64)
    w11 = (1 - wx) * (1 - wy)
    w21 = wx * (1 - wy)
    w12 = (1 - wx) * wy
    w22 = wx * wy
    return weighted_value((field[i1, j1], field[i2, j1], field[i1, j2], field[i2, j2]), (w11, w21, w12, w22))
end

@inline function bilinear_3d_level(
        field::Array{Float64, 3},
        i1::Int,
        i2::Int,
        wx::Float64,
        j1::Int,
        j2::Int,
        wy::Float64,
        k::Int,
    )
    w11 = (1 - wx) * (1 - wy)
    w21 = wx * (1 - wy)
    w12 = (1 - wx) * wy
    w22 = wx * wy
    return weighted_value((field[i1, j1, k], field[i2, j1, k], field[i1, j2, k], field[i2, j2, k]), (w11, w21, w12, w22))
end

@inline function ps_at(ic::ERA5IC, λ::Float64, φ::Float64)
    i1, i2, wx = lon_bracket(ic.lon, λ)
    j1, j2, wy = bracket_clamped(ic.lat, φ)
    return bilinear_2d(ic.ps, i1, i2, wx, j1, j2, wy)
end

@inline function interp_field_at_sigma(ic::ERA5IC, field::Array{Float64, 3}, λ::Float64, φ::Float64, σ::Float64)
    i1, i2, wx = lon_bracket(ic.lon, λ)
    j1, j2, wy = bracket_clamped(ic.lat, φ)

    ps = bilinear_2d(ic.ps, i1, i2, wx, j1, j2, wy)
    isfinite(ps) || return NaN

    p_target = clamp(σ * ps, ic.p[1], ic.p[end])
    k1, k2, wz = bracket_clamped(ic.p, p_target)

    v1 = bilinear_3d_level(field, i1, i2, wx, j1, j2, wy, k1)
    if k1 == k2
        return v1
    end
    v2 = bilinear_3d_level(field, i1, i2, wx, j1, j2, wy, k2)

    if isfinite(v1) && isfinite(v2)
        return muladd(wz, v2 - v1, v1)
    elseif isfinite(v1)
        return v1
    elseif isfinite(v2)
        return v2
    else
        return NaN
    end
end

function apply_era5_initial_conditions!(simulation, ic::ERA5IC)
    u_func = (λ, φ, σ) -> interp_field_at_sigma(ic, ic.u, λ, φ, σ)
    v_func = (λ, φ, σ) -> interp_field_at_sigma(ic, ic.v, λ, φ, σ)
    t_func = (λ, φ, σ) -> interp_field_at_sigma(ic, ic.t, λ, φ, σ)
    q_func = (λ, φ, σ) -> max(0.0, interp_field_at_sigma(ic, ic.q, λ, φ, σ))
    p_func = (λ, φ) -> begin
        ps = ps_at(ic, λ, φ)
        isfinite(ps) && ps > 0 ? log(ps) : NaN
    end

    set!(
        simulation;
        u = u_func,
        v = v_func,
        temp = t_func,
        humid = q_func,
        pres = p_func,
        lf = 1,
        static_func = false,
    )

    return simulation
end

function main(args = ARGS)
    cfg = parse_args(args)

    println("Loading ERA5 from:")
    println("  pressure levels: $(cfg.pressure_file)")
    println("  surface file:    $(cfg.surface_file)")
    println("  time index:      $(cfg.time_index)")

    ic = load_era5_ic(cfg)
    @printf("Loaded ERA5 grids: lon=%d lat=%d plev=%d\n", length(ic.lon), length(ic.lat), length(ic.p))
    @printf("Pressure range: %.1f to %.1f hPa\n", minimum(ic.p) / 100.0, maximum(ic.p) / 100.0)

    spectral_grid = SpectralGrid(trunc = cfg.trunc, nlayers = cfg.nlayers)
    time_stepping = Leapfrog(spectral_grid, Δt_at_T31 = Minute(20))
    output = NetCDFOutput(
        spectral_grid,
        PrimitiveWet,
        output_dt = Hour(cfg.output_dt_hours),
        path = @__DIR__,
        id = cfg.output_id,
    )
    add!(output, SpeedyWeather.SurfacePressureOutput())
    add!(output, SpeedyWeather.SurfaceFluxesOutput()...)
    add!(output, SpeedyWeather.RadiationOutput()...)
    add!(output, SpeedyWeather.PrecipitationOutput()...)
    add!(output, SpeedyWeather.TendencyBudgetOutput()...)

    model = PrimitiveWetModel(spectral_grid; physics = true, output, time_stepping)
    simulation = initialize!(model)

    println("Applying ERA5 initial conditions to wet PE state...")
    apply_era5_initial_conditions!(simulation, ic)

    println("Starting PrimitiveWet run at T$(cfg.trunc)/L$(cfg.nlayers) with full physics.")
    run!(simulation, period = Day(cfg.period_days), output = true)

    output_path = joinpath(simulation.model.output.run_path, simulation.model.output.filename)
    println("Finished run.")
    println("Output file: $(output_path)")

    return simulation
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
