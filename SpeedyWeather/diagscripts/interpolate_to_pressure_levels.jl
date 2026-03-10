#!/usr/bin/env julia

import Pkg

project_dir = joinpath(@__DIR__, "..")
Pkg.activate(project_dir)
Pkg.instantiate()

using NCDatasets
using Printf

const DEFAULT_RUN_ID = "primitive_wet_t63_l20_full_physics"
const DEFAULT_PRESSURE_LEVELS_HPA = [1000.0, 950.0, 850.0, 700.0, 600.0, 500.0, 400.0, 300.0, 250.0,200.0,150.0,150.0,70.0]
const DEFAULT_VARIABLES = ["u", "v", "temp"]
const STANDARD_GRAVITY = 9.80665

@kwdef struct PressureLevelVariableSpec
    request_name::String
    output_name::String
    source_name::String
    unit::String
    long_name::String
    scale_factor::Float64 = 1.0
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

function parse_csv_strings(s::AbstractString)
    vals = String[]
    for token in split(s, ",")
        stripped = strip(token)
        isempty(stripped) || push!(vals, stripped)
    end
    isempty(vals) && error("No variable names provided.")
    return vals
end

function copy_attrs_as_dict(var)
    d = Dict{String, Any}()
    for key in keys(var.attrib)
        d[string(key)] = var.attrib[key]
    end
    return d
end

function variable_spec(ds_in::NCDataset, request_name::AbstractString)
    request = String(request_name)
    request_lc = lowercase(request)

    if haskey(ds_in, request)
        src = ds_in[request]
        long_name = haskey(src.attrib, "long_name") ? string(src.attrib["long_name"]) : request
        unit = if haskey(src.attrib, "units")
            string(src.attrib["units"])
        elseif haskey(src.attrib, "unit")
            string(src.attrib["unit"])
        else
            "unknown"
        end

        return PressureLevelVariableSpec(
            request_name = request,
            output_name = request,
            source_name = request,
            unit = unit,
            long_name = "$(long_name) on pressure levels",
            scale_factor = 1.0,
        )
    end

    if request_lc in ("z", "zg", "gph", "geopotential_height")
        haskey(ds_in, "geopotential") || error(
            "Requested '$(request)', but input does not contain 'geopotential'. " *
            "Add SpeedyWeather.GeopotentialOutput() to your model output first."
        )
        return PressureLevelVariableSpec(
            request_name = request,
            output_name = "z",
            source_name = "geopotential",
            unit = "m",
            long_name = "geopotential height on pressure levels",
            scale_factor = 1 / STANDARD_GRAVITY,
        )
    end

    error("Requested variable '$(request)' does not exist in input and is not a supported alias.")
end

@inline function interp_linear(
        x::AbstractVector{Float64},
        y::AbstractVector{Float64},
        n::Int,
        xq::Float64,
    )
    n < 2 && return NaN
    xlo = x[1]
    xhi = x[n]
    if xq < xlo || xq > xhi
        return NaN
    end
    if xq == xlo
        return y[1]
    elseif xq == xhi
        return y[n]
    end

    lo = 1
    hi = n
    while hi - lo > 1
        mid = (lo + hi) >>> 1
        if x[mid] <= xq
            lo = mid
        else
            hi = mid
        end
    end

    x1 = x[lo]
    x2 = x[hi]
    y1 = y[lo]
    y2 = y[hi]
    x2 == x1 && return y1
    α = (xq - x1) / (x2 - x1)
    return muladd(α, y2 - y1, y1)
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

function interpolate_variable!(
        ds_out::NCDataset,
        ds_in::NCDataset,
        spec::PressureLevelVariableSpec,
        sigma::Vector{Float64},
        plev_hpa::Vector{Float64},
        surface_pressure_var::AbstractString,
    )
    haskey(ds_in, spec.source_name) || error("Input file does not contain variable '$(spec.source_name)'.")
    haskey(ds_in, surface_pressure_var) || error("Input file does not contain '$(surface_pressure_var)'.")

    raw = ds_in[spec.source_name]
    sz = size(raw)
    length(sz) == 4 || error("Variable '$(spec.source_name)' must have dims (lon,lat,layer,time), got size=$(sz).")
    nlon, nlat, nlayer, ntime = sz
    nlayer == length(sigma) || error("Layer mismatch for '$(spec.source_name)': file has $(nlayer), sigma has $(length(sigma)).")

    order = sigma_order(sigma)
    sigma_sorted = sigma[order]
    nplev = length(plev_hpa)

    pcol = Vector{Float64}(undef, nlayer)
    vcol = Vector{Float64}(undef, nlayer)
    pvalid = Vector{Float64}(undef, nlayer)
    vvalid = Vector{Float64}(undef, nlayer)
    out_slice = fill(Float32(NaN), nlon, nlat, nplev)

    for it in 1:ntime
        pres_t = Float64.(coalesce.(ds_in[surface_pressure_var][:, :, it], NaN))  # hPa
        var_t = Float64.(coalesce.(raw[:, :, :, it], NaN))                  # lon x lat x layer
        fill!(out_slice, Float32(NaN))

        for j in 1:nlat
            for i in 1:nlon
                ps = pres_t[i, j]
                isfinite(ps) || continue
                ps > 0 || continue

                @inbounds for k in 1:nlayer
                    pcol[k] = sigma_sorted[k] * ps
                    vcol[k] = var_t[i, j, order[k]]
                end

                nvalid = 0
                @inbounds for k in 1:nlayer
                    pk = pcol[k]
                    vk = vcol[k]
                    if isfinite(pk) && isfinite(vk)
                        nvalid += 1
                        pvalid[nvalid] = pk
                        vvalid[nvalid] = vk
                    end
                end

                nvalid >= 2 || continue

                @inbounds for ip in 1:nplev
                    val = interp_linear(pvalid, vvalid, nvalid, plev_hpa[ip])
                    isfinite(val) && (out_slice[i, j, ip] = Float32(val * spec.scale_factor))
                end
            end
        end

        ds_out[spec.output_name][:, :, :, it] = out_slice
        @printf("  %s: wrote time %d/%d\n", spec.output_name, it, ntime)
    end
end

function select_surface_pressure_var(ds_in::NCDataset)
    if haskey(ds_in, "pres")
        return "pres"
    elseif haskey(ds_in, "mslp")
        @warn "Variable 'pres' not found. Falling back to 'mslp' as surface pressure proxy for pressure-level interpolation."
        return "mslp"
    else
        error("Input file needs 'pres' (preferred) or 'mslp' to infer pressure levels.")
    end
end

function create_output_dataset(
        input_path::AbstractString,
        output_path::AbstractString,
        plev_hpa::Vector{Float64},
        requested_variables::Vector{String},
    )
    isfile(output_path) && error("Output file already exists: $(output_path)")

    ds_in = NCDataset(input_path, "r")
    try
        for req in ("lon", "lat", "layer", "time")
            haskey(ds_in, req) || error("Input file is missing required variable '$(req)'.")
        end
        surface_pressure_var = select_surface_pressure_var(ds_in)
        println("Using surface pressure variable: $(surface_pressure_var)")

        lon = Float64.(ds_in["lon"][:])
        lat = Float64.(ds_in["lat"][:])
        sigma = Float64.(ds_in["layer"][:])
        time_raw = Float64.(ds_in["time"].var[:])
        ntime = length(time_raw)
        specs = map(var -> variable_spec(ds_in, var), requested_variables)

        ds_out = NCDataset(output_path, "c")
        try
            defDim(ds_out, "lon", length(lon))
            defDim(ds_out, "lat", length(lat))
            defDim(ds_out, "plev", length(plev_hpa))
            defDim(ds_out, "time", ntime)

            lon_attrs = copy_attrs_as_dict(ds_in["lon"])
            lat_attrs = copy_attrs_as_dict(ds_in["lat"])
            time_attrs = copy_attrs_as_dict(ds_in["time"])

            defVar(ds_out, "lon", lon, ("lon",), attrib = lon_attrs)
            defVar(ds_out, "lat", lat, ("lat",), attrib = lat_attrs)
            defVar(
                ds_out, "plev", plev_hpa, ("plev",),
                attrib = Dict(
                    "units" => "hPa",
                    "long_name" => "pressure level",
                    "positive" => "down",
                    "standard_name" => "air_pressure",
                )
            )
            defVar(ds_out, "time", Float64, ("time",), attrib = time_attrs)
            ds_out["time"][:] = time_raw

            output_names = String[]
            for spec in specs
                spec.output_name in output_names && error("Duplicate output variable name requested: '$(spec.output_name)'.")
                push!(output_names, spec.output_name)
                defVar(
                    ds_out, spec.output_name, Float32, ("lon", "lat", "plev", "time"),
                    attrib = Dict(
                        "units" => spec.unit,
                        "long_name" => spec.long_name,
                        "coordinates" => "lon lat plev time",
                    )
                )
            end

            for spec in specs
                interpolate_variable!(ds_out, ds_in, spec, sigma, plev_hpa, surface_pressure_var)
            end
        finally
            close(ds_out)
        end
    finally
        close(ds_in)
    end

    return nothing
end

function main(
        input_path::Union{Nothing, String} = nothing,
        output_path::Union{Nothing, String} = nothing,
        plev_hpa::Vector{Float64} = DEFAULT_PRESSURE_LEVELS_HPA,
        requested_variables::Vector{String} = DEFAULT_VARIABLES,
    )
    resolved_input = isnothing(input_path) ? latest_output_path(@__DIR__) : input_path
    isfile(resolved_input) || error("Input file not found: $(resolved_input)")

    resolved_output = isnothing(output_path) ?
        joinpath(dirname(resolved_input), "output_pressure_levels.nc") :
        output_path

    println("Input:  $(resolved_input)")
    println("Output: $(resolved_output)")
    println("Levels [hPa]: $(join(plev_hpa, ", "))")
    println("Variables: $(join(requested_variables, ", "))")

    create_output_dataset(resolved_input, resolved_output, plev_hpa, requested_variables)
    println("Done.")
    return nothing
end

input_arg = length(ARGS) >= 1 && ARGS[1] != "-" ? ARGS[1] : nothing
output_arg = length(ARGS) >= 2 && ARGS[2] != "-" ? ARGS[2] : nothing
plev_arg = length(ARGS) >= 3 ? parse_csv_floats(ARGS[3], name = "pressure level") : DEFAULT_PRESSURE_LEVELS_HPA
vars_arg = length(ARGS) >= 4 ? parse_csv_strings(ARGS[4]) : DEFAULT_VARIABLES

main(input_arg, output_arg, plev_arg, vars_arg)
