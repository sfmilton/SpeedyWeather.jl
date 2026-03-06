#!/usr/bin/env julia

import Pkg

project_dir = joinpath(@__DIR__, "..")
Pkg.activate(project_dir)
Pkg.instantiate()

using NCDatasets
using Printf

const DEFAULT_RUN_ID = "primitive_wet_t63_l20_full_physics"
const DEFAULT_THETA_LEVELS_K = [315.0, 350.0]
const KAPPA = 0.2857142857142857          # R_d / c_p
const P0_HPA = 1000.0                     # reference pressure for potential temperature
const OMEGA = 7.292115e-5                 # Earth rotation rate [s^-1]
const GRAVITY = 9.80665                   # [m/s^2]

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

@inline function find_theta_bracket(θ_col::Vector{Float64}, θ_target::Float64)
    n = length(θ_col)
    @inbounds for k in 1:n-1
        θ1 = θ_col[k]
        θ2 = θ_col[k + 1]
        if !isfinite(θ1) || !isfinite(θ2) || θ1 == θ2
            continue
        end
        in_interval = (θ1 <= θ_target <= θ2) || (θ2 <= θ_target <= θ1)
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
    θ_col = Vector{Float64}(undef, nlayer)   # K
    ζ_col = Vector{Float64}(undef, nlayer)   # s^-1

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
                    T_c = temp_t[i, j, ks]
                    ζ = vor_t[i, j, ks]

                    if !isfinite(p_hpa) || p_hpa <= 0 || !isfinite(T_c) || !isfinite(ζ)
                        p_col[k] = NaN
                        θ_col[k] = NaN
                        ζ_col[k] = NaN
                        continue
                    end

                    p_col[k] = 100 * p_hpa                  # Pa
                    T_k = T_c + 273.15
                    θ_col[k] = T_k * (P0_HPA / p_hpa)^KAPPA
                    ζ_col[k] = ζ
                end

                for n in 1:ntheta
                    θ_target = theta_levels_k[n]
                    k = find_theta_bracket(θ_col, θ_target)
                    k == 0 && continue

                    k2 = k + 1
                    θ1 = θ_col[k]
                    θ2 = θ_col[k2]
                    p1 = p_col[k]
                    p2 = p_col[k2]
                    ζ1 = ζ_col[k]
                    ζ2 = ζ_col[k2]

                    if !isfinite(θ1) || !isfinite(θ2) || !isfinite(p1) || !isfinite(p2) || p1 == p2
                        continue
                    end

                    α = (θ_target - θ1) / (θ2 - θ1)
                    p_interp = muladd(α, p2 - p1, p1)       # Pa
                    ζ_interp = muladd(α, ζ2 - ζ1, ζ1)       # s^-1
                    dθdp = (θ2 - θ1) / (p2 - p1)            # K/Pa

                    pv = -GRAVITY * (ζ_interp + f) * dθdp   # K m^2 kg^-1 s^-1
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
                ds_out, ds_in, sigma, theta_levels_k, surface_pressure_var
            )
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
        theta_levels_k::Vector{Float64} = DEFAULT_THETA_LEVELS_K,
    )
    resolved_input = isnothing(input_path) ? latest_output_path(@__DIR__) : input_path
    isfile(resolved_input) || error("Input file not found: $(resolved_input)")

    resolved_output = isnothing(output_path) ?
        joinpath(dirname(resolved_input), "output_isentropic_pv.nc") :
        output_path

    println("Input:  $(resolved_input)")
    println("Output: $(resolved_output)")
    println("Theta levels [K]: $(join(theta_levels_k, ", "))")

    create_output_dataset(resolved_input, resolved_output, theta_levels_k)
    println("Done.")
    return nothing
end

input_arg = length(ARGS) >= 1 && ARGS[1] != "-" ? ARGS[1] : nothing
output_arg = length(ARGS) >= 2 && ARGS[2] != "-" ? ARGS[2] : nothing
theta_arg = length(ARGS) >= 3 ? parse_csv_floats(ARGS[3], name = "theta level") : DEFAULT_THETA_LEVELS_K

main(input_arg, output_arg, theta_arg)
