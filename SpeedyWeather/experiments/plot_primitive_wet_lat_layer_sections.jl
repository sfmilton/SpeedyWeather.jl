#!/usr/bin/env julia

import Pkg

project_dir = joinpath(@__DIR__, "..", "..", "docs")
Pkg.activate(project_dir)
Pkg.instantiate()

using CairoMakie
using Dates
using NCDatasets
using Printf
using Statistics

const DEFAULT_RUN_ID = "primitive_wet_t63_l20_full_physics"
const DEFAULT_INTERVAL_HOURS = 24
const DEFAULT_OUTPUT_SUBDIR = "lat_layer_sections"

const VAR_SPECS = (
    (name = "u", alias = "u", title = "Zonal wind u", unit = "m/s", colormap = :balance),
    (name = "v", alias = "v", title = "Meridional wind v", unit = "m/s", colormap = :balance),
    (name = "temp", alias = "t", title = "Temperature t", unit = "degC", colormap = :thermal),
)

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

function parse_int_arg(arg::AbstractString, name::AbstractString)
    parsed = tryparse(Int, arg)
    isnothing(parsed) && error("Invalid $(name): '$(arg)' (expected integer).")
    return parsed
end

hours_since_start(hours_axis::AbstractVector{<:Real}, i::Integer) =
    Int(round(hours_axis[i] - hours_axis[1]))

function select_time_indices(
        hours_axis::AbstractVector{<:Real};
        interval_hours::Int = DEFAULT_INTERVAL_HOURS,
        max_hour::Union{Nothing, Int} = nothing,
    )
    interval_hours > 0 || error("interval_hours must be > 0.")

    selected = Int[]
    for i in eachindex(hours_axis)
        h = hours_since_start(hours_axis, i)
        h <= 0 && continue
        h % interval_hours == 0 || continue
        !isnothing(max_hour) && h > max_hour && continue
        push!(selected, i)
    end
    return selected
end

function zonal_mean_section(ds::NCDataset, varname::AbstractString, time_index::Integer)
    data = ds[varname][:, :, :, time_index]                     # lon x lat x layer
    data_f64 = Float64.(coalesce.(data, NaN))
    section = dropdims(mean(data_f64, dims = 1), dims = 1)     # lat x layer
    return section
end

finite_values(A::AbstractArray) = A[isfinite.(A)]

function compute_colorranges(ds::NCDataset, time_indices::AbstractVector{Int})
    u_absmax = 0.0
    v_absmax = 0.0
    t_min = Inf
    t_max = -Inf
    found_u = false
    found_v = false
    found_t = false

    for ti in time_indices
        u_section = zonal_mean_section(ds, "u", ti)
        v_section = zonal_mean_section(ds, "v", ti)
        t_section = zonal_mean_section(ds, "temp", ti)

        u_vals = finite_values(u_section)
        v_vals = finite_values(v_section)
        t_vals = finite_values(t_section)

        if !isempty(u_vals)
            found_u = true
            u_absmax = max(u_absmax, maximum(abs, u_vals))
        end
        if !isempty(v_vals)
            found_v = true
            v_absmax = max(v_absmax, maximum(abs, v_vals))
        end
        if !isempty(t_vals)
            found_t = true
            t_min = min(t_min, minimum(t_vals))
            t_max = max(t_max, maximum(t_vals))
        end
    end

    found_u || error("Variable 'u' has no finite values at selected timesteps.")
    found_v || error("Variable 'v' has no finite values at selected timesteps.")
    found_t || error("Variable 'temp' has no finite values at selected timesteps.")

    u_absmax = u_absmax == 0 ? 1.0 : u_absmax
    v_absmax = v_absmax == 0 ? 1.0 : v_absmax

    if !isfinite(t_min) || !isfinite(t_max)
        t_min, t_max = -1.0, 1.0
    elseif t_min == t_max
        t_min -= 1.0
        t_max += 1.0
    end

    return Dict(
        "u" => (-u_absmax, u_absmax),
        "v" => (-v_absmax, v_absmax),
        "temp" => (t_min, t_max),
    )
end

format_time_label(t) = t isa DateTime ? Dates.format(t, "yyyy-mm-dd HH:MM") * " UTC" : string(t)

function plot_one!(
        lat::AbstractVector{<:Real},
        sigma::AbstractVector{<:Real},
        section::AbstractMatrix{<:Real},
        output_path::AbstractString;
        title::AbstractString,
        unit::AbstractString,
        colormap,
        colorrange::Tuple{<:Real, <:Real},
    )
    fig = Figure(size = (1050, 700), fontsize = 20)
    ax = Axis(
        fig[1, 1],
        title = title,
        xlabel = "Latitude [deg]",
        ylabel = "Sigma level",
        xticks = -90:30:90,
    )
    hm = heatmap!(ax, lat, sigma, section; colormap, colorrange)
    ax.yreversed = true
    Colorbar(fig[1, 2], hm, label = unit)
    save(output_path, fig)
    return nothing
end

function main(
        input_path::Union{Nothing, String} = nothing,
        output_dir::Union{Nothing, String} = nothing;
        interval_hours::Int = DEFAULT_INTERVAL_HOURS,
        max_hour::Union{Nothing, Int} = nothing,
    )
    resolved_input = isnothing(input_path) ? latest_output_path(@__DIR__) : input_path
    isfile(resolved_input) || error("Input file not found: $(resolved_input)")

    resolved_output_dir = isnothing(output_dir) ? joinpath(dirname(resolved_input), DEFAULT_OUTPUT_SUBDIR) : output_dir
    mkpath(resolved_output_dir)

    ds = NCDataset(resolved_input)
    try
        lat_raw = Float64.(ds["lat"][:])
        sigma = Float64.(ds["layer"][:])
        times = ds["time"][:]
        hours_axis = Float64.(ds["time"].var[:])
        lat_order = sortperm(lat_raw)
        lat = lat_raw[lat_order]

        time_indices = select_time_indices(hours_axis; interval_hours, max_hour)
        isempty(time_indices) && error(
            "No output timesteps matched T+$(interval_hours)h cadence. " *
            "Available range is T+0h to T+$(hours_since_start(hours_axis, length(hours_axis)))h."
        )

        colorranges = compute_colorranges(ds, time_indices)

        println("Input:  $(resolved_input)")
        println("Output: $(resolved_output_dir)")
        println("Timesteps selected: $(length(time_indices))")

        for ti in time_indices
            h = hours_since_start(hours_axis, ti)
            time_label = format_time_label(times[ti])

            for spec in VAR_SPECS
                section = zonal_mean_section(ds, spec.name, ti)
                section = section[lat_order, :]
                filename = @sprintf("%s_lat_layer_t%03dh.png", spec.alias, h)
                outfile = joinpath(resolved_output_dir, filename)
                title = "$(spec.title), zonal mean lat-layer cross section, T+$(h)h ($(time_label))"

                plot_one!(
                    lat, sigma, section, outfile;
                    title,
                    unit = spec.unit,
                    colormap = spec.colormap,
                    colorrange = colorranges[spec.name],
                )
                println("Wrote $(outfile)")
            end
        end
    finally
        close(ds)
    end

    return nothing
end

input_path = length(ARGS) >= 1 && ARGS[1] != "-" ? ARGS[1] : nothing
output_dir = length(ARGS) >= 2 && ARGS[2] != "-" ? ARGS[2] : nothing
interval_hours = length(ARGS) >= 3 ? parse_int_arg(ARGS[3], "interval_hours") : DEFAULT_INTERVAL_HOURS
max_hour = length(ARGS) >= 4 ? parse_int_arg(ARGS[4], "max_hour") : nothing

main(input_path, output_dir; interval_hours, max_hour)
