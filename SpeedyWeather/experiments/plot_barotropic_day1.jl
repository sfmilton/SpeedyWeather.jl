#!/usr/bin/env julia

import Pkg

project_dir = joinpath(@__DIR__, "..")
Pkg.activate(project_dir)
Pkg.instantiate()

using Dates
using NCDatasets
using Printf

const DEFAULT_INPUT = joinpath(@__DIR__, "run_barotropic_0001", "output.nc")
const DEFAULT_OUTPUT = joinpath(@__DIR__, "run_barotropic_0001", "vor_day1_latlon.svg")
const DEFAULT_TIME_INDEX = 2
const DEFAULT_LAYER_INDEX = 1

escape_xml(s::AbstractString) = replace(
    s,
    "&" => "&amp;",
    "\"" => "&quot;",
    "<" => "&lt;",
    ">" => "&gt;",
)

function lerp_channel(a::Int, b::Int, t::Float64)
    return round(Int, a + (b - a) * t)
end

function color_hex(value::Float64, scale::Float64)
    if !isfinite(value) || scale == 0
        return "#d9d9d9"
    end

    t = clamp(value / scale, -1.0, 1.0)
    blue = (49, 54, 149)
    white = (247, 247, 247)
    red = (165, 0, 38)

    if t < 0
        u = t + 1
        r = lerp_channel(blue[1], white[1], u)
        g = lerp_channel(blue[2], white[2], u)
        b = lerp_channel(blue[3], white[3], u)
    else
        u = t
        r = lerp_channel(white[1], red[1], u)
        g = lerp_channel(white[2], red[2], u)
        b = lerp_channel(white[3], red[3], u)
    end

    return @sprintf("#%02x%02x%02x", r, g, b)
end

function format_lon(lon::Real)
    lon == 0 && return "0"
    abs(lon) == 180 && return "180"
    return lon < 0 ? "$(Int(abs(lon)))W" : "$(Int(lon))E"
end

function format_lat(lat::Real)
    lat == 0 && return "0"
    abs(lat) == 90 && return string(Int(lat))
    return lat < 0 ? "$(Int(abs(lat)))S" : "$(Int(lat))N"
end

function centered_longitudes(lon::AbstractVector, field::AbstractMatrix)
    split = findfirst(x -> x >= 180, lon)
    split === nothing && return collect(lon), field

    lon_centered = vcat(lon[split:end] .- 360, lon[1:split-1])
    field_centered = vcat(field[split:end, :], field[1:split-1, :])
    return lon_centered, field_centered
end

function latitude_edges(lat::AbstractVector)
    n = length(lat)
    edges = Vector{Float64}(undef, n + 1)
    edges[1] = 90.0
    edges[end] = -90.0
    for j in 1:n-1
        edges[j + 1] = (lat[j] + lat[j + 1]) / 2
    end
    return edges
end

function write_svg(path::AbstractString, lon::Vector{Float64}, lat::Vector{Float64}, field::Matrix{Float64}, time::DateTime)
    width = 1100
    height = 620
    left = 95
    right = 170
    top = 60
    bottom = 85
    plot_width = width - left - right
    plot_height = height - top - bottom
    colorbar_width = 24
    colorbar_gap = 36

    lon_edges = collect(range(-180.0, 180.0; length = length(lon) + 1))
    lat_edges = latitude_edges(lat)
    scale = maximum(abs, field)

    xmap(x) = left + (x + 180.0) / 360.0 * plot_width
    ymap(y) = top + (90.0 - y) / 180.0 * plot_height

    open(path, "w") do io
        println(io, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>")
        println(io, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"$width\" height=\"$height\" viewBox=\"0 0 $width $height\">")
        println(io, "<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>")
        println(io, "<text x=\"$(left)\" y=\"32\" font-family=\"Helvetica, Arial, sans-serif\" font-size=\"24\" font-weight=\"700\">Relative vorticity, day 1</text>")
        println(io, "<text x=\"$(left)\" y=\"54\" font-family=\"Helvetica, Arial, sans-serif\" font-size=\"14\" fill=\"#444444\">$(escape_xml(string(time))) UTC, layer 1, units s^-1</text>")

        println(io, "<rect x=\"$left\" y=\"$top\" width=\"$plot_width\" height=\"$plot_height\" fill=\"#f6f6f6\" stroke=\"#222222\" stroke-width=\"1\"/>")

        for lon_tick in -180:60:180
            x = xmap(lon_tick)
            println(io, "<line x1=\"$x\" y1=\"$top\" x2=\"$x\" y2=\"$(top + plot_height)\" stroke=\"#cfcfcf\" stroke-width=\"1\"/>")
        end
        for lat_tick in -90:30:90
            y = ymap(lat_tick)
            println(io, "<line x1=\"$left\" y1=\"$y\" x2=\"$(left + plot_width)\" y2=\"$y\" stroke=\"#cfcfcf\" stroke-width=\"1\"/>")
        end

        for i in eachindex(lon)
            x0 = xmap(lon_edges[i])
            x1 = xmap(lon_edges[i + 1])
            for j in eachindex(lat)
                y0 = ymap(lat_edges[j])
                y1 = ymap(lat_edges[j + 1])
                color = color_hex(field[i, j], scale)
                println(io, "<rect x=\"$x0\" y=\"$y0\" width=\"$(x1 - x0)\" height=\"$(y1 - y0)\" fill=\"$color\" stroke=\"none\"/>")
            end
        end

        println(io, "<line x1=\"$(xmap(0))\" y1=\"$top\" x2=\"$(xmap(0))\" y2=\"$(top + plot_height)\" stroke=\"#666666\" stroke-width=\"1.2\"/>")
        println(io, "<line x1=\"$left\" y1=\"$(ymap(0))\" x2=\"$(left + plot_width)\" y2=\"$(ymap(0))\" stroke=\"#666666\" stroke-width=\"1.2\"/>")

        for lon_tick in -180:60:180
            x = xmap(lon_tick)
            println(io, "<line x1=\"$x\" y1=\"$(top + plot_height)\" x2=\"$x\" y2=\"$(top + plot_height + 6)\" stroke=\"#222222\" stroke-width=\"1\"/>")
            println(io, "<text x=\"$x\" y=\"$(top + plot_height + 24)\" text-anchor=\"middle\" font-family=\"Helvetica, Arial, sans-serif\" font-size=\"13\">$(format_lon(lon_tick))</text>")
        end

        for lat_tick in -90:30:90
            y = ymap(lat_tick)
            println(io, "<line x1=\"$(left - 6)\" y1=\"$y\" x2=\"$left\" y2=\"$y\" stroke=\"#222222\" stroke-width=\"1\"/>")
            println(io, "<text x=\"$(left - 12)\" y=\"$(y + 4)\" text-anchor=\"end\" font-family=\"Helvetica, Arial, sans-serif\" font-size=\"13\">$(format_lat(lat_tick))</text>")
        end

        println(io, "<text x=\"$(left + plot_width / 2)\" y=\"$(height - 24)\" text-anchor=\"middle\" font-family=\"Helvetica, Arial, sans-serif\" font-size=\"15\">Longitude</text>")
        println(io, "<text x=\"26\" y=\"$(top + plot_height / 2)\" text-anchor=\"middle\" transform=\"rotate(-90 26 $(top + plot_height / 2))\" font-family=\"Helvetica, Arial, sans-serif\" font-size=\"15\">Latitude</text>")

        cb_x = left + plot_width + colorbar_gap
        cb_y = top
        cb_h = plot_height

        for k in 0:255
            frac0 = k / 256
            frac1 = (k + 1) / 256
            y0 = cb_y + (1 - frac1) * cb_h
            y1 = cb_y + (1 - frac0) * cb_h
            value = (2 * frac0 - 1) * scale
            println(io, "<rect x=\"$cb_x\" y=\"$y0\" width=\"$colorbar_width\" height=\"$(y1 - y0 + 1)\" fill=\"$(color_hex(value, scale))\" stroke=\"none\"/>")
        end
        println(io, "<rect x=\"$cb_x\" y=\"$cb_y\" width=\"$colorbar_width\" height=\"$cb_h\" fill=\"none\" stroke=\"#222222\" stroke-width=\"1\"/>")

        for frac in 0:0.25:1
            value = (2 * frac - 1) * scale
            y = cb_y + (1 - frac) * cb_h
            label = @sprintf("%.2e", value)
            println(io, "<line x1=\"$(cb_x + colorbar_width)\" y1=\"$y\" x2=\"$(cb_x + colorbar_width + 6)\" y2=\"$y\" stroke=\"#222222\" stroke-width=\"1\"/>")
            println(io, "<text x=\"$(cb_x + colorbar_width + 12)\" y=\"$(y + 4)\" font-family=\"Helvetica, Arial, sans-serif\" font-size=\"13\">$label</text>")
        end
        println(io, "<text x=\"$(cb_x + colorbar_width / 2)\" y=\"$(cb_y - 12)\" text-anchor=\"middle\" font-family=\"Helvetica, Arial, sans-serif\" font-size=\"14\">s^-1</text>")

        println(io, "</svg>")
    end
end

function main(input_path::AbstractString = DEFAULT_INPUT, output_path::AbstractString = DEFAULT_OUTPUT)
    ds = NCDataset(input_path)
    lon = Float64.(ds["lon"][:])
    lat = Float64.(ds["lat"][:])
    time = ds["time"][DEFAULT_TIME_INDEX]
    slice = ds["vor"][:, :, DEFAULT_LAYER_INDEX, DEFAULT_TIME_INDEX]
    close(ds)

    field = Float64.(coalesce.(slice, NaN))
    lon_centered, field_centered = centered_longitudes(lon, field)

    write_svg(output_path, lon_centered, lat, field_centered, time)
    println("Wrote $output_path")
end

main(length(ARGS) >= 1 ? ARGS[1] : DEFAULT_INPUT, length(ARGS) >= 2 ? ARGS[2] : DEFAULT_OUTPUT)
