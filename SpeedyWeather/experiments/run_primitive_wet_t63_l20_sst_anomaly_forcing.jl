#!/usr/bin/env julia

import Pkg

project_dir = joinpath(@__DIR__, "..")
Pkg.activate(project_dir)
Pkg.instantiate()

using Dates
using SpeedyWeather

const DEFAULT_OUTPUT_ID = "primitive_wet_t63_l20_sst_anomaly"

Base.@kwdef mutable struct Config
    start_date::DateTime = DateTime(1979, 1, 1)
    period_days::Int = 10
    trunc::Int = 63
    nlayers::Int = 20
    output_dt_hours::Int = 6
    sst_forcing::String = "anomaly"
    output_id::String = DEFAULT_OUTPUT_ID
end

function print_usage()
    println(
        """
Usage:
  julia experiments/run_primitive_wet_t63_l20_sst_anomaly_forcing.jl [options]

Options:
  --start-date YYYY-MM-DD  Simulation start date (default: 1979-01-01)
  --period-days N          Run length in days (default: 10)
  --trunc N                Spectral truncation (default: 63)
  --nlayers N              Number of vertical sigma layers (default: 20)
  --output-dt-hours N      NetCDF output interval in hours (default: 6)
  --sst-forcing MODE       SST forcing mode: anomaly or climatology (default: anomaly)
  --output-id ID           Output/run identifier (default: $(DEFAULT_OUTPUT_ID))
"""
    )
end

function parse_start_date(s::String)
    try
        return DateTime(Date(s))
    catch err
        error("Invalid --start-date '$(s)'. Use YYYY-MM-DD. Original error: $(err)")
    end
end

function parse_args(args::Vector{String})
    cfg = Config()
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "-h" || arg == "--help"
            print_usage()
            exit(0)
        elseif arg == "--start-date"
            i < length(args) || error("Option $(arg) requires a value.")
            cfg.start_date = parse_start_date(args[i + 1])
            i += 2
        elseif arg == "--period-days"
            i < length(args) || error("Option $(arg) requires a value.")
            cfg.period_days = parse(Int, args[i + 1])
            i += 2
        elseif arg == "--trunc"
            i < length(args) || error("Option $(arg) requires a value.")
            cfg.trunc = parse(Int, args[i + 1])
            i += 2
        elseif arg == "--nlayers"
            i < length(args) || error("Option $(arg) requires a value.")
            cfg.nlayers = parse(Int, args[i + 1])
            i += 2
        elseif arg == "--output-dt-hours"
            i < length(args) || error("Option $(arg) requires a value.")
            cfg.output_dt_hours = parse(Int, args[i + 1])
            i += 2
        elseif arg == "--sst-forcing"
            i < length(args) || error("Option $(arg) requires a value.")
            cfg.sst_forcing = lowercase(strip(args[i + 1]))
            i += 2
        elseif arg == "--output-id"
            i < length(args) || error("Option $(arg) requires a value.")
            cfg.output_id = args[i + 1]
            i += 2
        else
            error("Unknown option: $(arg). Use --help for usage.")
        end
    end

    cfg.period_days >= 0 || error("--period-days must be >= 0")
    cfg.trunc > 0 || error("--trunc must be > 0")
    cfg.nlayers > 0 || error("--nlayers must be > 0")
    cfg.output_dt_hours > 0 || error("--output-dt-hours must be > 0")
    cfg.sst_forcing in ("anomaly", "climatology") ||
        error("--sst-forcing must be either 'anomaly' or 'climatology'")

    return cfg
end

function build_output(spectral_grid::SpectralGrid, cfg::Config)
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
    add!(output, SpeedyWeather.BoundaryLayerOutput()...)
    add!(output, SpeedyWeather.OceanOutput()...)
    add!(output, SpeedyWeather.LandOutput()...)
    add!(output, SpeedyWeather.GeopotentialOutput())
    return output
end

function main(args::Vector{String} = ARGS)
    cfg = parse_args(args)

    spectral_grid = SpectralGrid(trunc = cfg.trunc, nlayers = cfg.nlayers)
    time_stepping = Leapfrog(spectral_grid, Δt_at_T31 = Minute(20))
    ocean = cfg.sst_forcing == "anomaly" ?
        SeasonalOceanClimatologyAnomaly(spectral_grid) :
        SeasonalOceanClimatology(spectral_grid)
    output = build_output(spectral_grid, cfg)

    model = PrimitiveWetModel(spectral_grid; physics = true, ocean, output, time_stepping)
    simulation = initialize!(model, time = cfg.start_date)

    println("Starting PrimitiveWet run at T$(cfg.trunc)/L$(cfg.nlayers) with SST $(cfg.sst_forcing) forcing.")
    println("Start date: $(Dates.format(cfg.start_date, "yyyy-mm-dd"))")
    println("Run length: $(cfg.period_days) days")
    println("Output interval: $(cfg.output_dt_hours) hours")

    run!(simulation, period = Day(cfg.period_days), output = true)

    println("Finished PrimitiveWet run.")
    println("Output file: $(joinpath(simulation.model.output.run_path, simulation.model.output.filename))")
    return simulation
end

main()
