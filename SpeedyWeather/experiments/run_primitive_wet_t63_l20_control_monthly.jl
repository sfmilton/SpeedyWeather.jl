#!/usr/bin/env julia

import Pkg

project_dir = joinpath(@__DIR__, "..")
Pkg.activate(project_dir)
Pkg.instantiate()

using Dates
using Printf
using SpeedyWeather

const DEFAULT_OUTPUT_ID = "primitive_wet_t63_l20_control_1979_1981"

Base.@kwdef mutable struct Config
    spinup_start::DateTime = DateTime(1979, 11, 1)
    control_start::DateTime = DateTime(1980, 1, 1)
    control_end::DateTime = DateTime(1981, 12, 31)

    # If true: start directly at control period (or from --restart-file).
    no_spinup::Bool = false

    # Optional full path to a restart.jld2-like file.
    restart_file::String = ""

    trunc::Int = 63
    nlayers::Int = 20
    output_dt_hours::Int = 6
    sst_forcing::String = "climatology"
    output_id::String = DEFAULT_OUTPUT_ID

    # Parent directory for master run folders.
    output_parent::String = @__DIR__

    # Optional explicit master run directory. If empty, create run_<id>_####.
    run_dir::String = ""
end

function print_usage()
    println(
        """
Usage:
  julia experiments/run_primitive_wet_t63_l20_control_monthly.jl [options]

Default behavior:
  - Spinup: Nov 1979 to end Dec 1979
  - Control: Jan 1980 to end Dec 1981
  - Monthly output/restart files in one master run directory

Options:
  --spinup-start-date YYYY-MM-DD   Spinup start date (month start, default: 1979-11-01)
  --control-start-date YYYY-MM-DD  Control start date (month start, default: 1980-01-01)
  --control-end-date YYYY-MM-DD    Control end date (inclusive month, default: 1981-12-31)
  --no-spinup                      Skip spinup phase and start from control period
  --restart-file PATH              Start from a specific restart file (typically with --no-spinup)

  --trunc N                        Spectral truncation (default: 63)
  --nlayers N                      Number of vertical sigma layers (default: 20)
  --output-dt-hours N              NetCDF output interval in hours (default: 6)
  --sst-forcing MODE               SST forcing mode: anomaly or climatology (default: climatology)
  --output-id ID                   Master run identifier (default: $(DEFAULT_OUTPUT_ID))
  --output-parent PATH             Parent directory for master run folders (default: experiments/)
  --run-dir PATH                   Explicit master run directory (optional)

Examples:
  # Full spinup + control (default window)
  julia experiments/run_primitive_wet_t63_l20_control_monthly.jl

  # Continue control from a named restart file
  julia experiments/run_primitive_wet_t63_l20_control_monthly.jl \\
      --no-spinup --restart-file /path/to/restart_197912.jld2
"""
    )
end

function parse_date_ymd(s::String; option_name::String)
    try
        return DateTime(Date(s))
    catch err
        error("Invalid $(option_name) value '$(s)'. Use YYYY-MM-DD. Original error: $(err)")
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
        elseif arg == "--spinup-start-date"
            i < length(args) || error("Option $(arg) requires a value.")
            cfg.spinup_start = parse_date_ymd(args[i + 1], option_name = arg)
            i += 2
        elseif arg == "--control-start-date"
            i < length(args) || error("Option $(arg) requires a value.")
            cfg.control_start = parse_date_ymd(args[i + 1], option_name = arg)
            i += 2
        elseif arg == "--control-end-date"
            i < length(args) || error("Option $(arg) requires a value.")
            cfg.control_end = parse_date_ymd(args[i + 1], option_name = arg)
            i += 2
        elseif arg == "--no-spinup"
            cfg.no_spinup = true
            i += 1
        elseif arg == "--restart-file"
            i < length(args) || error("Option $(arg) requires a value.")
            cfg.restart_file = args[i + 1]
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
        elseif arg == "--output-parent"
            i < length(args) || error("Option $(arg) requires a value.")
            cfg.output_parent = args[i + 1]
            i += 2
        elseif arg == "--run-dir"
            i < length(args) || error("Option $(arg) requires a value.")
            cfg.run_dir = args[i + 1]
            i += 2
        else
            error("Unknown option: $(arg). Use --help for usage.")
        end
    end

    cfg.trunc > 0 || error("--trunc must be > 0")
    cfg.nlayers > 0 || error("--nlayers must be > 0")
    cfg.output_dt_hours > 0 || error("--output-dt-hours must be > 0")
    cfg.sst_forcing in ("anomaly", "climatology") ||
        error("--sst-forcing must be either 'anomaly' or 'climatology'")

    # Require month-aligned spinup/control starts.
    Dates.day(cfg.spinup_start) == 1 || error("--spinup-start-date must be first day of month.")
    Dates.day(cfg.control_start) == 1 || error("--control-start-date must be first day of month.")
    Dates.hour(cfg.spinup_start) == 0 || error("--spinup-start-date must be midnight.")
    Dates.hour(cfg.control_start) == 0 || error("--control-start-date must be midnight.")
    cfg.control_end >= cfg.control_start || error("--control-end-date must be >= --control-start-date")

    if !isempty(cfg.restart_file)
        cfg.restart_file = abspath(cfg.restart_file)
        isfile(cfg.restart_file) || error("--restart-file does not exist: $(cfg.restart_file)")
        cfg.no_spinup || error("--restart-file currently requires --no-spinup to avoid ambiguous phase setup.")
    end

    return cfg
end

function first_day_next_month(dt::DateTime)
    d = Date(dt)
    return DateTime(Dates.firstdayofmonth(d + Month(1)))
end

function control_end_exclusive(cfg::Config)
    d = Date(cfg.control_end)
    return DateTime(Dates.firstdayofmonth(d + Month(1)))
end

function create_master_run_dir(cfg::Config)
    if !isempty(cfg.run_dir)
        run_dir = abspath(cfg.run_dir)
        mkpath(run_dir)
        return run_dir
    end

    parent = abspath(cfg.output_parent)
    mkpath(parent)
    prefix = "run_$(cfg.output_id)_"
    for n in 1:9999
        candidate = joinpath(parent, string(prefix, @sprintf("%04d", n)))
        if !ispath(candidate)
            mkdir(candidate)
            return candidate
        end
    end
    error("Could not allocate a master run directory under $(parent).")
end

function build_output(
        spectral_grid::SpectralGrid,
        cfg::Config,
        master_run_dir::String,
    )
    output = NetCDFOutput(
        spectral_grid,
        PrimitiveWet,
        output_dt = Hour(cfg.output_dt_hours),
        path = master_run_dir,
        id = "segment",
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

function restart_initial_conditions(spectral_grid::SpectralGrid, restart_path::String)
    return StartFromFile(
        spectral_grid;
        path = dirname(restart_path),
        run_folder = ".",
        filename = basename(restart_path),
    )
end

function archive_monthly_files!(
        simulation,
        master_run_dir::String,
        phase::String,
        month_tag::String,
    )
    segment_path = simulation.model.output.run_path
    src_nc = joinpath(segment_path, simulation.model.output.filename)
    src_restart = joinpath(segment_path, "restart.jld2")
    src_progress = joinpath(segment_path, "progress.txt")
    src_parameters = joinpath(segment_path, "parameters.txt")

    dst_nc = joinpath(master_run_dir, "$(phase)_$(month_tag).nc")
    dst_restart = joinpath(master_run_dir, "$(phase)_$(month_tag).restart.jld2")
    dst_progress = joinpath(master_run_dir, "$(phase)_$(month_tag).progress.txt")
    dst_parameters = joinpath(master_run_dir, "$(phase)_$(month_tag).parameters.txt")

    isfile(src_nc) || error("Expected monthly output file not found: $(src_nc)")
    isfile(src_restart) || error("Expected monthly restart file not found: $(src_restart)")

    mv(src_nc, dst_nc; force = true)
    mv(src_restart, dst_restart; force = true)
    isfile(src_progress) && mv(src_progress, dst_progress; force = true)
    isfile(src_parameters) && mv(src_parameters, dst_parameters; force = true)

    return dst_restart, dst_nc
end

function build_model_and_initialize(
        cfg::Config,
        spectral_grid::SpectralGrid,
        time_stepping,
        master_run_dir::String;
        restart_path::Union{Nothing, String} = nothing,
        fresh_start::DateTime,
    )
    ocean = cfg.sst_forcing == "anomaly" ?
        SeasonalOceanClimatologyAnomaly(spectral_grid) :
        SeasonalOceanClimatology(spectral_grid)
    output = build_output(spectral_grid, cfg, master_run_dir)

    if isnothing(restart_path)
        model = PrimitiveWetModel(spectral_grid; physics = true, ocean, output, time_stepping)
        simulation = initialize!(model, time = fresh_start)
    else
        ic = restart_initial_conditions(spectral_grid, restart_path)
        model = PrimitiveWetModel(spectral_grid; physics = true, ocean, output, time_stepping, initial_conditions = ic)
        simulation = initialize!(model)  # keep timestamp from restart file
    end

    return simulation
end

function write_run_summary(cfg::Config, master_run_dir::String)
    summary_path = joinpath(master_run_dir, "run_summary.txt")
    open(summary_path, "w") do io
        println(io, "SpeedyWeather monthly control configuration")
        println(io, "created_utc = ", Dates.format(Dates.now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SS"))
        println(io, "spinup_start = ", cfg.spinup_start)
        println(io, "control_start = ", cfg.control_start)
        println(io, "control_end = ", cfg.control_end)
        println(io, "no_spinup = ", cfg.no_spinup)
        println(io, "restart_file = ", isempty(cfg.restart_file) ? "(none)" : cfg.restart_file)
        println(io, "trunc = ", cfg.trunc)
        println(io, "nlayers = ", cfg.nlayers)
        println(io, "output_dt_hours = ", cfg.output_dt_hours)
        println(io, "sst_forcing = ", cfg.sst_forcing)
        println(io, "output_id = ", cfg.output_id)
        println(io, "output_parent = ", abspath(cfg.output_parent))
        println(io, "run_dir = ", master_run_dir)
    end
    return summary_path
end

function main(args::Vector{String} = ARGS)
    cfg = parse_args(args)
    master_run_dir = create_master_run_dir(cfg)
    summary_path = write_run_summary(cfg, master_run_dir)

    spectral_grid = SpectralGrid(trunc = cfg.trunc, nlayers = cfg.nlayers)
    time_stepping = Leapfrog(spectral_grid, Δt_at_T31 = Minute(20))

    final_stop = control_end_exclusive(cfg)

    # Initial state source:
    # 1) restart file if provided
    # 2) fresh IC at control_start when --no-spinup
    # 3) fresh IC at spinup_start otherwise
    restart_path = isempty(cfg.restart_file) ? nothing : cfg.restart_file
    fresh_start = cfg.no_spinup ? cfg.control_start : cfg.spinup_start

    println("Master run directory: $(master_run_dir)")
    println("Run summary: $(summary_path)")
    println("SST forcing: $(cfg.sst_forcing)")
    println("Time window end (exclusive): $(final_stop)")
    println("Mode: ", cfg.no_spinup ? "control-only" : "spinup+control")
    !isnothing(restart_path) && println("Starting from restart: $(restart_path)")

    month_counter = 0
    while true
        simulation = build_model_and_initialize(
            cfg,
            spectral_grid,
            time_stepping,
            master_run_dir;
            restart_path,
            fresh_start,
        )

        start_time = simulation.prognostic_variables.clock.time
        start_time >= final_stop && break

        if cfg.no_spinup && start_time < cfg.control_start
            error(
                "Restart time $(start_time) is before control_start $(cfg.control_start), " *
                "but --no-spinup was requested."
            )
        end

        phase = start_time < cfg.control_start ? "spinup" : "control"
        month_tag = Dates.format(Date(start_time), "yyyymm")
        stop_time = min(first_day_next_month(start_time), final_stop)
        period = stop_time - start_time
        period > Second(0) || break

        println("\n=== Segment $(month_counter + 1) ===")
        println("Phase: $(phase)")
        println("Start: $(start_time)")
        println("Stop:  $(stop_time)")
        println("Period: $(period)")

        run!(simulation, period = period, output = true)
        restart_path, monthly_nc = archive_monthly_files!(simulation, master_run_dir, phase, month_tag)
        println("Monthly NetCDF: $(monthly_nc)")
        println("Monthly restart: $(restart_path)")

        fresh_start = stop_time
        month_counter += 1
    end

    println("\nFinished monthly control workflow.")
    println("Segments completed: $(month_counter)")
    println("Outputs in: $(master_run_dir)")
    return nothing
end

main()
