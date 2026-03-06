#!/usr/bin/env julia

import Pkg

project_dir = joinpath(@__DIR__, "..")
Pkg.activate(project_dir)
Pkg.instantiate()

using Dates
using SpeedyWeather

struct AbortOnNaN <: SpeedyWeather.AbstractCallback end
SpeedyWeather.initialize!(::AbortOnNaN, args...) = nothing
SpeedyWeather.finalize!(::AbortOnNaN, args...) = nothing
function SpeedyWeather.callback!(::AbortOnNaN, progn, diagn, model)
    model.feedback.nans_detected || return nothing
    elapsed_ms = Dates.value(progn.clock.time - progn.clock.start)
    h = round(elapsed_ms / (1000 * 60 * 60); digits = 2)
    error("NaN/Inf detected in run at T+$(h)h. Aborting early.")
end

function main()
    spectral_grid = SpectralGrid(trunc = 63, nlayers = 20)
    time_stepping = Leapfrog(spectral_grid, Δt_at_T31 = Minute(20))

    ic = StartFromFile(spectral_grid;
        path = "SpeedyWeather/experiments",
        run_folder = "run_primitive_wet_t63_l20_full_physics_0004",  # change to your finished run folder
        filename = "restart.jld2",
    )    
    output = NetCDFOutput(
        spectral_grid,
        PrimitiveWet,
        output_dt = Hour(6),
        path = @__DIR__,
        id = "primitive_wet_t63_l20_full_physics",
    )
    add!(output, SpeedyWeather.SurfacePressureOutput())
    add!(output, SpeedyWeather.SurfaceFluxesOutput()...)
    add!(output, SpeedyWeather.RadiationOutput()...)
    add!(output, SpeedyWeather.PrecipitationOutput()...)
    add!(output, SpeedyWeather.TendencyBudgetOutput()...)
    add!(output, SpeedyWeather.BoundaryLayerOutput()...)  # includes tsurf, u10, v10, bld
    add!(output, SpeedyWeather.OceanOutput()...)          # includes sst, sic
    add!(output, SpeedyWeather.LandOutput()...)

    model = PrimitiveWetModel(spectral_grid; physics = true, output, time_stepping, initial_conditions=ic)
    add!(model.callbacks, :abort_on_nan => AbortOnNaN())
    simulation = initialize!(model)

    period = Day(10)
    println("Starting PrimitiveWet run at T63/L20 with full physics.")
    println("Using Leapfrog with Δt_at_T31 = Minute(20) (about 10 minutes at T63).")
    println("Writing 6-hourly output to NetCDF (including surface pressure 'pres').")
    run!(simulation, period = period, output = true)
    println("Finished PrimitiveWet run.")
    println("Output file: $(joinpath(simulation.model.output.run_path, simulation.model.output.filename))")

    return simulation
end

main()
