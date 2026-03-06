#!/usr/bin/env julia

import Pkg

project_dir = joinpath(@__DIR__, "..")
Pkg.activate(project_dir)
Pkg.instantiate()

using Dates
using SpeedyWeather

function main()
    spectral_grid = SpectralGrid(trunc = 63, nlayers = 1)
    output = NetCDFOutput(
        spectral_grid,
        Barotropic,
        output_dt = Day(1),
        path = @__DIR__,
        id = "barotropic",
    )
    delete!(output, :u, :v)

    model = BarotropicModel(spectral_grid; output)
    simulation = initialize!(model)

    period = Day(20)
    println("Starting barotropic run.")
    println("Writing daily relative vorticity to NetCDF.")
    run!(simulation, period = period, output = true)
    println("Finished barotropic run.")
    println("Output file: $(joinpath(simulation.model.output.run_path, simulation.model.output.filename))")

    return simulation
end

main()
