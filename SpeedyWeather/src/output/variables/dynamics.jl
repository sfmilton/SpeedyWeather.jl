"""Defines netCDF output of vorticity. Fields are
$(TYPEDFIELDS)

Custom variable output defined similarly with required fields marked,
optional fields otherwise use variable-independent defaults. Initialize with `VorticityOutput()`
and non-default fields can always be passed on as keyword arguments,
e.g. `VorticityOutput(long_name="relative vorticity", compression_level=0)`.
Custom variable output also requires the `path(::MyOutputVariable, simulation)`
to be extended to return the AbstractField subject to output.
Custom element-wise variable transforms, e.g. scale and/or offset to change
units, or even exp(x)/100 to change from log surface pressure to hPa
are passed on as `transform::Function = x -> exp(x)/100`."""
@kwdef mutable struct VorticityOutput <: AbstractOutputVariable

    "[Required] short name of variable (unique) used in netCDF file and key for dictionary"
    name::String = "vor"

    "[Required] unit of variable"
    unit::String = "s^-1"

    "[Required] long name of variable used in netCDF file"
    long_name::String = "relative vorticity"

    "[Required] NetCDF dimensions the variable uses, lon, lat, layer, time"
    dims_xyzt::NTuple{4, Bool} = (true, true, true, true)

    "[Optional] missing value for the variable, if not specified uses NaN"
    missing_value::Float64 = NaN

    "[Optional] compression level of the lossless compressor, 1=lowest/fastest, 9=highest/slowest, 3=default"
    compression_level::Int = 3

    "[Optional] bitshuffle the data for compression, false = default"
    shuffle::Bool = true

    "[Optional] number of mantissa bits to keep for compression (default: 15)"
    keepbits::Int = 5

    "[Optional] Unscale the variable for output? (default: true)"
    unscale::Bool = true
end

"""$TYPEDSIGNATURES To be extended for every output variable to define
the path where in `simulation` to find that output variable `::AbstractField`."""
path(::VorticityOutput, simulation) = simulation.diagnostic_variables.grid.vor_grid

"""Defines netCDF output for a specific variables, see [`VorticityOutput`](@ref) for details.
Fields are: $(TYPEDFIELDS)"""
@kwdef mutable struct ZonalVelocityOutput <: AbstractOutputVariable
    name::String = "u"
    unit::String = "m/s"
    long_name::String = "zonal wind"
    dims_xyzt::NTuple{4, Bool} = (true, true, true, true)
    missing_value::Float64 = NaN
    compression_level::Int = 3
    shuffle::Bool = true
    keepbits::Int = 7
end

path(::ZonalVelocityOutput, simulation) = simulation.diagnostic_variables.grid.u_grid

"""Defines netCDF output for a specific variables, see [`VorticityOutput`](@ref) for details.
Fields are: $(TYPEDFIELDS)"""
@kwdef mutable struct MeridionalVelocityOutput <: AbstractOutputVariable
    name::String = "v"
    unit::String = "m/s"
    long_name::String = "meridional wind"
    dims_xyzt::NTuple{4, Bool} = (true, true, true, true)
    missing_value::Float64 = NaN
    compression_level::Int = 3
    shuffle::Bool = true
    keepbits::Int = 7
end

path(::MeridionalVelocityOutput, simulation) = simulation.diagnostic_variables.grid.v_grid

"""Defines netCDF output for a specific variables, see [`VorticityOutput`](@ref) for details.
Fields are: $(TYPEDFIELDS)"""
@kwdef mutable struct DivergenceOutput <: AbstractOutputVariable
    name::String = "div"
    unit::String = "s^-1"
    long_name::String = "divergence"
    dims_xyzt::NTuple{4, Bool} = (true, true, true, true)
    missing_value::Float64 = NaN
    compression_level::Int = 3
    shuffle::Bool = true
    keepbits::Int = 5
    unscale::Bool = true
end

path(::DivergenceOutput, simulation) = simulation.diagnostic_variables.grid.div_grid

"""Defines netCDF output for a specific variables, see [`VorticityOutput`](@ref) for details.
Fields are: $(TYPEDFIELDS)"""
@kwdef mutable struct InterfaceDisplacementOutput <: AbstractOutputVariable
    name::String = "eta"
    unit::String = "m"
    long_name::String = "interface displacement"
    dims_xyzt::NTuple{4, Bool} = (true, true, false, true)
    missing_value::Float64 = NaN
    compression_level::Int = 3
    shuffle::Bool = true
    keepbits::Int = 7
end

path(::InterfaceDisplacementOutput, simulation) = simulation.diagnostic_variables.grid.pres_grid

"""Defines netCDF output for a specific variables, see [`VorticityOutput`](@ref) for details.
Fields are: $(TYPEDFIELDS)"""
@kwdef mutable struct SurfacePressureOutput{F} <: AbstractOutputVariable
    name::String = "pres"
    unit::String = "hPa"
    long_name::String = "surface pressure"
    dims_xyzt::NTuple{4, Bool} = (true, true, false, true)
    missing_value::Float64 = NaN
    compression_level::Int = 3
    shuffle::Bool = true
    keepbits::Int = 12
    transform::F = (x) -> exp(x) / 100     # log(Pa) to hPa
end

path(::SurfacePressureOutput, simulation) = simulation.diagnostic_variables.grid.pres_grid

"""Defines netCDF output for a specific variables, see [`VorticityOutput`](@ref) for details.
Fields are: $(TYPEDFIELDS)"""
@kwdef mutable struct MeanSeaLevelPressureOutput{F} <: AbstractOutputVariable
    name::String = "mslp"
    unit::String = "hPa"
    long_name::String = "mean sea-level pressure"
    dims_xyzt::NTuple{4, Bool} = (true, true, false, true)
    missing_value::Float64 = NaN
    compression_level::Int = 3
    shuffle::Bool = true
    keepbits::Int = 12
    transform::F = (x) -> exp(x) / 100     # log(Pa) to hPa
end

# points to surface not mean sea level pressure but core variable to read in
path(::MeanSeaLevelPressureOutput, simulation) = simulation.diagnostic_variables.grid.pres_grid

function output!(
        output::NetCDFOutput,
        variable::MeanSeaLevelPressureOutput,
        simulation::AbstractSimulation,
    )
    # escape immediately after first call if variable doesn't have a time dimension
    ~hastime(variable) && output.output_counter > 1 && return nothing

    # get log(surface pressure) field
    lnpₛ = path(variable, simulation)
    h = simulation.model.orography.orography
    (; R_dry, κ) = simulation.model.atmosphere
    g = simulation.model.planet.gravity

    # compute virtual temperature on the fly
    (; nlayers) = simulation.diagnostic_variables
    T = simulation.diagnostic_variables.physics.surface_air_temperature

    # At initial output this field may still be all zeros; fall back to model-level temperature.
    if !simulation.model.physics || all(iszero, T)
        # calculate the surface air temperature from lowest model level temperature
        # via dry adiabatic lapse rate
        T .= field_view(simulation.diagnostic_variables.grid.temp_grid, :, nlayers)
        # σ vertical coordinate at lowest model level
        GPUArrays.@allowscalar σ = simulation.model.geometry.σ_levels_full[nlayers]
        σ⁻ᵏ = σ^(-κ)    # precalculate adiabatic descent factor
        T .*= σ⁻ᵏ       # lower to surface assuming dry adiabatic lapse rate
    end

    q = field_view(simulation.diagnostic_variables.grid.humid_grid, :, nlayers)
    Tᵥ = simulation.diagnostic_variables.dynamics.a_2D_grid

    (; atmosphere) = simulation.model
    Tᵥ .= virtual_temperature.(T, q, atmosphere)

    # calculate mean sea-level pressure on model grid
    mslp = simulation.diagnostic_variables.dynamics.b_2D_grid
    (; transform) = variable                    # to change units from log(Pa) to hPa
    @. mslp = transform(g * h / R_dry / Tᵥ + lnpₛ)    # Pa to hPa

    # interpolate 2D/3D variables
    mslp_output = output.field2D
    mslp_grid = on_architecture(CPU(), mslp)
    RingGrids.interpolate!(mslp_output, mslp_grid, output.interpolator)

    if hasproperty(variable, :keepbits)     # round mantissabits for compression
        round!(mslp_output, variable.keepbits)
    end

    i = output.output_counter               # output time step i to write
    indices = get_indices(i, variable)      # returns (:, :, i) for example, depending on dims
    output.netcdf_file[variable.name][indices...] = mslp_output     # actually write to file
    return nothing
end

"""Defines netCDF output for a specific variables, see [`VorticityOutput`](@ref) for details.
Fields are: $(TYPEDFIELDS)"""
@kwdef mutable struct TemperatureOutput{F} <: AbstractOutputVariable
    name::String = "temp"
    unit::String = "degC"
    long_name::String = "temperature"
    dims_xyzt::NTuple{4, Bool} = (true, true, true, true)
    missing_value::Float64 = NaN
    compression_level::Int = 3
    shuffle::Bool = true
    keepbits::Int = 10
    transform::F = (x) -> x - 273.15     # K to ˚C
end

path(::TemperatureOutput, simulation) = simulation.diagnostic_variables.grid.temp_grid

"""Defines netCDF output for a specific variables, see [`VorticityOutput`](@ref) for details.
Fields are: $(TYPEDFIELDS)"""
@kwdef mutable struct HumidityOutput <: AbstractOutputVariable
    name::String = "humid"
    unit::String = "kg/kg"
    long_name::String = "specific humidity"
    dims_xyzt::NTuple{4, Bool} = (true, true, true, true)
    missing_value::Float64 = NaN
    compression_level::Int = 3
    shuffle::Bool = true
    keepbits::Int = 7
end

path(::HumidityOutput, simulation) = simulation.diagnostic_variables.grid.humid_grid

abstract type AbstractTendencyBudgetOutput <: AbstractOutputVariable end

const TENDENCY_BUDGET_VARIABLES = (
    (:u, "u", "m/s/day", "zonal wind", "ZonalWind"),
    (:v, "v", "m/s/day", "meridional wind", "MeridionalWind"),
    (:temp, "temp", "K/day", "temperature", "Temperature"),
    (:humid, "q", "kg/kg/day", "specific humidity", "Humidity"),
)

const TENDENCY_BUDGET_TERMS = (
    (:physics, "phys", "physics", "Physics"),
    (:dynamics, "dyn", "dynamics", "Dynamics"),
    (:conv, "conv", "convection", "Convective"),
    (:bl, "bl", "boundary-layer processes", "BoundaryLayer"),
    (:vdiff, "vdiff", "vertical diffusion", "VerticalDiffusion"),
    (:lsc, "lsc", "large-scale condensation", "Condensation"),
    (:sw, "sw", "shortwave radiation", "Shortwave"),
    (:lw, "lw", "longwave radiation", "Longwave"),
    (:smf, "smf", "surface momentum flux", "SurfaceMomentum"),
    (:shf, "shf", "surface sensible heat flux", "SurfaceSensibleHeat"),
    (:shuf, "shuf", "surface humidity flux", "SurfaceHumidity"),
    (:stoch, "stoch", "stochastic physics", "Stochastic"),
)

const TENDENCY_BUDGET_OUTPUT_SPECS = let specs = Tuple{Symbol, String, String, String, Symbol}[]
    for (varsymbol, short_name, unit, long_var_name, var_type_name) in TENDENCY_BUDGET_VARIABLES
        for (term_symbol, term_short, term_long, term_type_name) in TENDENCY_BUDGET_TERMS
            typename = Symbol(term_type_name, var_type_name, "TendencyOutput")
            name = string(short_name, "_tend_", term_short)
            long_name = string(long_var_name, " tendency from ", term_long)
            field = Symbol(varsymbol, :_tend_, term_symbol, :_grid)
            push!(specs, (typename, name, unit, long_name, field))
        end
    end
    Tuple(specs)
end

for spec in TENDENCY_BUDGET_OUTPUT_SPECS
    typename, name, unit, long_name, field = spec
    @eval begin
        @kwdef mutable struct $typename <: AbstractTendencyBudgetOutput
            name::String = $name
            unit::String = $unit
            long_name::String = $long_name
            dims_xyzt::NTuple{4, Bool} = (true, true, true, true)
            missing_value::Float64 = NaN
            compression_level::Int = 3
            shuffle::Bool = true
            keepbits::Int = 8
        end

        path(::$typename, simulation) =
            simulation.diagnostic_variables.tendencies.$field
    end
end

function output!(
        output::NetCDFOutput,
        variable::AbstractTendencyBudgetOutput,
        simulation::AbstractSimulation,
    )
    # escape immediately after first call if variable doesn't have a time dimension
    ~hastime(variable) && output.output_counter > 1 && return nothing

    # interpolate tendency on output grid
    tend = output.field3D
    raw = on_architecture(CPU(), path(variable, simulation))
    RingGrids.interpolate!(tend, raw, output.interpolator)

    # convert from radius-scaled internal tendency to physical K/day
    radius = simulation.model.planet.radius
    tend .*= (86_400 / radius)

    if hasproperty(variable, :keepbits)
        round!(tend, variable.keepbits)
    end

    i = output.output_counter
    indices = get_indices(i, variable)
    output.netcdf_file[variable.name][indices...] = tend
    return nothing
end

tendency_budget_calls = [:( $(spec[1])() ) for spec in TENDENCY_BUDGET_OUTPUT_SPECS]
@eval TendencyBudgetOutput() = ($(tendency_budget_calls...),)

# collect all in one for convenience
DynamicsOutput() = (
    VorticityOutput(),
    ZonalVelocityOutput(),
    MeridionalVelocityOutput(),
    DivergenceOutput(),
    InterfaceDisplacementOutput(),
    SurfacePressureOutput(),
    MeanSeaLevelPressureOutput(),
    TemperatureOutput(),
    HumidityOutput(),
)
