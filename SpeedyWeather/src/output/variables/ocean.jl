"""Defines netCDF output for a specific variables, see [`VorticityOutput`](@ref) for details.
Fields are: $(TYPEDFIELDS)"""
@kwdef mutable struct SeaSurfaceTemperatureOutput{F} <: AbstractOutputVariable
    name::String = "sst"
    unit::String = "degC"
    long_name::String = "sea surface temperature"
    dims_xyzt::NTuple{4, Bool} = (true, true, false, true)
    missing_value::Float64 = NaN
    compression_level::Int = 3
    shuffle::Bool = true
    keepbits::Int = 10
    transform::F = (x) -> x - 273.15
end

path(::SeaSurfaceTemperatureOutput, simulation) =
    simulation.prognostic_variables.ocean.sea_surface_temperature

"""Defines netCDF output of SST anomaly (for anomaly-forced ocean runs).
Fields are: $(TYPEDFIELDS)"""
@kwdef mutable struct SeaSurfaceTemperatureAnomalyOutput <: AbstractOutputVariable
    name::String = "ssta"
    unit::String = "degC"
    long_name::String = "sea surface temperature anomaly"
    dims_xyzt::NTuple{4, Bool} = (true, true, false, true)
    missing_value::Float64 = NaN
    compression_level::Int = 3
    shuffle::Bool = true
    keepbits::Int = 10
end

function output!(
        output::NetCDFOutput,
        variable::SeaSurfaceTemperatureAnomalyOutput,
        simulation::AbstractSimulation,
    )
    # escape immediately after first call if variable doesn't have a time dimension
    ~hastime(variable) && output.output_counter > 1 && return nothing

    ssta = output.field2D

    if simulation.model.ocean isa SeasonalOceanClimatologyAnomaly
        ocean = simulation.model.ocean
        time = simulation.prognostic_variables.clock.time
        NF = eltype(ssta)

        this_anomaly, next_anomaly, weight_anomaly = anomaly_interpolation_indices(
            NF, time, ocean.anomaly_time; clamp_outside_range = ocean.clamp_outside_range
        )

        # Reconstruct the anomaly forcing used in the ocean timestep.
        anomaly_this = field_view(ocean.monthly_anomaly, :, this_anomaly)
        if this_anomaly == next_anomaly
            simulation.diagnostic_variables.dynamics.a_2D_grid .= anomaly_this
        else
            anomaly_next = field_view(ocean.monthly_anomaly, :, next_anomaly)
            simulation.diagnostic_variables.dynamics.a_2D_grid .=
                (1 - weight_anomaly) .* anomaly_this .+ weight_anomaly .* anomaly_next
        end

        raw = on_architecture(CPU(), simulation.diagnostic_variables.dynamics.a_2D_grid)
        RingGrids.interpolate!(ssta, raw, output.interpolator)
    else
        ssta .= variable.missing_value
    end

    if hasproperty(variable, :keepbits)     # round mantissa bits for compression
        round!(ssta, variable.keepbits)
    end

    i = output.output_counter
    indices = get_indices(i, variable)
    output.netcdf_file[variable.name][indices...] = ssta
    return nothing
end

"""Defines netCDF output for a specific variables, see [`VorticityOutput`](@ref) for details.
Fields are: $(TYPEDFIELDS)"""
@kwdef mutable struct SeaIceConcentrationOutput <: AbstractOutputVariable
    name::String = "sic"
    unit::String = "m^2/m^2"
    long_name::String = "sea ice concentration"
    dims_xyzt::NTuple{4, Bool} = (true, true, false, true)
    missing_value::Float64 = NaN
    compression_level::Int = 3
    shuffle::Bool = true
    keepbits::Int = 10
end

path(::SeaIceConcentrationOutput, simulation) =
    simulation.prognostic_variables.ocean.sea_ice_concentration

OceanOutput() = (
    SeaSurfaceTemperatureOutput(),
    SeaSurfaceTemperatureAnomalyOutput(),
    SeaIceConcentrationOutput(),
)
