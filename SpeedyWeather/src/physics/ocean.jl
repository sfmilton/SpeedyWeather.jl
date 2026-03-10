"""
Abstract super type for ocean models, which control the sea surface temperature
and sea ice concentration as boundary conditions to a SpeedyWeather simulation.
A new ocean model has to be defined as

    CustomOceanModel <: AbstractOcean

and can have parameters like `CustomOceanModel{T}` and fields. They need to extend
the following functions

    function initialize!(ocean_model::CustomOceanModel, model::PrimitiveEquation)
        # your code here to initialize the ocean model itself
        # you can use other fields from model, e.g. model.geometry
    end

    function initialize!(
        ocean,
        progn::PrognosticVariables,
        diagn::DiagnosticVariables,
        ocean_model::CustomOceanModel,
        model::PrimitiveEquation,
    )
        # your code here to initialize the prognostic variables for the ocean
        # namely, ocean.sea_surface_temperature, ocean.sea_ice_concentration, e.g.
        # ocean.sea_surface_temperature .= 300      # 300K everywhere
    end

    function timestep!(
        progn::PrognosticVariables,
        diagn::DiagnosticVariables,
        ocean_model::CustomOceanModel,
        model::PrimitiveEquation,
    )
        # your code here to change the progn.ocean.sea_surface_temperature
    end

Temperatures in ocean.sea_surface_temperature have units of Kelvin,
or NaN for no ocean. Note that neither sea surface temperature, land-sea mask
or orography have to agree. It is possible to have an ocean on top of a mountain.
For an ocean grid-cell that is (partially) masked by the land-sea mask, its value will
be (fractionally) ignored in the calculation of surface fluxes (potentially leading
to a zero flux depending on land surface temperatures). For an ocean grid-cell that is NaN
but not masked by the land-sea mask, its value is always ignored.
"""
abstract type AbstractOcean <: AbstractModelComponent end

function Base.show(io::IO, O::AbstractOcean)
    println(io, "$(typeof(O)) <: AbstractOcean")
    keys = propertynames(O)
    return print_fields(io, O, keys)
end

# variable that AbstractOcean requires
variables(::AbstractOcean) =
    (
    PrognosticVariable(
        name = :sea_surface_temperature, dims = Grid2D(),
        namespace = :ocean, units = "K", desc = "Sea surface temperature"
    ),
)

# function barrier for all oceans
function initialize!(
        ocean::PrognosticVariablesOcean,
        progn::PrognosticVariables,
        diagn::DiagnosticVariables,
        ocean_model::AbstractOcean,
        model::PrimitiveEquation
    ) where {PrognosticVariablesOcean}
    initialize!(ocean, progn, diagn, ocean_model, model)
    return initialize!(ocean, progn, diagn, model.sea_ice, model)
end

# function barrier for all oceans
function ocean_timestep!(
        progn::PrognosticVariables,
        diagn::DiagnosticVariables,
        model::PrimitiveEquation
    )
    return timestep!(progn, diagn, model.ocean, model)
end

const OCEAN_LON_DIM_NAMES = ("lon", "longitude", "x")
const OCEAN_LAT_DIM_NAMES = ("lat", "latitude", "y")
const OCEAN_TIME_DIM_NAMES = ("time", "t")

@inline canonical_dim_name(name::AbstractString) = lowercase(replace(String(name), "_" => ""))

function find_dim_index(dim_names, expected_names::Tuple{Vararg{String}})
    for (i, name) in pairs(dim_names)
        canonical_dim_name(name) in expected_names && return i
    end
    return nothing
end

function ocean_data_path(path::String, file::String)
    return path == "SpeedyWeather.jl/input_data" ?
        joinpath(@__DIR__, "../../input_data", file) :
        joinpath(path, file)
end

function load_ocean_variable_3d(ncfile::NCDataset, varname::String)
    variable = ncfile[varname]
    dim_names = String.(NCDatasets.dimnames(variable))

    i_lon = find_dim_index(dim_names, OCEAN_LON_DIM_NAMES)
    i_lat = find_dim_index(dim_names, OCEAN_LAT_DIM_NAMES)
    i_time = find_dim_index(dim_names, OCEAN_TIME_DIM_NAMES)

    if isnothing(i_lon) || isnothing(i_lat) || isnothing(i_time)
        throw(ArgumentError("Cannot identify lon/lat/time dimensions for '$varname'. " *
            "Dimensions are $(Tuple(dim_names))."))
    end

    data = variable.var[:, :, :]
    perm = (i_lon, i_lat, i_time)
    return perm == (1, 2, 3) ? data : permutedims(data, perm)
end

function load_time_axis(ncfile::NCDataset, time_varname::String)
    times_raw = ncfile[time_varname][:]
    times = DateTime[]
    for time in times_raw
        if ismissing(time)
            continue
        elseif time isa DateTime
            push!(times, time)
        elseif time isa Dates.Date
            push!(times, DateTime(time))
        else
            throw(ArgumentError("Time coordinate '$time_varname' must decode to DateTime."))
        end
    end

    isempty(times) && throw(ArgumentError("Time coordinate '$time_varname' is empty."))
    issorted(times) || throw(ArgumentError("Time coordinate '$time_varname' is not sorted."))
    return times
end

@inline function month_interpolation_weight(::Type{NF}, time::DateTime) where {NF}
    # Keep monthly interpolation daily-resolved for consistency with existing climatology behavior.
    midnight = DateTime(Dates.Date(time))
    return convert(NF, Dates.days(midnight - Dates.firstdayofmonth(midnight)) / Dates.daysinmonth(midnight))
end

function anomaly_interpolation_indices(
        ::Type{NF},
        time::DateTime,
        anomaly_time::AbstractVector{DateTime};
        clamp_outside_range::Bool = true,
    ) where {NF}
    ntime = length(anomaly_time)
    ntime > 0 || throw(ArgumentError("Anomaly time axis is empty."))
    ntime == 1 && return 1, 1, zero(NF)

    # Update anomaly forcing daily.
    midnight = DateTime(Dates.Date(time))

    if midnight <= anomaly_time[1]
        clamp_outside_range || throw(ArgumentError("Time $time is before anomaly range $(anomaly_time[1])."))
        return 1, 1, zero(NF)
    elseif midnight >= anomaly_time[end]
        clamp_outside_range || throw(ArgumentError("Time $time is after anomaly range $(anomaly_time[end])."))
        return ntime, ntime, zero(NF)
    end

    next_idx = searchsortedfirst(anomaly_time, midnight)

    if anomaly_time[next_idx] == midnight
        return next_idx, next_idx, zero(NF)
    end

    this_idx = next_idx - 1
    total = Millisecond(anomaly_time[next_idx] - anomaly_time[this_idx]).value
    elapsed = Millisecond(midnight - anomaly_time[this_idx]).value
    weight = convert(NF, elapsed / total)
    return this_idx, next_idx, weight
end


## SEASONAL OCEAN CLIMATOLOGY
export SeasonalOceanClimatology

"""
Seasonal ocean climatology that reads monthly sea surface temperature
fields from file, and interpolates them in time on every time step
and writes them to the prognostic variables.
Fields and options are
$(TYPEDFIELDS)"""
@kwdef struct SeasonalOceanClimatology{NF, Grid, GridVariable3D} <: AbstractOcean

    "Grid used for the model"
    grid::Grid

    "[OPTION] Path to the folder containing the sea surface temperatures, pkg path default"
    path::String = "SpeedyWeather.jl/input_data"

    "[OPTION] Filename of sea surface temperatures"
    file::String = "sea_surface_temperature.nc"

    "[OPTION] Variable name in netcdf file"
    varname::String = "sst"

    "[OPTION] Grid the sea surface temperature file comes on"
    file_Grid::Type{<:AbstractGrid} = FullGaussianGrid

    "[OPTION] The missing value in the data respresenting land"
    missing_value::NF = NaN

    # to be filled from file
    "Monthly sea surface temperatures [K], interpolated onto Grid"
    monthly_temperature::GridVariable3D = zeros(GridVariable3D, grid, 12)
end

# generator function
function SeasonalOceanClimatology(SG::SpectralGrid; kwargs...)
    (; NF, GridVariable3D, grid) = SG
    return SeasonalOceanClimatology{NF, typeof(grid), GridVariable3D}(; grid, kwargs...)
end

function initialize!(ocean::SeasonalOceanClimatology, model::PrimitiveEquation)
    (; monthly_temperature) = ocean

    # LOAD NETCDF FILE
    path = ocean_data_path(ocean.path, ocean.file)
    ncfile = NCDataset(path)

    # create interpolator from grid in file to grid used in model
    fill_value = ncfile[ocean.varname].attrib["_FillValue"]
    sst = ocean.file_Grid(load_ocean_variable_3d(ncfile, ocean.varname), input_as = Matrix)
    sst[sst .=== fill_value] .= ocean.missing_value      # === to include NaN

    # transfer to architecture of model if needed
    sst = on_architecture(model.architecture, sst)

    @boundscheck fields_match(monthly_temperature, sst, vertical_only = true) ||
        throw(DimensionMismatch(monthly_temperature, sst))

    # create interpolator from grid in file to grid used in model
    interp = RingGrids.interpolator(monthly_temperature, sst, NF = Float32)
    interpolate!(monthly_temperature, sst, interp)
    close(ncfile)
    return nothing
end

function initialize!(
        ocean::PrognosticVariablesOcean,
        progn::PrognosticVariables,
        diagn::DiagnosticVariables,
        ocean_model::SeasonalOceanClimatology,
        model::PrimitiveEquation,
    ) where {PrognosticVariablesOcean}
    return timestep!(progn, diagn, ocean_model, model)
end

function timestep!(
        progn::PrognosticVariables,
        diagn::DiagnosticVariables,
        ocean::SeasonalOceanClimatology,
        model::PrimitiveEquation,
    )
    (; time) = progn.clock

    this_month = Dates.month(time)
    next_month = (this_month % 12) + 1      # mod for dec 12 -> jan 1

    # linear interpolation weight between the two months
    # TODO check whether this shifts the climatology by 1/2 a month
    (; monthly_temperature) = ocean
    (; sea_surface_temperature) = progn.ocean
    NF = eltype(sea_surface_temperature)
    weight = month_interpolation_weight(NF, time)

    return launch!(
        architecture(sea_surface_temperature), LinearWorkOrder, size(sea_surface_temperature),
        seasonal_ocean_kernel!,
        sea_surface_temperature, monthly_temperature, weight, this_month, next_month
    )
end

@kernel inbounds = true function seasonal_ocean_kernel!(
        sst, monthly_temp, weight, this_month, next_month
    )

    ij = @index(Global, Linear)

    sst[ij] = (1 - weight) * monthly_temp[ij, this_month] + weight * monthly_temp[ij, next_month]
end

## SEASONAL OCEAN CLIMATOLOGY + ANOMALY
export SeasonalOceanClimatologyAnomaly

"""
Seasonal ocean climatology plus externally prescribed monthly anomalies.
Climatology and anomaly data are both interpolated in time and then added,
and the resulting SST is used as lower boundary forcing.
Fields and options are
$(TYPEDFIELDS)"""
@kwdef mutable struct SeasonalOceanClimatologyAnomaly{NF, Grid, GridVariable3D} <: AbstractOcean

    "Grid used for the model"
    grid::Grid

    "[OPTION] Path to the folder containing sea surface temperature climatology, pkg path default"
    path::String = "SpeedyWeather.jl/input_data"

    "[OPTION] Filename of sea surface temperature climatology"
    file::String = "sea_surface_temperature.nc"

    "[OPTION] Variable name of sea surface temperature climatology in netcdf file"
    varname::String = "sst"

    "[OPTION] Grid the climatology file comes on"
    file_Grid::Type{<:AbstractGrid} = FullGaussianGrid

    "[OPTION] Missing value in climatology data representing land"
    missing_value::NF = NaN

    "[OPTION] Path to the folder containing SST anomalies, pkg path default"
    anomaly_path::String = "SpeedyWeather.jl/input_data"

    "[OPTION] Filename of SST anomalies"
    anomaly_file::String = "sst_anomaly.nc"

    "[OPTION] Variable name of SST anomalies in netcdf file"
    anomaly_varname::String = "ssta"

    "[OPTION] Time variable name for SST anomalies in netcdf file"
    anomaly_time_varname::String = "time"

    "[OPTION] Grid the anomaly file comes on"
    anomaly_file_Grid::Type{<:AbstractGrid} = FullGaussianGrid

    "[OPTION] Missing value in anomaly data representing land"
    anomaly_missing_value::NF = NaN

    "[OPTION] Clamp anomalies to first/last available month when simulation time is out of range"
    clamp_outside_range::Bool = true

    # to be filled from file
    "Monthly climatological SST [K], interpolated onto Grid"
    monthly_temperature::GridVariable3D = zeros(GridVariable3D, grid, 12)

    "Monthly SST anomalies [K], interpolated onto Grid"
    monthly_anomaly::GridVariable3D = zeros(GridVariable3D, grid, 0)

    "DateTime axis of monthly SST anomalies"
    anomaly_time::Vector{DateTime} = DateTime[]

    "Store whether the out-of-range anomaly warning has already been shown"
    clamp_warning_issued::Bool = false
end

# generator function
function SeasonalOceanClimatologyAnomaly(SG::SpectralGrid; kwargs...)
    (; NF, GridVariable3D, grid) = SG
    return SeasonalOceanClimatologyAnomaly{NF, typeof(grid), GridVariable3D}(; grid, kwargs...)
end

function initialize!(ocean::SeasonalOceanClimatologyAnomaly, model::PrimitiveEquation)
    (; monthly_temperature) = ocean

    # LOAD CLIMATOLOGY
    path = ocean_data_path(ocean.path, ocean.file)
    ncfile = NCDataset(path)

    clim_fill_value = ncfile[ocean.varname].attrib["_FillValue"]
    sst = ocean.file_Grid(load_ocean_variable_3d(ncfile, ocean.varname), input_as = Matrix)
    sst[sst .=== clim_fill_value] .= ocean.missing_value      # === to include NaN
    close(ncfile)

    # transfer to architecture of model if needed
    sst = on_architecture(model.architecture, sst)

    @boundscheck fields_match(monthly_temperature, sst, vertical_only = true) ||
        throw(DimensionMismatch(monthly_temperature, sst))

    interp = RingGrids.interpolator(monthly_temperature, sst, NF = Float32)
    interpolate!(monthly_temperature, sst, interp)

    # LOAD ANOMALIES
    anomaly_path = ocean_data_path(ocean.anomaly_path, ocean.anomaly_file)
    ncfile = NCDataset(anomaly_path)

    anomaly_fill_value = ncfile[ocean.anomaly_varname].attrib["_FillValue"]
    ssta = ocean.anomaly_file_Grid(load_ocean_variable_3d(ncfile, ocean.anomaly_varname), input_as = Matrix)
    ssta[ssta .=== anomaly_fill_value] .= ocean.anomaly_missing_value
    anomaly_time = load_time_axis(ncfile, ocean.anomaly_time_varname)
    close(ncfile)

    ssta = on_architecture(model.architecture, ssta)
    monthly_anomaly = zeros(typeof(monthly_temperature), ocean.grid, length(anomaly_time))

    @boundscheck fields_match(monthly_anomaly, ssta, vertical_only = true) ||
        throw(DimensionMismatch(monthly_anomaly, ssta))

    interp = RingGrids.interpolator(monthly_anomaly, ssta, NF = Float32)
    interpolate!(monthly_anomaly, ssta, interp)

    ocean.monthly_anomaly = monthly_anomaly
    ocean.anomaly_time = anomaly_time
    ocean.clamp_warning_issued = false

    return nothing
end

function initialize!(
        ocean::PrognosticVariablesOcean,
        progn::PrognosticVariables,
        diagn::DiagnosticVariables,
        ocean_model::SeasonalOceanClimatologyAnomaly,
        model::PrimitiveEquation,
    ) where {PrognosticVariablesOcean}
    return timestep!(progn, diagn, ocean_model, model)
end

function timestep!(
        progn::PrognosticVariables,
        diagn::DiagnosticVariables,
        ocean::SeasonalOceanClimatologyAnomaly,
        model::PrimitiveEquation,
    )
    (; time) = progn.clock

    this_month = Dates.month(time)
    next_month = (this_month % 12) + 1      # mod for dec 12 -> jan 1

    (; sea_surface_temperature) = progn.ocean
    NF = eltype(sea_surface_temperature)
    weight_clim = month_interpolation_weight(NF, time)

    (; anomaly_time, clamp_outside_range) = ocean
    midnight = DateTime(Dates.Date(time))
    if clamp_outside_range && !ocean.clamp_warning_issued &&
        (midnight < anomaly_time[1] || midnight > anomaly_time[end])

        @warn "Time $time is outside anomaly range $(anomaly_time[1]) to $(anomaly_time[end]). " *
            "Clamping SST anomalies to nearest available month."
        ocean.clamp_warning_issued = true
    end

    this_anomaly, next_anomaly, weight_anomaly = anomaly_interpolation_indices(
        NF, time, anomaly_time; clamp_outside_range
    )

    return launch!(
        architecture(sea_surface_temperature), LinearWorkOrder, size(sea_surface_temperature),
        seasonal_ocean_anomaly_kernel!,
        sea_surface_temperature, ocean.monthly_temperature, ocean.monthly_anomaly,
        weight_clim, this_month, next_month, weight_anomaly, this_anomaly, next_anomaly
    )
end

@kernel inbounds = true function seasonal_ocean_anomaly_kernel!(
        sst, monthly_temp, monthly_anomaly, weight_clim, this_month, next_month,
        weight_anomaly, this_anomaly, next_anomaly
    )

    ij = @index(Global, Linear)

    sst_clim = (1 - weight_clim) * monthly_temp[ij, this_month] + weight_clim * monthly_temp[ij, next_month]
    sst_anomaly = (1 - weight_anomaly) * monthly_anomaly[ij, this_anomaly] + weight_anomaly * monthly_anomaly[ij, next_anomaly]
    sst[ij] = sst_clim + sst_anomaly
end

## CONSTANT OCEAN CLIMATOLOGY
export ConstantOceanClimatology

"""
Constant ocean climatology that reads monthly sea surface temperature
fields from file, and interpolates them only for the initial conditions
in time to be stored in the prognostic variables. It is therefore an
ocean from climatology but without a seasonal cycle that is constant in time.
To be created like

    ocean = SeasonalOceanClimatology(spectral_grid)

and the ocean time is set with `initialize!(model, time=time)`.
Fields and options are
$(TYPEDFIELDS)"""
@kwdef struct ConstantOceanClimatology <: AbstractOcean
    "[OPTION] path to the folder containing the land-sea mask file, pkg path default"
    path::String = "SpeedyWeather.jl/input_data"

    "[OPTION] filename of sea surface temperatures"
    file::String = "sea_surface_temperature.nc"

    "[OPTION] Variable name in netcdf file"
    varname::String = "sst"

    "[OPTION] Grid the sea surface temperature file comes on"
    file_Grid::Type{<:AbstractGrid} = FullGaussianGrid

    "[OPTION] The missing value in the data respresenting land"
    missing_value::Float64 = NaN
end

# generator
ConstantOceanClimatology(SG::SpectralGrid; kwargs...) = ConstantOceanClimatology(; kwargs...)

# nothing to initialize for model.ocean
initialize!(::ConstantOceanClimatology, ::PrimitiveEquation) = nothing

# initialize
function initialize!(
        ocean::PrognosticVariablesOcean,
        progn::PrognosticVariables,
        diagn::DiagnosticVariables,
        ocean_model::ConstantOceanClimatology,
        model::PrimitiveEquation,
    ) where {PrognosticVariablesOcean}
    # create a seasonal model, initialize it and the variables
    (; path, file, varname, file_Grid, missing_value) = ocean_model
    (; NF, GridVariable3D, grid) = model.spectral_grid
    seasonal_model = SeasonalOceanClimatology{NF, typeof(grid), GridVariable3D}(;
        grid, path, file, varname, file_Grid, missing_value
    )
    initialize!(seasonal_model, model)
    return initialize!(ocean, progn, diagn, seasonal_model, model)
    # (seasonal model will be garbage collected hereafter)
end

function timestep!(
        progn::PrognosticVariables,
        diagn::DiagnosticVariables,
        ocean_model::ConstantOceanClimatology,
        model::PrimitiveEquation,
    )
    return nothing
end

## CONSTANT OCEAN CLIMATOLOGY
export AquaPlanet

"""
AquaPlanet sea surface temperatures that are constant in time and longitude,
but vary in latitude following a coslat². To be created like

    ocean = AquaPlanet(spectral_grid, temp_equator=302, temp_poles=273)

Fields and options are
$(TYPEDFIELDS)"""
@parameterized @kwdef struct AquaPlanet{NF} <: AbstractOcean
    "[OPTION] Temperature on the Equator [K]"
    @param temp_equator::NF = 302 (bounds = Positive,)

    "[OPTION] Temperature at the poles [K]"
    @param temp_poles::NF = 273 (bounds = Positive,)

    "[OPTION] Mask the sea surface temperature according to model.land_sea_mask?"
    mask::Bool = true
end

# generator function
AquaPlanet(SG::SpectralGrid; kwargs...) = AquaPlanet{SG.NF}(; kwargs...)

# nothing to initialize for AquaPlanet
initialize!(::AquaPlanet, ::PrimitiveEquation) = nothing

# initialize
function initialize!(
        ocean::PrognosticVariablesOcean,
        progn::PrognosticVariables,
        diagn::DiagnosticVariables,
        ocean_model::AquaPlanet,
        model::PrimitiveEquation,
    ) where {PrognosticVariablesOcean}
    (; sea_surface_temperature) = ocean
    Te, Tp = ocean_model.temp_equator, ocean_model.temp_poles
    sst(λ, φ) = (Te - Tp) * cosd(φ)^2 + Tp
    set!(sea_surface_temperature, sst, model.geometry)
    return ocean_model.mask && mask!(sea_surface_temperature, model.land_sea_mask, :land)
end

function timestep!(
        progn::PrognosticVariables,
        diagn::DiagnosticVariables,
        ocean_model::AquaPlanet,
        model::PrimitiveEquation,
    )
    return nothing
end


export SlabOcean

@parameterized @kwdef mutable struct SlabOcean{NF} <: AbstractOcean
    "[OPTION] Specific heat capacity of water [J/kg/K]"
    specific_heat_capacity::NF = 4184

    "[OPTION] Average mixed-layer depth [m]"
    @param mixed_layer_depth::NF = 50 (bounds = Positive,)

    "[OPTION] Density of water [kg/m³]"
    density::NF = 1000

    "[OPTION] Mask initial sea surface temperature with land-sea mask?"
    mask::Bool = true

    "[OPTION] SST over land [K]"
    land_temperature::NF = 283

    "[DERIVED] Effective mixed-layer heat capacity [J/K/m²]"
    heat_capacity_mixed_layer::NF = specific_heat_capacity * mixed_layer_depth * density
end

# generator function
SlabOcean(SG::SpectralGrid; kwargs...) = SlabOcean{SG.NF}(; kwargs...)

function variables(::SlabOcean)
    return (
        PrognosticVariable(name = :sea_surface_temperature, dims = Grid2D(), namespace = :ocean, desc = "Sea surface temperature", units = "K"),

        DiagnosticVariable(name = :surface_shortwave_down, dims = Grid2D(), desc = "Surface shortwave radiation down", units = "W/m^2"),
        DiagnosticVariable(name = :surface_shortwave_up, dims = Grid2D(), desc = "Surface shortwave radiation up over ocean", units = "W/m^2", namespace = :ocean),
        DiagnosticVariable(name = :surface_longwave_down, dims = Grid2D(), desc = "Surface longwave radiation down", units = "W/m^2"),
        DiagnosticVariable(name = :surface_longwave_up, dims = Grid2D(), desc = "Surface longwave radiation up over ocean", units = "W/m^2", namespace = :ocean),

        DiagnosticVariable(name = :surface_humidity_flux, dims = Grid2D(), desc = "Surface humidity flux", units = "kg/s/m^2", namespace = :ocean),
        DiagnosticVariable(name = :surface_sensible_heat_flux, dims = Grid2D(), desc = "Surface sensible heat flux", units = "kg/s/m^2", namespace = :ocean),
    )
end

# nothing to initialize for SlabOcean
initialize!(ocean_model::SlabOcean, model::PrimitiveEquation) = nothing

# initialize
function initialize!(
        ocean::PrognosticVariablesOcean,
        progn::PrognosticVariables,
        diagn::DiagnosticVariables,
        ocean_model::SlabOcean,
        model::PrimitiveEquation,
    ) where {PrognosticVariablesOcean}
    # create a seasonal model, initialize it and the variables
    seasonal_model = SeasonalOceanClimatology(model.spectral_grid)
    initialize!(seasonal_model, model)
    initialize!(ocean, progn, diagn, seasonal_model, model)
    # (seasonal model will be garbage collected hereafter)

    # set land "sst" points (100% land only)
    return if ocean_model.mask
        masked_value = ocean_model.land_temperature
        sst = progn.ocean.sea_surface_temperature.data
        # TODO: broadcasting over views of Fields of GPUArrays doesn't work
        sst[isnan.(sst)] .= masked_value
        mask!(progn.ocean.sea_surface_temperature, model.land_sea_mask, :land; masked_value)
    end
end

function timestep!(
        progn::PrognosticVariables,
        diagn::DiagnosticVariables,
        ocean_model::SlabOcean,
        model::PrimitiveEquation,
    )
    sst = progn.ocean.sea_surface_temperature

    Lᵥ = latent_heat_condensation(model.atmosphere)
    C₀ = ocean_model.heat_capacity_mixed_layer
    Δt = model.time_stepping.Δt_sec
    Δt_C₀ = Δt / C₀

    (; mask) = model.land_sea_mask

    # Frierson et al. 2006, eq (1), all W/m² except humidity flux in kg/m²/s
    Rsd = diagn.physics.surface_shortwave_down          # before albedo
    Rsu = diagn.physics.ocean.surface_shortwave_up      # reflected from albedo
    Rld = diagn.physics.surface_longwave_down
    Rlu = diagn.physics.ocean.surface_longwave_up
    S = diagn.physics.ocean.sensible_heat_flux
    H = diagn.physics.ocean.surface_humidity_flux       # [kg/m²/s]

    params = (; Δt_C₀, Lᵥ)                              # pack into NamedTuple for kernel

    launch!(
        architecture(sst), LinearWorkOrder, size(sst), slab_ocean_kernel!,
        sst, mask, Rsd, Rsu, Rld, Rlu, H, S, params
    )
    return nothing
end

@kernel inbounds = true function slab_ocean_kernel!(sst, mask, Rsd, Rsu, Rld, Rlu, H, S, params)
    ij = @index(Global, Linear)         # every grid point ij
    if mask[ij] < 1                     # at least partially ocean
        (; Δt_C₀, Lᵥ) = params
        sst[ij] += Δt_C₀ * (Rsd[ij] - Rsu[ij] - Rlu[ij] + Rld[ij] - Lᵥ * H[ij] - S[ij])
    end
end
