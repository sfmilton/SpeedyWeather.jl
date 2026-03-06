"""$(TYPEDSIGNATURES)
Compute tendencies for u, v, temp, humid from physical parametrizations."""
function parameterization_tendencies!(
        diagn::DiagnosticVariables,
        progn::PrognosticVariables,
        model::PrimitiveEquation,
    )
    # parameterizations with their own kernel
    (; time) = progn.clock
    cos_zenith!(diagn, time, model)
    reset_variables!(diagn)

    (; architecture, npoints) = model.spectral_grid
    if architecture isa Architectures.AbstractCPU
        # bypass kernel launch on CPU
        parameterization_tendencies_cpu!(diagn, progn, model)
    else
        # GPU: all other parameterizations are fused into a single kernel over horizontal grid point index ij
        launch!(
            architecture, LinearWorkOrder, (npoints,), parameterization_tendencies_kernel!,
            diagn, progn, get_parameterizations(model), model
        )
    end
    return nothing
end

# GPU kernel, unrolling NamedTuple iteration at compile time, fuses all parameterizations
@kernel inbounds = true function parameterization_tendencies_kernel!(diagn, progn, parameterizations, model)

    ij = @index(Global, Linear)     # every horizontal grid point ij

    # manually unroll loop over all parameterizations (NamedTuple iteration not GPU-compatible)
    _call_parameterizations!(ij, diagn, progn, parameterizations, model)

    # tendencies have to be scaled by the radius for the dynamical core
    scale!(ij, diagn.tendencies, model.planet.radius)
    scale_individual_physics_tendency_terms!(ij, diagn.tendencies, model.planet.radius)
    compute_boundary_layer_tendency_terms!(ij, diagn.tendencies)
end

# CPU without kernel, just a loop, change loop order compared to GPU though:
# outer loop over parameterizations, inner loop over horizontal grid points
# this yields a more contiguous memory access pattern on CPU
function parameterization_tendencies_cpu!(diagn, progn, model)
    @inbounds _call_parameterizations_cpu!(diagn, progn, get_parameterizations(model), model)

    radius = model.planet.radius
    @inbounds for ij in 1:model.geometry.npoints
        # tendencies have to be scaled by the radius for the dynamical core
        scale!(ij, diagn.tendencies, radius)
        scale_individual_physics_tendency_terms!(ij, diagn.tendencies, radius)
        compute_boundary_layer_tendency_terms!(ij, diagn.tendencies)
    end
    return nothing
end

# Use @generated to unroll NamedTuple iteration at compile time for GPU compatibility
@generated function _call_parameterizations!(ij, diagn, progn, parameterizations::NamedTuple{names}, model) where {names}
    calls = [
        :(
                parameterization_with_tracking!(ij, diagn, progn, parameterizations.$name, Val{$(QuoteNode(name))}(), model)
            ) for name in names
    ]
    return quote
        Base.@_propagate_inbounds_meta
        $(Expr(:block, calls...))
    end
end

# Use @generated to unroll NamedTuple iteration at compile time also on CPU for performance
@generated function _call_parameterizations_cpu!(diagn, progn, parameterizations::NamedTuple{names}, model) where {names}
    calls = [
        quote
                for ij in 1:model.geometry.npoints      # horizontal grid points inner loop
                    parameterization_with_tracking!(ij, diagn, progn, parameterizations.$name, Val{$(QuoteNode(name))}(), model)
            end
            end for name in names
    ]                    # parameterizations outer loop
    return quote
        Base.@_propagate_inbounds_meta
        $(Expr(:block, calls...))
    end
end

@generated function tendency_term_fields(tendencies::Tendencies, ::Val{name}) where {name}
    return if name === :convection
        :((
            tendencies.u_tend_conv_grid, tendencies.v_tend_conv_grid,
            tendencies.temp_tend_conv_grid, tendencies.humid_tend_conv_grid
        ))
    elseif name === :vertical_diffusion
        :((
            tendencies.u_tend_vdiff_grid, tendencies.v_tend_vdiff_grid,
            tendencies.temp_tend_vdiff_grid, tendencies.humid_tend_vdiff_grid
        ))
    elseif name === :large_scale_condensation
        :((
            tendencies.u_tend_lsc_grid, tendencies.v_tend_lsc_grid,
            tendencies.temp_tend_lsc_grid, tendencies.humid_tend_lsc_grid
        ))
    elseif name === :shortwave_radiation
        :((
            tendencies.u_tend_sw_grid, tendencies.v_tend_sw_grid,
            tendencies.temp_tend_sw_grid, tendencies.humid_tend_sw_grid
        ))
    elseif name === :longwave_radiation
        :((
            tendencies.u_tend_lw_grid, tendencies.v_tend_lw_grid,
            tendencies.temp_tend_lw_grid, tendencies.humid_tend_lw_grid
        ))
    elseif name === :surface_momentum_flux
        :((
            tendencies.u_tend_smf_grid, tendencies.v_tend_smf_grid,
            tendencies.temp_tend_smf_grid, tendencies.humid_tend_smf_grid
        ))
    elseif name === :surface_heat_flux
        :((
            tendencies.u_tend_shf_grid, tendencies.v_tend_shf_grid,
            tendencies.temp_tend_shf_grid, tendencies.humid_tend_shf_grid
        ))
    elseif name === :surface_humidity_flux
        :((
            tendencies.u_tend_shuf_grid, tendencies.v_tend_shuf_grid,
            tendencies.temp_tend_shuf_grid, tendencies.humid_tend_shuf_grid
        ))
    elseif name === :stochastic_physics
        :((
            tendencies.u_tend_stoch_grid, tendencies.v_tend_stoch_grid,
            tendencies.temp_tend_stoch_grid, tendencies.humid_tend_stoch_grid
        ))
    else
        :(nothing)
    end
end

@inline function parameterization_with_tracking!(
        ij::Integer,
        diagn::DiagnosticVariables,
        progn::PrognosticVariables,
        parameterization,
        ::Val{name},
        model::PrimitiveEquation,
    ) where {name}
    tendencies = diagn.tendencies
    nlayers = tendencies.nlayers

    # Use dynamics tendency buffers as temporary snapshots during physics accumulation.
    u_prev = tendencies.u_tend_dynamics_grid
    v_prev = tendencies.v_tend_dynamics_grid
    temp_prev = tendencies.temp_tend_dynamics_grid
    humid_prev = tendencies.humid_tend_dynamics_grid

    @inbounds for k in 1:nlayers
        u_prev[ij, k] = tendencies.u_tend_grid[ij, k]
        v_prev[ij, k] = tendencies.v_tend_grid[ij, k]
        temp_prev[ij, k] = tendencies.temp_tend_grid[ij, k]
        humid_prev[ij, k] = tendencies.humid_tend_grid[ij, k]
    end

    parameterization!(ij, diagn, progn, parameterization, model)

    term_fields = tendency_term_fields(tendencies, Val(name))
    isnothing(term_fields) && return nothing

    u_term, v_term, temp_term, humid_term = term_fields
    @inbounds for k in 1:nlayers
        u_term[ij, k] = tendencies.u_tend_grid[ij, k] - u_prev[ij, k]
        v_term[ij, k] = tendencies.v_tend_grid[ij, k] - v_prev[ij, k]
        temp_term[ij, k] = tendencies.temp_tend_grid[ij, k] - temp_prev[ij, k]
        humid_term[ij, k] = tendencies.humid_tend_grid[ij, k] - humid_prev[ij, k]
    end
    return nothing
end

function scale_individual_physics_tendency_terms!(ij::Integer, tendencies::Tendencies, scale::Real)
    return @inbounds for k in eachlayer(tendencies.u_tend_grid)
        tendencies.u_tend_conv_grid[ij, k] *= scale
        tendencies.v_tend_conv_grid[ij, k] *= scale
        tendencies.temp_tend_conv_grid[ij, k] *= scale
        tendencies.humid_tend_conv_grid[ij, k] *= scale

        tendencies.u_tend_vdiff_grid[ij, k] *= scale
        tendencies.v_tend_vdiff_grid[ij, k] *= scale
        tendencies.temp_tend_vdiff_grid[ij, k] *= scale
        tendencies.humid_tend_vdiff_grid[ij, k] *= scale

        tendencies.u_tend_lsc_grid[ij, k] *= scale
        tendencies.v_tend_lsc_grid[ij, k] *= scale
        tendencies.temp_tend_lsc_grid[ij, k] *= scale
        tendencies.humid_tend_lsc_grid[ij, k] *= scale

        tendencies.u_tend_sw_grid[ij, k] *= scale
        tendencies.v_tend_sw_grid[ij, k] *= scale
        tendencies.temp_tend_sw_grid[ij, k] *= scale
        tendencies.humid_tend_sw_grid[ij, k] *= scale

        tendencies.u_tend_lw_grid[ij, k] *= scale
        tendencies.v_tend_lw_grid[ij, k] *= scale
        tendencies.temp_tend_lw_grid[ij, k] *= scale
        tendencies.humid_tend_lw_grid[ij, k] *= scale

        tendencies.u_tend_smf_grid[ij, k] *= scale
        tendencies.v_tend_smf_grid[ij, k] *= scale
        tendencies.temp_tend_smf_grid[ij, k] *= scale
        tendencies.humid_tend_smf_grid[ij, k] *= scale

        tendencies.u_tend_shf_grid[ij, k] *= scale
        tendencies.v_tend_shf_grid[ij, k] *= scale
        tendencies.temp_tend_shf_grid[ij, k] *= scale
        tendencies.humid_tend_shf_grid[ij, k] *= scale

        tendencies.u_tend_shuf_grid[ij, k] *= scale
        tendencies.v_tend_shuf_grid[ij, k] *= scale
        tendencies.temp_tend_shuf_grid[ij, k] *= scale
        tendencies.humid_tend_shuf_grid[ij, k] *= scale

        tendencies.u_tend_stoch_grid[ij, k] *= scale
        tendencies.v_tend_stoch_grid[ij, k] *= scale
        tendencies.temp_tend_stoch_grid[ij, k] *= scale
        tendencies.humid_tend_stoch_grid[ij, k] *= scale
    end
end

function compute_boundary_layer_tendency_terms!(ij::Integer, tendencies::Tendencies)
    return @inbounds for k in eachlayer(tendencies.u_tend_grid)
        tendencies.u_tend_bl_grid[ij, k] = tendencies.u_tend_vdiff_grid[ij, k] +
            tendencies.u_tend_smf_grid[ij, k] + tendencies.u_tend_shf_grid[ij, k] +
            tendencies.u_tend_shuf_grid[ij, k]
        tendencies.v_tend_bl_grid[ij, k] = tendencies.v_tend_vdiff_grid[ij, k] +
            tendencies.v_tend_smf_grid[ij, k] + tendencies.v_tend_shf_grid[ij, k] +
            tendencies.v_tend_shuf_grid[ij, k]
        tendencies.temp_tend_bl_grid[ij, k] = tendencies.temp_tend_vdiff_grid[ij, k] +
            tendencies.temp_tend_smf_grid[ij, k] + tendencies.temp_tend_shf_grid[ij, k] +
            tendencies.temp_tend_shuf_grid[ij, k]
        tendencies.humid_tend_bl_grid[ij, k] = tendencies.humid_tend_vdiff_grid[ij, k] +
            tendencies.humid_tend_smf_grid[ij, k] + tendencies.humid_tend_shf_grid[ij, k] +
            tendencies.humid_tend_shuf_grid[ij, k]
    end
end

"""$(TYPEDSIGNATURES)
Flux `flux` into surface layer with surface pressure `pₛ` [Pa] and gravity `g` [m/s^2]
converted to tendency [?/s]."""
@propagate_inbounds surface_flux_to_tendency(flux::Real, pₛ::Real, model) =
    flux_to_tendency(flux, pₛ, model.planet.gravity, model.geometry.σ_levels_thick[end])

"""$(TYPEDSIGNATURES)
Flux `flux` into layer `k` of thickness `Δσ`  converted to tendency [?/s].
Using surface pressure `pₛ` [Pa] and gravity `g` [m/s^2]."""
@propagate_inbounds flux_to_tendency(flux::Real, pₛ::Real, g::Real, Δσ_k::Real) = g / (pₛ * Δσ_k) * flux
@propagate_inbounds flux_to_tendency(flux::Real, pₛ::Real, k::Int, model) =
    flux_to_tendency(flux, pₛ, model.planet.gravity, model.geometry.σ_levels_thick[k])

# hacky, temporary placement, and also modularize this?
function reset_variables!(diagn::DiagnosticVariables)
    reset_variable!(diagn.physics, :cloud_top, diagn.nlayers + 1)   # reset to below top layer
    reset_variable!(diagn.physics, :rain_rate, 0)
    reset_variable!(diagn.physics, :snow_rate, 0)
    reset_variable!(diagn.physics, :surface_humidity_flux, 0)
    reset_variable!(diagn.physics, :sensible_heat_flux, 0)
    return nothing
end

function reset_variable!(vars, var::Symbol, reset_value)
    return if haskey(vars, var)
        field = getfield(vars, var)
        field .= reset_value
    end
end
