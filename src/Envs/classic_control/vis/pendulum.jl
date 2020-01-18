include("utils.jl")

obs(env::PendulumEnv, ::Nothing) = obs(env, [0.0, 0.0])
obs(env::PendulumEnv, (θ, θ̄)) = Flux.data(θ)

struct PendulumDrawParams <: AbstractDrawParams
    screen_height::UInt32
    screen_width::UInt32
    world_width::Float32
    scale::Float32
    arm_length::Float32
    arm_width::Float32
    axle_radius::Float32
end

PendulumDrawParams() =
    PendulumDrawParams(
        500,       # screen_height
        500,       # screen_width
        44f-1,     # world_width
        500/44f-1, # scale (screen_width / world_width)
        1f0,       # arm_length
        2f-1,      # arm_width
        5f-2       # axle_radius
    )

function Ctx(env::PendulumEnv, mode::Symbol=:no_render)
    if mode == :human_pane
        draw_params = PendulumDrawParams()
        viewer = CairoRGBSurface(draw_params.screen_width, draw_params.screen_height)

function Ctx(::PendulumEnv, ::Val{:human_pane})
    draw_params = PendulumDrawParams()
    viewer = CairoRGBSurface(draw_params.screen_width, draw_params.screen_height)

    CairoCtx(draw_params, viewer)
end

@init @require Gtk="4c0ca9eb-093a-5379-98c5-f87ac0bbbf44" function Ctx(::PendulumEnv, ::Val{human_window})
    draw_params = PendulumDrawParams()
    viewer = CairoRGBSurface(draw_params.screen_width, draw_params.screen_height)

        RGBCtx(draw_params, viewer)
    elseif mode == :no_render
        return
    else
        error("Unrecognized mode in Ctx(): $(mode)")
    end

    GtkCtx(draw_params, canvas, win)
end

function Ctx(::PendulumEnv, ::Val{:rgb})
    draw_params = PendulumDrawParams()
    viewer = CairoRGBSurface(draw_params.screen_width, draw_params.screen_height)

    RGBCtx(draw_params, viewer)
end

function drawcanvas!(env::PendulumEnv, viewer::CairoContext, params::PendulumDrawParams)
    # Background
    set_source_rgb(viewer, 1, 1, 1)
    rectangle(viewer, 0, 0, params.screen_width, params.screen_height)
    fill(viewer)

    # Move to center of screen
    translate_dist = Pair(params.screen_width/2, params.screen_height/2)
    translate(viewer, translate_dist.first, translate_dist.second)

    # Rotate
    rotate(viewer, env.state[2] * env.dt)

    # Arm Start Circle
    set_source_rgb(viewer, 8f-1, 3f-1, 3f-1)
    move_to(viewer, 0, 0)
    arc(viewer, 0, 0, params.scale * params.arm_width/2, π, 2*π)

    # Arm Side 1
    rel_line_to(viewer, 0, params.scale * params.arm_length)

    # Arm End Circle
    arc(viewer, 0, params.scale * params.arm_length, params.scale * params.arm_width/2, 0, π)

    # Arm Side 2
    rel_line_to(viewer, 0, -params.scale * params.arm_length)

    # Fill arm
    fill(viewer)

    # Axle
    set_source_rgb(viewer, 0, 0, 0)
    circle(viewer, 0, 0, params.scale * params.axle_radius)
    fill(viewer)

    # Undo translation and rotation
    rotate(viewer, -env.state[2] * env.dt)
    translate(viewer, -translate_dist.first, -translate_dist.second)
end
