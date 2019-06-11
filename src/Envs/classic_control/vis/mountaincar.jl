include("utils.jl")

height_ = (xs) -> 45f-2Float32(sin(3*xs)) + 55f-2
convert_x = (xs, parameters) -> (xs - -12f-1)*parameters.scale  # min_position = -12f-1
convert_y = (ys, parameters) -> (parameters.screen_height - ys*parameters.scale)

struct MountainCarDrawParams <: AbstractDrawParams
    screen_height::UInt32
    screen_width::UInt32
    scale::Float32
    world_width::Float32
    car_width::Float32
    car_height::Float32
end

function MountainCarDrawParams()

    MountainCarDrawParams(
        400,        # screen_height
        600,        # screen_width
        600/18f-1,  # scale
        18f-1,      # world_width
        40f0,       # car_width
        20f0,       # car_height
    )
end

function Ctx(env::MountainCarEnv, mode::Symbol=:no_render)
    if mode == :human_pane
        draw_parameters = MountainCarDrawParams()
        viewer = CairoRGBSurface(draw_parameters.screen_width, draw_parameters.screen_height)

        CairoCtx(draw_parameters, viewer)
    elseif mode == :human_window
        draw_parameters = MountainCarDrawParams()
        viewer = CairoRGBSurface(draw_parameters.screen_width, draw_parameters.screen_height)

        canvas = @GtkCanvas()
        canvas.backcc = CairoContext(viewer)
        win = GtkWindow(canvas, "MountainCar",
                draw_parameters.screen_width, draw_parameters.screen_height; resizable=false)
        show(canvas)
        visible(win, false)
        signal_connect(win, "delete-event") do widget, event
            ccall((:gtk_widget_hide_on_delete, Gtk.libgtk), Bool, (Ptr{GObject},), win)
        end

        GtkCtx(draw_parameters, canvas, win)
    elseif mode == :rgb
        draw_parameters = MountainCarDrawParams()
        viewer = CairoRGBSurface(draw_parameters.screen_width, draw_parameters.screen_height)

        RGBCtx(draw_parameters, viewer)
    elseif mode == :no_render
        NoCtx()
    else
        error("Unrecognized mode in Ctx(): $(mode)")
    end
end

function drawcanvas!(env::MountainCarEnv, viewer::CairoContext, parameters::MountainCarDrawParams)
    xs = collect(Float32.(range(-12f-1, 6f-1, length=100)))
    ys = convert_y.(height_.(xs), [parameters])
    xs = convert_x.(xs, [parameters])
    set_source_rgb(viewer, 1, 1, 1)
    rectangle(viewer, 0, 0, parameters.screen_width, parameters.screen_height)
    fill(viewer)

    # Track
    set_source_rgb(viewer, 0, 0, 0)
    move_to(viewer, xs[1], ys[1])
    for i=2:length(xs)
        line_to(viewer, xs[i], ys[i])
    end
    stroke(viewer)
    close_path(viewer)

    #Goalpost
    goal_coordinates = Pair(convert_x(env.goal_position, parameters),           # x_cordinate
                            convert_y(height_(env.goal_position), parameters))  # y_coordinate
    set_source_rgb(viewer, 1, 0, 0)
    move_to(viewer, goal_coordinates.first, goal_coordinates.second)
    line_to(viewer, goal_coordinates.first, goal_coordinates.second-50)
    set_line_width(viewer, 8)
    stroke(viewer)
    set_line_width(viewer, 1)  # Resetting line width
    close_path(viewer)

    # Car
    cartx = convert_x(env.state[1].data, parameters)
    carty = convert_y(height_(env.state[1].data), parameters)
    set_source_rgb(viewer, 0, 0, 0)
    translate(viewer, cartx, carty)
    #rotate(viewer, cos(3state[1]))
    rectangle(viewer, parameters.car_width/2, 0, -parameters.car_width, -parameters.car_height)  # cartx-car_width/2
    fill(viewer)

    # Wheels
    set_source_rgb(viewer, 0.5, 0.5, 0.5)
    circle(viewer, parameters.car_width/4, -5, 5)
    circle(viewer, -parameters.car_width/4, -5, 5)
    fill(viewer)
    translate(viewer, -cartx, -carty)  # Resetting the cursor

    #=
    # Calibration circle
    set_source_rgb(viewer, 0, 0, 0)
    calibration_radius = 10
    circle(viewer, calibration_radius, calibration_radius, calibration_radius)
    set_line_width(viewer, 1)
    stroke(viewer)
    =#
end
