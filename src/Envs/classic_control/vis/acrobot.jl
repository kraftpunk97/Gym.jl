include("utils.jl")

struct AcrobotDrawParams <: AbstractDrawParams end

function Ctx(env::AcrobotEnv, mode::Symbol=:no_render)
    if mode == :human_pane
        draw_parameters = AcrobotDrawParams()
        viewer = CairoRGBSurface(500, 500)  # screen_size = 500 x 500

        CairoCtx(draw_parameters, viewer)
    elseif mode == :human_window
        draw_parameters = AcrobotDrawParams()
        viewer = CairoRGBSurface(draw_parameters.screen_width, draw_parameters.screen_height)

        canvas = @GtkCanvas()
        canvas.backcc = CairoContext(viewer)
        win = GtkWindow(canvas, "Acrobot",
                draw_parameters.screen_width, draw_parameters.screen_height; resizable=false)
        show(canvas)
        visible(win, false)
        signal_connect(win, "delete-event") do widget, event
            ccall((:gtk_widget_hide_on_delete, Gtk.libgtk), Bool, (Ptr{GObject},), win)
        end

        GtkCtx(draw_parameters, canvas, win)
    elseif mode == :rgb
        draw_parameters = AcrobotDrawParams()
        viewer = CairoRGBSurface(draw_parameters.screen_width, draw_parameters.screen_height)

        RGBCtx(draw_parameters, viewer)
    elseif mode == :no_render
        NoCtx()
    else
        error("Unrecognized mode in Ctx(): $(mode)")
    end
end

function drawcanvas!(env::AcrobotEnv, viewer::CairoContext, parameters::AcrobotDrawParams)
    # Set background
    set_source_rgb(viewer, 1, 1, 1)
    rectangle(viewer, 0, 0, 500, 500) # screen_size = 500 x 500
    fill(viewer)

    # Draw goal
    set_source_rgb(viewer, 0, 0, 0)
    move_to(viewer, 0, 136)
    line_to(viewer, 500, 136)
    stroke(viewer)
    close_path(viewer)

    # First translation and rotation
    translate(viewer, 250, 250)  # Bring the origin to the center of the screen
    theta1 = -env.state[1].data
    rotate(viewer, theta1)

    # Draw the first link
    set_source_rgb(viewer, 0, 0.8, 0.8)
    set_line_width(viewer, 19)
    move_to(viewer, 0, 0)
    line_to(viewer, 0, 113)  # Link length translated into the view coordinates
    stroke(viewer)
    set_line_width(viewer, 1)  # Reset line width to avoid unexpected surprises

    # Draw the first joint
    set_source_rgb(viewer, 0.8, 0.8, 0)
    circle(viewer, 0, 0, 10)
    fill(viewer)

    # Second translation and rotation
    translate(viewer, 0, 113)
    theta2 = -env.state[2].data
    rotate(viewer, theta2)

    # Draw the second link
    set_source_rgb(viewer, 0, 0.8, 0.8)
    set_line_width(viewer, 19)
    move_to(viewer, 0, 0)
    line_to(viewer, 0, 113)
    stroke(viewer)
    close_path(viewer)
    set_line_width(viewer, 1)

    # Draw the second joint
    set_source_rgb(viewer, 0.8, 0.8, 0)
    circle(viewer, 0, 0, 10)
    fill(viewer)

    # Undo all translations and rotations
    rotate(viewer, -theta2)
    translate(viewer, 0, -113)
    rotate(viewer, -theta1)
    translate(viewer, -250, -250)

    #=
    # Calibration circle
    set_source_rgb(viewer, 0, 0, 0)
    calibration_radius = 10
    circle(viewer, calibration_radius, calibration_radius, calibration_radius)
    set_line_width(viewer, 1)
    stroke(viewer)
    move_to(viewer, calibration_radius, calibration_radius)
    line_to(viewer, 2calibration_radius, calibration_radius)
    close_path(viewer)
    set_line_width(viewer, 1)
    stroke(viewer)
    set_line_width(viewer, 1)
    =#
end
