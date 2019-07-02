using Cairo, Gtk, Colors

struct AtariCairoCtx <: AbstractCtx end
struct AtariRGBCtx <: AbstractCtx end
struct NoCtx <: AbstractCtx end
struct AtariGtkCtx <: AbstractCtx
    canvas::GtkCanvas
    win::GtkWindowLeaf
end

function Ctx(env::AtariEnv, mode::Symbol=:no_render)
    if mode == :human_pane
        AtariCairoCtx()

    elseif mode == :human_window
        canvas = @GtkCanvas()
        win = GtkWindow(canvas, "AtariEnv")
        show(canvas)
        visible(win, false)
        signal_connect(win, "delete-event") do widget, event
            ccall((:gtk_widget_hide_on_delete, Gtk.libgtk), Bool, (Ptr{GObject},), win)
        end

        AtariGtkCtx(canvas, win)
    elseif mode == :rgb

        AtariRGBCtx()
    elseif mode == :no_render
        NoCtx()
    else
        error("Unrecognized mode in Ctx(): $(mode)")
    end
end

function drawcanvas(env::AtariEnv)
    screen_grab = getScreenRGB(env.ale)
    w, h = getScreenWidth(env.ale), getScreenHeight(env.ale)

    # Converting RGB values to 32 bit color...
    r_screen_grab = screen_grab[1:3:end] * 0x00010000
    g_screen_grab = screen_grab[2:3:end] * 0x00000100
    b_screen_grab = screen_grab[3:3:end]

    rgb_array = reshape(r_screen_grab .+ b_screen_grab .+ g_screen_grab, w, h) |> transpose |> Array
end

render!(env::AbstractEnv, ctx::NoCtx) = nothing
render!(env::AtariEnv, ctx::AtariCairoCtx) = env |> drawcanvas |> CairoRGBSurface |> display

function render!(env::AtariEnv, ctx::AtariGtkCtx)
    !visible(ctx.win) && visible(ctx.win, true)
    @guarded draw(ctx.canvas) do widget
        context = getgc(ctx.canvas)
        image(context, CairoRGBSurface(drawcanvas(env)),
            0, 0, getScreenWidth(env.ale), getScreenHeight(env.ale))
    end
end

render!(env::AtariEnv, ctx::AtariRGBCtx) = get_preprocessed_RGB(env)
