struct HumanCtx <: AbstractCtx end
struct ANSICtx <: AbstractCtx end
struct NoCtx <: AbstractCtx end

render!(env::AbstractEnv, ctx::ANSICtx) = drawcanvas!(env)
render!(env::AbstractEnv, ctx::NoCtx) = nothing
function render!(env::AbstractEnv, ctx::HumanCtx)
    print(drawcanvas!(env))
end

function colorize(string::Union{<:AbstractString, Char}, color::Symbol; bold::Bool=false,
    highlight::Bool=false)

    colormap = Dict(
    :gray => 30, # highlights to black
    :red => 31,
    :green => 32,
    :yellow => 33,
    :blue => 34,
    :magenta => 35,
    :cyan => 36,
    :white => 37,
    :crimson => 38)

    num = colormap[color]
    num = highlight ? num + 10 : num # For highlighting
    attr = bold ? "$num;1" : "$num"
    return "\x1b[" * attr * "m" * string * "\x1b[0m"
end
