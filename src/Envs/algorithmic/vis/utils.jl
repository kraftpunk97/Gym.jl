struct HumanCtx <: AbstractCtx end
struct ANSICtx <: AbstractCtx end
struct NoCtx <: AbstractCtx end

render!(env::AbstractEnv, ctx::ANSICtx) = drawcanvas!(env)
render!(env::AbstractEnv, ctx::NoCtx) = nothing
function render!(env::AbstractEnv, ctx::HumanCtx)
    print(drawcanvas!(env))
end
