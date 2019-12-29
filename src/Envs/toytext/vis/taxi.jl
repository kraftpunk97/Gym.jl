include("utils.jl")

function Ctx(env::TaxiEnv, mode::Symbol=:human)
    if mode == :human
        HumanCtx()
    elseif mode == :ansi
        ANSICtx()
    elseif mode == :no_render
        NoCtx()
    else
        error("Unrecognized mode in Ctx(): $(mode)")
    end
end
