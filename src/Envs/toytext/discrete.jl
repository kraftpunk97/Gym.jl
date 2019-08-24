using Random
using GymSpaces: Discrete
abstract type DiscreteEnv <: AbstractEnv end

function categorical_sample(prob_n, seed)
    csprob_n = cumsum(prob_n)
    return argmax(csprob_n .> rand(Float32))
end

mutable struct DiscreteEnvObj
    P
    isd
    lastaction
    nS
    nA

    action_space::Discrete
    observation_space::Discrete

    s
    seed::MersenneTwister
end

function DiscreteEnvObj(nS, nA, P, isd)
    action_space = Discrete(nA)
    observation_space = Discrete(nS)
    seed = MersenneTwister()

    s = categorical_sample(isd, seed)
    DiscreteEnvObj(P, isd, nothing, nS, nA, action_space, observation_space, s, seed)
end

# All DiscreteEnv must have a `discenv_obj` field...
function Base.getproperty(env::DiscreteEnv, sym::Symbol)
    if sym == :P
        return env.discenv_obj.P
    elseif sym == :isd
        return env.discenv_obj.isd
    elseif sym == :lastaction
        return env.discenv_obj.lastaction
    elseif sym == :nS
        return env.discenv_obj.nS
    elseif sym == :nA
        return env.discenv_obj.nA
    elseif sym == :action_space
        return env.discenv_obj.action_space
    elseif sym == :observation_space
        return env.discenv_obj.observation_space
    elseif sym == :s
        return env.discenv_obj.s
    elseif sym == :seed
        return env.discenv_obj.seed
    else
        return Base.getfield(env, sym)
    end
end

seed!(env::DiscreteEnv) = (env.seed = MersenneTwister())
seed!(env::DiscreteEnv, int) = (env.seed = MersenneTwister(int))

function reset!(env::DiscreteEnv)
    env.discenv_obj.s = categorical_sample(env.isd, env.seed)
    env.discenv_obj.lastaction = nothing
    return env.s
end

function step!(env::DiscreteEnv, action)
    @assert action ∈ env.action_space "Invalid action"
    transitions = env.discenv_obj.P[env.s][action]
    i = categorical_sample([t[1] for t ∈ transitions], env.seed)
    p, s, r, d = transitions[i]
    env.discenv_obj.s = s
    env.discenv_obj.lastaction = action
    return s, r, d, Dict(:prob => p)
end
