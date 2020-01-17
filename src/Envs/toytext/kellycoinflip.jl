using Random
using Distributions: beta
using GymSpaces: Discrete, TupleSpace, Box

@inline flip(edge, seed::MersenneTwister) = rand(seed, Float32) < edge ? 1 : -1

mutable struct KellyCoinFlipEnv <: AbstractEnv
    action_space::Discrete
    observation_space::TupleSpace
    reward_range
    wealth
    seed::MersenneTwister
    rounds
    initial_wealth
    edge
    max_wealth
    max_rounds
end

include("vis/kellycoinflip.jl")

function KellyCoinFlipEnv(;initial_wealth=25f0, edge=6f-1, max_wealth=250f0, max_rounds=300)
    action_space = Discrete(Int(max_wealth * 100)) # Betting in penny increments
    observation_space = TupleSpace([
        Box(0, max_wealth, (1, ), Float32),
        Discrete(max_rounds + 1)
    ])
    reward_range = (0, max_wealth)
    wealth = initial_wealth
    seed = MersenneTwister()
    rounds = max_rounds
    KellyCoinFlipEnv(action_space, observation_space, reward_range, wealth, seed,
                     rounds, initial_wealth, max_wealth, max_rounds)
end

function reset!(env::KellyCoinFlipEnv)
    env.rounds = env.max_rounds
    env.wealth = env.initial_wealth
    return _get_obs(env)
end

_get_obs(env::KellyCoinFlipEnv) = [env.wealth], env.rounds

function step!(env::KellyCoinFlipEnv, action)
    bet_in_dollars = min(action/100.0, env.wealth)
    env.rounds -= 1

    coinflip = flip(env.edge, env.seed)
    env.wealth = min(env.max_wealth, env.wealth + coinflip *  bet_in_dollars)

    done = env.wealth < 0.01 || env.wealth == env.max_wealth || !Bool(env.rounds)
    reward = done ? env.wealth : 0f0

    return _get_obs(env), reward, done, Dict()
end

function drawcanvas!(env::KellyCoinFlipEnv)
    return "Current wealth: $(env.wealth); Rounds left: $(env.rounds)"
end

function seed!(env::KellyCointFlip, seed::Unsigned)
    env.seed = MersenneTwister(seed)
    return nothing
end
