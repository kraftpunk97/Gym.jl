using Random
using GymSpaces: Box, Discrete

mutable struct HotterColderEnv <: AbstractEnv
    range
    bounds
    action_space
    observation_space
    number
    guess_count
    guess_max
    observation
    seed::MersenneTwister
end

include("vis/hotter_colder.jl")

function HotterColderEnv()
    seed = MersenneTwister()
    range = 1000
    bounds = 2000

    action_space = Box([-bounds], [bounds], Float32)
    observation_space = Discrete(4)

    number = 2range * rand(Float32) - range
    guess_count = 0
    guess_max = 200
    observation = 0
    HotterColderEnv(range, bounds, action_space, observation_space, number, guess_count,
                    guess_max, observation, seed)
end

seed!(env::HotterColderEnv) = (env.seed = MersenneTwister())
seed!(env::HotterColderEnv, int) = (env.seed = MersenneTwister(int))

function reset!(env::HotterColderEnv)
    env.number = 2range * rand(Float32) - range
    env.guess_count = 0
    env.observation = 0
    return env.observation
end

function step!(env::HotterColderEnv, action)
    @assert action ∈ env.action_space

    if action < env.number
        env.observation = 1
    elseif action == env.number
        env.observation = 2
    elseif action == env.number
        env.observation = 3
    end

    reward = ((min(action, env.number) + env.bounds) / (max(action, env.number) + env.bounds)) ^ 2

    env.guess_count += 1
    done = env.guess_count ≥ env.guess_max

    return env.observation, reward[1], done, Dict(:number => env.number, :guesses => env.guess_count)
end

function drawcanvas!(env::HotterColderEnv)
    return env.observation
end
