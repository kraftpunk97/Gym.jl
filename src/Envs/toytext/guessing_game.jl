using Random
using GymSpaces: Box, Discrete

mutable struct GuessingGameEnv <: AbstractEnv
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

include("vis/guessing_game.jl")

function GuessingGameEnv()
    seed = MersenneTwister()
    range = 1000
    bounds = 10000

    action_space = Box([-bounds], [bounds], Float32)
    observation_space = Discrete(4)

    number = 2range * rand(Float32) - range
    guess_count = 0
    guess_max = 200
    observation = 0
    GuessingGameEnv(range, bounds, action_space, observation_space, number,
                    guess_count, guess_max, observation, seed)
end

seed!(env::GuessingGameEnv) = (env.seed = MersenneTwister())
seed!(env::GuessingGameEnv, int) = (env.seed = MersenneTwister(int))

function reset!(env::GuessingGameEnv)
    env.number = 2env.range * rand(Float32) - env.range
    env.guess_count = 0
    env.observation = 0
    return env.observation
end

function step!(env::GuessingGameEnv, action)
    @assert action ∈ env.action_space

    if action < env.number
        env.observation = 1
    elseif action == env.number
        env.observation = 2
    elseif action == env.number
        env.observation = 3
    end

    reward = 0
    done = false

    if (env.number - env.range * 0.01) < action < (env.number + env.range * 0.01)
        reward = 1
        done = true
    end

    env.guess_count += 1
    if env.guess_count ≥ env.guess_max
        done = true
    end

    return env.observation, reward, done, Dict(:number => env.number, :guesses => env.guess_count)
end

function drawcanvas!(env::GuessingGameEnv)
    return env.observation
end

_get_obs(env::GuessingGameEnv) = env.observation
