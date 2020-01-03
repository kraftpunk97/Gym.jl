using Random
using DataStructures: CircularBuffer
include("AlgorithmicEnv.jl")

mutable struct ReversedAdditionEnv <: GridAlgorithmicEnv
    MIN_REWARD_SHORTFALL_FOR_PROMOTION::Float32
    base::Int
    episode_total_reward::Float32
    reward_shortfalls::CircularBuffer{Float32}
    charmap::Array{Char, 1}
    min_length::Int
    action_space::TupleSpace
    observation_space::Discrete

    MOVEMENTS::NTuple
    READ_HEAD_START::Array{Int8, 1}

    rows::Int

    target
    input_data
    time
    read_head_position
    write_head_position
    last_action
    last_reward
    seed::MersenneTwister
end

ReversedAdditionEnv(base::Int=3; rows::Int=2) =
    ReversedAdditionEnv(-1f0, # MIN_REWARD_SHORTFALL_FOR_PROMOTION
            base, # last
            0f0, # episode_total_reward
            CircularBuffer{Float32}(10),  # reward_shortfalls
            push!(['0'+i for i=0:base-1], ' '),  # charmap
            2,  # starting min_length
            TupleSpace([Discrete(2), Discrete(2), Discrete(base)]),  # action_space
            Discrete(base+1),  # observation_space
            (:left, :right, :up, :down), [1, 1],  # MOVEMENTS and READ_HEAD_START
            rows,
            Int8[], Int8[], 0, [1, 1], [1, 1], nothing, 0.0, MersenneTwister())

function target_from_input_data!(env::ReversedAdditionEnv)
    curry = 0
    env.target = []
    for digits in env.input_data
        total = sum(digits) + curry
        push!(env.target, total % env.base)
        curry = div(total, env.base)
    end
    curry > 0 && (push!(env.target, curry))
end

time_limit(env::ReversedAdditionEnv) = length(env.input_data) * 4
