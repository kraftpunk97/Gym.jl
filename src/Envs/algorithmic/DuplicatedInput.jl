using DataStructures: CircularBuffer
include("AlgorithmicEnv.jl")

"""
Task is to return every nth character from the input tape.
http://arxiv.org/abs/1511.07275
"""
mutable struct DuplicatedInputEnv <: TapeAlgorithmicEnv
    MIN_REWARD_SHORTFALL_FOR_PROMOTION::Float32
    base::Int
    episode_total_reward::Float32
    reward_shortfalls::CircularBuffer{Float32}
    charmap::Array{Char, 1}
    min_length::Int
    action_space::TupleSpace
    observation_space::Discrete

    MOVEMENTS::NTuple
    READ_HEAD_START::Int

    duplication::Int

    target::Array{Int8, 1}  # target tape
    input_data::Array{Int8, 1}
    time::Int8
    read_head_position::Int8  # current position of the read head.
    write_head_position::Int8
    last_action
    last_reward::Float32
end


DuplicatedInputEnv(base::Int=5, duplication::Int=2) =
    DuplicatedInputEnv(-1f0, # MIN_REWARD_SHORTFALL_FOR_PROMOTION
            base,
            0f0, # episode_total_reward
            CircularBuffer{Float32}(10),  # reward_shortfalls
            push!(['A'+i for i=0:base-1], ' '),  # charmap
            2,  # starting min_length
            TupleSpace([Discrete(2), Discrete(2), Discrete(base)]),  # action_space
            Discrete(base+1),  # observation_space
            (:left, :right), 1,  # MOVEMENTS and READ_HEAD_START
            duplication,
            Int8[],
            Int8[],
            0, 1, 1, nothing, 0.0)


function generate_input_data!(env::DuplicatedInputEnv, size_)
    res = []
    size_ < env.duplication &&
        (size_ = env.duplication)
    for i=1:div(size_, env.duplication)
        char = rand(1:env.base)
        for _ in 1:env.duplication
            push!(res, char)
        end
    end
    env.input_data = res
end


target_from_input_data!(env::DuplicatedInputEnv) =
    env.target = [env.input_data[i] for i=1:env.duplication:length(env.input_data)]
