using DataStructures: CircularBuffer
include("AlgorithmicEnv.jl")

"""
Task is to reverse content over the input tape.
http://arxiv.org/abs/1511.07275
"""
mutable struct ReverseEnv <: TapeAlgorithmicEnv
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

    target::Array{Int8, 1}  # target tape
    input_data::Array{Int8, 1}
    time::Int8
    read_head_position::Int8  # current position of the read head.
    write_head_position::Int8
    last_action
    last_reward::Float32
end

ReverseEnv(base::Int=2) =
    ReverseEnv(-1f-1, # MIN_REWARD_SHORTFALL_FOR_PROMOTION
            base, # last
            0f0, # episode_total_reward
            CircularBuffer{Float32}(50),  # reward_shortfalls
            push!(['A'+i for i=0:base-1], ' '),  # charmap
            1,  # starting min_length
            TupleSpace([Discrete(2), Discrete(2), Discrete(base)]),  # action_space
            Discrete(base+1),  # observation_space
            (:left, :right), 1,  # MOVEMENTS and READ_HEAD_START
            Int8[], Int8[], 0, 1, 1, nothing, 0.0)

target_from_input_data!(env::ReverseEnv) =
    env.target = [char for char in reverse(env.input_data)]
