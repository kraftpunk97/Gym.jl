"""
Task is to reverse content over the input tape.
http://arxiv.org/abs/1511.07275
"""
mutable struct ReverseEnv <: TapeAlgorithmicEnv
    MIN_REWARD_SHORTFALL_FOR_PROMOTION::Float32
    base::Int
    last::Int
    episode_total_reward::Float32
    reward_shortfalls::Array{Int, 1}
    charmap::Array{Char, 1}
    min_length::Int
    action_space::TupleSpace
    observation_space::Discrete

    MOVEMENTS::NTuple
    READ_HEAD_START::Int

    target
    input_data
    time
    read_head_position
    write_head_position
    last_action
    last_reward
end

ReverseEnv(base::Int=2) =
    ReverseEnv(-1f-1, # MIN_REWARD_SHORTFALL_FOR_PROMOTION
            base, 50, # last
            0f0, # episode_total_reward
            Array{Int, 1}(),  # reward_shortfalls
            ['A'+i for i=0:base-1],  # charmap
            1,  # starting min_length
            TupleSpace([Discrete(2), Discrete(2), Discrete(base)]),  # action_space
            Discrete(base+1),  # observation_space
            (:left, :right), 1,  # MOVEMENTS and READ_HEAD_START
            duplication,
            nothing, nothing, nothing, nothing, nothing, nothing, nothing)

targetfrominputdata(revenv::ReverseEnv, input_str) = [char for char in reverse(input_str)]
