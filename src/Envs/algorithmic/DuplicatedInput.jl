"""
Task is to return every nth character from the input tape.
http://arxiv.org/abs/1511.07275
"""
mutable struct DuplicatedInputEnv <: TapeAlgorithmicEnv
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

    duplication::Int

    target::Array{Char, 1}
    input_data::Array{Char, 1}
    time
    read_head_position
    write_head_position
    last_action
    last_reward
end


DuplicatedInputEnv(base::Int=5, duplication::Int=2) =
    DuplicatedInputEnv(-1f0, # MIN_REWARD_SHORTFALL_FOR_PROMOTION
            base, 10, # last
            0f0, # episode_total_reward
            Array{Int, 1}(),  # reward_shortfalls
            ['A'+i for i=0:base-1],  # charmap
            2,  # starting min_length
            TupleSpace([Discrete(2), Discrete(2), Discrete(base)]),  # action_space
            Discrete(base+1),  # observation_space
            (:left, :right), 1,  # MOVEMENTS and READ_HEAD_START
            duplication,
            Array{Char, 1}(),
            Array{Char, 1}(),
            nothing, nothing, nothing, nothing, nothing)


function generateinputdata(dupinpenv::DuplicatedInputEnv, shape)
    res = []
    shape < dupinpenv.duplication &&
        (shape = dupinpenv.duplication)
    for i=1:div(shape, dupinpenv.duplication)
        char = rand(1:dupinpenv.base)
        for _ in 1:dupinpenv.duplication
            push!(res, char)
    end
end


targetfrominputdata(dupinpenv::DuplicatedInputEnv, input_data) =
    [input_data[i] for i=1:dupinpenv.duplication:length(input_data)]
