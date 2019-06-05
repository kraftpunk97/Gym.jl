mutable struct ReversedAdditionEnv <: GridAlgorithmicEnv
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
    READ_HEAD_START::NTuple

    rows::Int

    target
    input_data
    time
    read_head_position
    write_head_position
    last_action
    last_reward
end

ReversedAdditionEnv(base::Int=3, rows::Int=2) =
    ReversedAdditionEnv(-1f0, # MIN_REWARD_SHORTFALL_FOR_PROMOTION
            base, 10, # last
            0f0, # episode_total_reward
            Array{Int, 1}(),  # reward_shortfalls
            ['0'+i for i=0:base-1],  # charmap
            2,  # starting min_length
            TupleSpace([Discrete(2), Discrete(2), Discrete(base)]),  # action_space
            Discrete(base+1),  # observation_space
            (:left, :right, :up, :down), [1, 1],  # MOVEMENTS and READ_HEAD_START
            rows,
            nothing, nothing, nothing, nothing, nothing, nothing, nothing)

function targetfrominputdata(revaddenv::ReversedAdditionEnv, input_strings)
    curry = 0
    target = []
    for digits in input_strings
        total = sum(digits) + curry
        push!(total % revaddenv.base)
        curry = div(total, revaddenv.base)
    end
    curry > 0 && (push!(target, curry))
    return target
end
