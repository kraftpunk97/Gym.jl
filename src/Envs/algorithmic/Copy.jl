"""
Task is to copy content from the input tape to
the output tape. http://arxiv.org/abs/1511.07275
"""

include("AlgorithmicEnv.jl")

RealOrNothing = Union{Float32, Nothing}

mutable struct CopyEnv <: TapeAlgorithmicEnv
    MIN_REWARD_SHORTFALL_FOR_PROMOTION::Float32
    base::Int8
    last::Int8
    episode_total_reward::RealOrNothing
    reward_shortfalls::Array{Float32, 1}
    charmap::Array{Char, 1}
    min_length::Int8
    action_space::TupleSpace
    observation_space::Discrete

    MOVEMENTS::NTuple
    READ_HEAD_START::Int8

    target::Array{N, 1} where N <: Union{Char, Integer}
    input_data::Array{N, 1} where N <: Union{Char, Integer}
    time::Int8  # pre-init = 0
    read_head_position::Int8  # pre-init = 0
    write_head_position::Int8  # pre-init = 0
    last_action
    last_reward::Float32 # pre-init = 0
end

function CopyEnv(base::Int=5, chars::Bool=true)
    starting_char = chars ? 'A' : '0'
    CopyEnv(-1f0, # MIN_REWARD_SHORTFALL_FOR_PROMOTION
            base, 10, # last
            0f0, # episode_total_reward
            Array{Float32, 1}(),  # reward_shortfalls
            push!([starting_char+i for i=0:base-1], ' '),  # charmap
            2,  # starting min_length
            TupleSpace([Discrete(2), Discrete(2), Discrete(base)]),  # action_space
            Discrete(base+1),  # observation_space
            (:left, :right), 1,  # MOVEMENTS and READ_HEAD_START
            Array{(chars ? Char : Int), 1}(), # target
            Array{(chars ? Char : Int), 1}(), # input_data
            0,  # time
            1,  # read_head_posiion
            1,  # write_head_position
            nothing,  # last_action
            0.0)  # last_reward
end

targetfrominputdata(cpenv::CopyEnv) = cpenv.input_data

function Ctx(cpenv::CopyEnv, render_mode::Symbol=:human_window)
    return NoCtx()
end
