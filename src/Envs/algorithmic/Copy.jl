"""
Task is to copy content from the input tape to
the output tape. http://arxiv.org/abs/1511.07275
"""
using DataStructures: CircularBuffer
include("AlgorithmicEnv.jl")

RealOrNothing = Union{Float32, Nothing}

mutable struct CopyEnv <: TapeAlgorithmicEnv
    MIN_REWARD_SHORTFALL_FOR_PROMOTION::Float32
    base::Int8  # Number of distinct characters
    episode_total_reward::RealOrNothing  # Cumulative reward earned this episode

    # Running tally of reward shortfalls. eg If there were 10 points to earn
    # and we got 8, we'd append -2
    reward_shortfalls::CircularBuffer{Float32}

    charmap::Array{Char, 1}  # Array of all the characters used while rendering
    min_length::Int8  # Minimum length of the input tape
    action_space::TupleSpace
    observation_space::Discrete

    MOVEMENTS::NTuple # Possible directions to move in.
    READ_HEAD_START::Int8  # Starting position of the read head

    target::Array{N, 1} where N <: Union{Char, Integer}  # target tape
    input_data::Array{N, 1} where N <: Union{Char, Integer}
    time::Int8
    read_head_position::Int8  # current position of the read head.
    write_head_position::Int8
    last_action
    last_reward::Float32
end

function CopyEnv(base::Int=5, chars::Bool=true)
    starting_char = chars ? 'A' : '0'
    CopyEnv(-1f0, # MIN_REWARD_SHORTFALL_FOR_PROMOTION
            base,
            0f0, # episode_total_reward
            CircularBuffer{}(10),  # reward_shortfalls
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
