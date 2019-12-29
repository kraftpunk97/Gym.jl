using StatsBase: sample
using GymSpaces: Discrete

include("discrete.jl")

const LEFT = 1
const DOWN = 2
const RIGHT = 3
const UP = 4

mutable struct FrozenLakeEnv <: DiscreteEnv
    desc::Array{Char, 2}
    nrow::Int
    ncol::Int
    reward_range
    discenv_obj::DiscreteEnvObj
    action_space::Discrete
    observation_space::Discrete
end

include("vis/frozen_lake.jl")

MAPS = Dict(
    :fourxfour => [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ],

    :eightxeight => [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG"
    ]
)

"""
    generate_random_map(size::Integer, prob::AbstractFloat)

Generates a random valid map of size `size` and `prob` determines the probability
of a tile being frozen.
"""
function generate_random_map(shape=8, p=8f-1)
    valid = false

    # BFS to check that it's a valid path...
    function is_valid(arr, r=1, c=1)
        if arr[r, c] == 'G'
            return true end

        tmp = arr[r, c]
        arr[r, c] = '#'

        if r+1 ≤ shape && !(arr[r+1, c] ∈ "#H")
            if is_valid(arr, r+1, c)
                arr[r, c] = tmp
                return true end end

        if c+1 ≤ shape && !(arr[r, c+1] ∈ "#H")
            if is_valid(arr, r, c+1)
                arr[r, c] = tmp
                return true end end

        if r-1 ≥ 1 && !(arr[r-1, c] ∈ "#H")
            if is_valid(arr, r-1, c)
                arr[r, c] = tmp
                return true end end

        if c-1 ≥ 1 && !(arr[r, c-1] ∈ "#H")
            if is_valid(arr, r, c-1)
                arr[r, c] = tmp
                return true end end

        arr[r, c] = tmp
        return false
    end

    res = Array{Char, 2}(undef, shape, shape)
    while !valid
        p = min(1, p)
        res = StatsBase.sample(['F', 'H'], ProbabilityWeights(Float32[p, 1-p]), (shape, shape), replace=true)
        res[1, 1] = 'S'
        res[end, end] = 'G'
        valid = is_valid(res)
    end
    return [join(res[x, :]) for x in 1:size(res, 1)]
end

function FrozenLakeEnv(desc_::AbstractArray{<:AbstractString, 1}, is_slippery::Bool)
    desc = Char.(vcat([Int.(collect(row))' for row in desc_]...))

    nrow, ncol = size(desc)

    reward_range = [0, 1]

    nA = 4
    nS = nrow * ncol

    isd = Float64.(desc .== 'S')
    isd ./= sum(isd)
    isd = reshape(isd, length(isd))

    P = Dict(s => Dict(a => [] for a=1:nA) for s=1:nS)

    @inline to_s(row, col) = (col-1)*nrow + row

    function inc(row, col, a)
        if a == LEFT
            col = max(col-1, 1)
        elseif a == RIGHT
            col = min(col+1, ncol)
        elseif a == DOWN
            row = min(row+1, nrow)
        elseif a == UP
            row = max(row-1, 1)
        end
        return (row, col)
    end

    for row=1:nrow
        for col=1:ncol
            s = to_s(row, col)
            for a=1:4
                li = P[s][a]
                letter = desc[row, col]
                if letter ∈ "GH"
                    push!(li, (1.0, s, 0, true))
                else
                    if is_slippery
                        for b ∈[((a-2)%4)+1, a, ((a%4)+1)]
                            newrow, newcol = inc(row, col, b)
                            newstate = to_s(newrow, newcol)
                            newletter = desc[newrow, newcol]
                            done = newletter == 'G' || newletter == 'H'
                            rew = Float32(newletter == 'G')
                            push!(li, (1f0/3f0, newstate, rew, done))
                        end
                    else
                        newrow, newcol = inc(row, col, a)
                        newstate = to_s(newrow, newcol)
                        newletter = desc[newrow, newcol]
                        done = newletter == 'G' || newletter == 'H'
                        rew = Float32(newletter == 'G')
                        push!(li, (1f0, newstate, rew, done))
                    end
                end
            end
        end
    end
    discenv_obj = DiscreteEnvObj(nS, nA, P, isd)
    action_space = Discrete(nA)
    observation_space = Discrete(nS)
    FrozenLakeEnv(desc, nrow, ncol, reward_range, discenv_obj, action_space, observation_space)
end

function FrozenLakeEnv(; map_name::Union{Symbol,Nothing}=nothing, is_slippery::Bool=true)
    desc = isnothing(map_name) ? generate_random_map() : MAPS[map_name]
    FrozenLakeEnv(desc, is_slippery)
end

function drawcanvas!(env::FrozenLakeEnv)
    actionarr = ["Left", "Down", "Right", "Up"]
    outfile = ""
    row, col = ((env.s-1)÷env.ncol) + 1, (env.s-1)%env.ncol + 1
    desc = ["$char" for char in env.desc]
    desc[col, row] = colorize(desc[row, col], :red, highlight=true)
    outfile *= isnothing(env.lastaction) ? "\n" : " ($(actionarr[env.lastaction]))\n"
    outfile *= join([join(desc[row, :], "") for row in 1:env.nrow], "\n")
    return outfile
end
