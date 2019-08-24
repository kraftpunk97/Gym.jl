include("discrete.jl")

mutable struct CliffWalkingEnv <: DiscreteEnv
    shape
    start_state_index

    _cliff
    discenv_obj
end

UP = 1
RIGHT = 2
DOWN = 3
LEFT = 4

@inline ind2sub(index, shape) = Int[((index-1) % shape[1])+1, ceil(index/shape[1])]

function CliffWalkingEnv()
    shape = [4, 12]
    start_state_index = 4

    nS = prod(shape)
    nA = 4

    _cliff = zeros(Bool, shape...)
    _cliff[4, 2:11] .= true

    P = Dict()

    """Determine the outcome for an action. Transition Prob is always 1.0
    returns a tuple of form `(1.0, new_state, reward, done)`"""
    function calculate_transition_prob(current, delta)
        new_position = current .+ delta
        limit_coordinates!(new_position)
        new_state = shape[1] * (new_position[2] - 1) + new_position[1]
        _cliff[new_position...] &&
            return [(1.0, start_state_index, -100, false)]
        terminal_state = shape
        is_done = all(new_position .== terminal_state)
        return [(1.0, new_state, -1, is_done)]
    end

    """Don't fall off..."""
    function limit_coordinates!(coord)
        coord[1] = clamp(coord[1], 1, shape[1])
        coord[2] = clamp(coord[2], 1, shape[2])
    end

    P = Dict()
    for s=1:nS
        position = ind2sub(s, shape)
        P[s] = Dict(a => [] for a=1:nA)
        P[s][UP]    = calculate_transition_prob(position, [-1, 0])
        P[s][RIGHT] = calculate_transition_prob(position, [0, 1] )
        P[s][DOWN]  = calculate_transition_prob(position, [1, 0] )
        P[s][LEFT]  = calculate_transition_prob(position, [0, -1])
    end

    isd = zeros(Float32, nS)
    isd[start_state_index] = 1.0

    discenv_obj = DiscreteEnvObj(nS, nA, P, isd)
    CliffWalkingEnv(shape, start_state_index, _cliff, discenv_obj)
end

function render(env::CliffWalkingEnv)
    output = ""
    for y ∈ 1:env.shape[1]
        output_line = ""
        for x ∈ 1:env.shape[2]
            s = env.shape[1]*(x-1) + y
            if env.s == s
                output_line *= " x "
            elseif (y, x) == (4, 12)
                output_line *= " T "
            elseif env._cliff[y, x]
                output_line *= " C "
            else
                output_line *= " o "
            end

            if x == 1
                output_line = lstrip(output_line)
            end
            if x == 12
                output_line = rstrip(output_line)
                output *= '\n'
            end
        end
        output *= output_line
    end
    println(output)
end
