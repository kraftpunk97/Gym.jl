include("discrete.jl")

MAP = [
    "+---------+",
    "|R: | : :G|",
    "| : : : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
]

mutable struct TaxiEnv <: DiscreteEnv
    desc
    locs
    discenv_obj::DiscreteEnvObj
end

function TaxiEnv()
    desc = Char.(vcat([Int.(collect(row))' for row in MAP]...))

    locs = [[1, 1], [1, 5], [5, 1], [5, 4]]

    num_states = 500
    num_rows = 5
    num_columns = 5
    max_row = num_rows
    max_col = num_columns
    initial_state_distrib = zeros(Float32, num_states)
    num_actions = 6
    P = Dict(state => Dict(action => [] for action=1:num_actions) for state=1:num_states)
    for row=1:num_rows
        for col=1:num_columns
            for pass_idx=1:length(locs)+1
                for dest_idx=1:length(locs)
                    state = encode(row, col, pass_idx, dest_idx)
                    if pass_idx ≤ 4 && pass_idx != dest_idx
                        initial_state_distrib[state] += 1
                    end
                    for action=1:num_actions
                        # defaults
                        new_row, new_col, new_pass_idx = row, col, pass_idx
                        reward = -1
                        done = false
                        taxi_loc = [row, col]

                        if action == 1
                            new_row = min(row + 1, max_row)
                        elseif action == 2
                            new_row = max(row - 1, 1)
                        end

                        if action == 3 && desc[row+1, 2*(col-1) + 3] == ':'
                            new_col = min(col + 1, max_col)
                        elseif action == 4 && desc[row+1, 2 * (col-1) + 1] == ':'
                            new_col = max(col - 1, 1)
                        elseif action == 5 # pickup
                            if pass_idx ≤ 4 && taxi_loc == locs[pass_idx]
                                new_pass_idx = 5
                            else # passenger not at location
                                reward = -10
                            end
                        elseif action == 6 # dropoff
                            if (taxi_loc == locs[dest_idx]) && pass_idx == 5
                                new_pass_idx = dest_idx
                                done = true
                                reward = 20
                            elseif taxi_loc ∈ locs && pass_idx == 4
                                new_pass_idx = findall(x -> x == taxi_loc, locs)[1]
                            else
                                reward = -10
                            end
                        end
                        new_state = encode(new_row, new_col, new_pass_idx, dest_idx)
                        push!(P[state][action], (1.0, new_state, reward, done))
                    end
                end
            end
        end
    end
    initial_state_distrib ./= sum(initial_state_distrib)
    discenv_obj = DiscreteEnvObj(num_states, num_actions, P, initial_state_distrib)
    TaxiEnv(desc, locs, discenv_obj)
end

function encode(taxi_row, taxi_col, pass_loc, dest_idx)
    i = taxi_row - 1
    i *= 5
    i += taxi_col - 1
    i *= 5
    i += pass_loc - 1
    i *= 4
    i += dest_idx
    return i
end

function decode(i)
    i -= 1
    out = []
    push!(out, i%4+1)
    i = i ÷ 4
    push!(out, i%5+1)
    i = i ÷ 5
    push!(out, i%5+1)
    i = i ÷ 5
    @assert 1 ≤ i+1 ≤ 5
    push!(out, i+1)
    reverse!(out)
    return out
end
