using GymSpaces: Discrete

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
    action_space::Discrete
    observation_space::Discrete
end

include("vis/taxi.jl")

"""
The Taxi Problem
from "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition"
by Tom Dietterich

Description:
There are four designated locations in the grid world indicated by R(ed), B(lue), G(reen), and Y(ellow). When the episode starts, the taxi starts off at a random square and the passenger is at a random location. The taxi drive to the passenger's location, pick up the passenger, drive to the passenger's destination (another one of the four specified locations), and then drop off the passenger. Once the passenger is dropped off, the episode ends.

Observations:
There are 500 discrete states since there are 25 taxi positions, 5 possible locations of the passenger (including the case when the passenger is the taxi), and 4 destination locations.

Actions:
There are 6 discrete deterministic actions:
    - 1: move south
    - 2: move north
    - 3: move east
    - 4: move west
    - 5: pickup passenger
    - 6: dropoff passenger

Rewards:
There is a reward of -1 for each action and an additional reward of +20 for delievering the passenger. There is a reward of -10 for executing actions "pickup" and "dropoff" illegally.


Rendering:
    - blue: passenger
    - magenta: destination
    - yellow: empty taxi
    - green: full taxi
    - other letters (R, G, B and Y): locations for passengers and destinations

actions:
    - 1: south
    - 2: north
    - 3: east
    - 4: west
    - 5: pickup
    - 6: dropoff

    state space is represented by:
        (taxi_row, taxi_col, passenger_location, destination)
"""
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
    action_space = Discrete(num_actions)
    observation_space = Discrete(num_states)
    TaxiEnv(desc, locs, discenv_obj, action_space, observation_space)
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

function drawcanvas!(env::TaxiEnv)
    actionarr = ["South", "North", "East", "West", "Pickup", "Dropoff"]
    outfile = ""

    out = ["$char" for char in env.desc]
    taxi_row, taxi_col, pass_idx, dest_idx = decode(env.s)

    ul(x) = x == " " ? "_" : x

    if pass_idx ≤ 4
        out[1 + taxi_row, 2 * (taxi_col-1) + 2] = colorize(
                    out[1 + taxi_row, 2 * (taxi_col-1) + 2], :yellow, highlight=true)
        pi, pj = env.locs[pass_idx]
        out[1 + pi, 2 * (pj-1) + 2] = colorize(ul(out[1 + pi, 2 * (pj-1) + 2]), :blue, bold=true)
    else
        out[1 + taxi_row, 2 * (taxi_col-1) + 2] = colorize_bold(
                     out[1 + taxi_row, 2 * (taxi_col-1) + 2], :green, highlight=true)
    end
    di, dj = env.locs[dest_idx]
    out[1 + di, 2 * (dj-1) + 2] = colorize(out[1 + di, 2 * (dj-1) + 2], :magenta)
    outfile *= join([join(out[row, :], "") for row in 1:size(env.desc, 1)], "\n")
    outfile *= isnothing(env.lastaction) ? "\n" : " ($(actionarr[env.lastaction]))\n"

    return outfile
end
