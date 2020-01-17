using GymSpaces: TupleSpace, Discrete

#=
Algorithmic environments have the following traits in common:

- A 1-d "input tape" or 2-d "input grid" of characters
- A target string which is a deterministic function of the input characters

Agents control a read head that moves over the input tape. Observations consist
of the single character currently under the read head. The read head may fall
off the end of the tape in any direction. When this happens, agents will observe
a special blank character (with index=0) until they get back in bounds.

Actions consist of 3 sub-actions:
    - Direction to move the read head (left or right, plus up and down for 2-d envs)
    - Whether to write to the output tape
    - Which character to write (ignored if the above sub-action is 0)

An episode ends when:
    - The agent writes the full target string to the output tape.
    - The agent writes an incorrect character.
    - The agent runs out the time limit. (Which is fairly conservative.)

Reward schedule:
    write a correct character: +1
    write a wrong character: -.5
    run out the clock: -1
    otherwise: 0

In the beginning, input strings will be fairly short. After an environment has
been consistently solved over some window of episodes, the environment will
increase the average length of generated strings. Typical env specs require
leveling up many times to reach their reward threshold.
=#

abstract type AlgorithmicEnv <: AbstractEnv end
abstract type TapeAlgorithmicEnv <: AlgorithmicEnv end
abstract type GridAlgorithmicEnv <: AlgorithmicEnv end

include("vis/algorithmicenv.jl")

#=
Called between episodes. Update our running record of episode rewards and,
if appropriate, 'level up' minimum input length.
=#
function check_level_up!(env::AlgorithmicEnv)
    push!(env.reward_shortfalls, env.episode_total_reward - length(env.target))
    if length(env.reward_shortfalls) == env.reward_shortfalls.capacity &&
       minimum(env.reward_shortfalls) >= env.MIN_REWARD_SHORTFALL_FOR_PROMOTION &&
       env.min_length < 30
       env.min_length += 1
       env.reward_shortfalls = CircularBuffer{Float32}(10)
   end
end

function seed!(env::AlgorithmicEnv, seed::Unsigned)
    env.seed = MersenneTwister(seed)
    return nothing
end

function reset!(env::AlgorithmicEnv)
    check_level_up!(env)
    env.last_action = nothing
    env.last_reward = 0
    env.read_head_position = env.READ_HEAD_START
    env.write_head_position = 1
    env.episode_total_reward = 0f0
    env.time = 0
    len = rand(env.seed, 0:2) + env.min_length
    generate_input_data!(env, len)
    target_from_input_data!(env)
    _get_obs(env)
end


function step!(env::AlgorithmicEnv, action)
    @assert action ∈ env.action_space "$(action) is not a valid action for this environment."
    env.last_action = action
    inp_act = action[1:1]
    out_act = action[2:2]
    pred = action[3:3]
    done = false
    reward = 0f0
    env.time += 1  # Clock ticks one time
    @assert 1 <= env.write_head_position  # Ensure that we have someplace to write to
    if out_act[1] == 2
        correct = false  # initialising 'correct'
        try
            correct = pred[1] == env.target[env.write_head_position]
        catch BoundsError
            @warn "It looks like you're calling step() even though this " *
            "environement has already returned done=true. You should " *
            "always call reset!() once you recieve done=true. Any " *
            "further steps are undefined behaviour."
            correct = false
        end
        reward += correct ? 1f0 : -5f-1  # Calculate reward.
        !correct && (done = true)  # Bail immediately if wrong character is written.

        env.write_head_position += 1
        env.write_head_position > length(env.target) && (done = true)
    end
    move!(env, inp_act)
    # If an agent takes more than this many timesteps, end the episode
    # immediately and return negative reward
    timelimit = time_limit(env)
    if env.time > timelimit
        reward = 0reward - 1f0
        done = true
    end
    obs = _get_obs(env)
    env.last_reward = reward
    env.episode_total_reward += reward
    return (obs, reward, done, Dict())
end

generate_input_data!(env::TapeAlgorithmicEnv, size_) =
    env.input_data = [rand(env.seed, 1:env.base) for _ in 1:size_]
generate_input_data!(env::GridAlgorithmicEnv, size_) =
    env.input_data = [[rand(env.seed, 1:env.base) for _  in 1:env.rows] for __ in 1:size_]


function move!(env::TapeAlgorithmicEnv, movement)
    named = env.MOVEMENTS[movement[1]]
    env.read_head_position += named == :right ? 1 : -1
end

function move!(env::GridAlgorithmicEnv, movement)
    named = env.MOVEMENTS[movement[1]]
    x, y = env.read_head_posiion
    if named == :left
        x -= 1
    elseif named == :right
        x += 1
    elseif named == :up
        y -= 1
    elseif named == :down
        y += 1
    else
        throw(ArgumentError("Unrecognized direction: $(named)"))
    end
    env.read_head_posiion = [x, y]
end


# return character under read_head/pos. If the read_head/pos is out-of-bounds, then return null(0).
_get_obs(env::AlgorithmicEnv) = _get_obs(env, env.read_head_position)
function _get_obs(env::TapeAlgorithmicEnv, pos::Integer)
    try
        return [env.input_data[pos]]
    catch BoundsError
        return [env.base + 1]
    end
end

function _get_obs(env::GridAlgorithmicEnv, pos::Array)
    x, y = pos
    try
        return [env.input_data[x][y]]
    catch BoundsError
        return [env.base + 1]
    end
end

time_limit(env::TapeAlgorithmicEnv) = length(env.input_data) + length(env.target) + 4
