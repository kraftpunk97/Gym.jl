using .Space: TupleSpace, Discrete

#=
Algorithmic environments have the following traits in common:

- A 1-d "input tape" or 2-d "input grid" of characters
- A target string which is a deterministic function of the input characters

Agents control a read head that moves over the input tape. Observations consist
of the single character currently under the read head. The read head may fall
off the end of the tape in any direction. When this happens, agents will observe
a special blank character (with index=env.base) until they get back in bounds.

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


getstrobs(algoenv::AlgorithmicEnv, pos::Int=-1) = algoenv.charmap[_get_obs(algoenv, pos)]


#=
Return the ith character of the target string (or " " if index out of bounds)
=#
getstrtarget(algoenv::AlgorithmicEnv, pos::Int) = pos < 1 || length(algoenv.target) <= pos ?
    ' ' : algoenv.charmap[algoenv.target[pos]]


#=
Called between episodes. Update our running record of episode rewards and,
if appropriate, 'level up' minimum input length.
=#
function checklevelup!(algoenv::AlgorithmicEnv)
    push!(algoenv.reward_shortfalls, algoenv.episode_total_reward - length(algoenv.target))
    # Keep `algoenv.last` number of rewards. If number of rewards is less, then
    # keep all the rewards.
    last_elem_idx = length(algoenv.reward_shortfalls) >= algoenv.last ?
        algoenv.last : length(algoenv.reward_shortfalls)
    algoenv.reward_shortfalls = algoenv.reward_shortfalls[(end-last_elem_idx+1):end]

    if length(algoenv.reward_shortfalls) == algoenv.last &&
       minimum(algoenv.reward_shortfalls) >= algoenv.MIN_REWARD_SHORTFALL_FOR_PROMOTION &&
       algoenv.min_length < 30
       algoenv.min_length += 1
       algoenv.reward_shortfalls = []
   end
end


function reset!(algoenv::AlgorithmicEnv)
    checklevelup!(algoenv)
    algoenv.last_action = nothing
    algoenv.last_reward = 0
    algoenv.read_head_position = algoenv.READ_HEAD_START
    algoenv.write_head_position = 1
    algoenv.episode_total_reward = 0f0
    algoenv.time = 0
    len = rand(0:2) + algoenv.min_length
    algoenv.input_data = generateinputdata(algoenv, len)
    algoenv.target = targetfrominputdata(algoenv)
    _get_obs(algoenv)
end


function step!(algoenv::AlgorithmicEnv, action)
    @assert action ∈ algoenv.action_space "$(action)"
    algoenv.last_action = action
    inp_act = action[1:1]
    out_act = action[2:2]
    pred = action[3:3]
    done = false
    reward = 0f0
    algoenv.time += 1  # Clock ticks one time
    @assert 1 <= algoenv.write_head_position  # Ensure that we have someplace to write to
    if out_act[1] == 2
        correct = false  # initialising 'correct'
        try
            correct = pred == algoenv.target[algoenv.write_head_position]
        catch BoundsError
            @warn "It looks like you're calling step() even though this " *
            "environement has already returned done=true. You should " *
            "always call reset!() once you recieve done=true. Any " *
            "further steps are undefined behaviour."
            correct = false
        end
        reward = correct ? 1f0 : -5f-1  # Calculate reward.
        !correct && (done = true)  # Bail immediately if wrong character is written.

        algoenv.write_head_position += 1
        algoenv.write_head_position >= length(algoenv.target) && (done = true)
    end
    _move!(algoenv, inp_act)
    # If an agent takes more than this many timesteps, end the episode
    # immediately and return negative reward
    timelimit = length(algoenv.input_data) + length(algoenv.target) + 4
    if algoenv.time > timelimit
        reward = -1f0
        done = true
    end
    obs = _get_obs(algoenv)
    algoenv.last_reward = reward
    algoenv.episode_total_reward += reward
    return (obs, reward, done, Dict())
end

generateinputdata(tapeenv::TapeAlgorithmicEnv, shape) = [rand(1:tapeenv.base) for _ in 1:shape]
generateinputdata(gridenv::GridAlgorithmicEnv, shape) = [[rand(1:gridenv.base) for _  in 1:gridenv.rows] for __ in 1:shape]

function _move!(tapeenv::TapeAlgorithmicEnv, movement)
    named = tapeenv.MOVEMENTS[movement[1]]
    tapeenv.read_head_position += named == :right ? 1 : -1
end

function _move!(gridenv::GridAlgorithmicEnv, movement)
    named = gridenv.MOVEMENTS[movement]
    x, y = tapenv.read_head_posiion
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
end


_get_obs(algoenv::AlgorithmicEnv) = _get_obs(algoenv, algoenv.read_head_position)
function _get_obs(tapeenv::TapeAlgorithmicEnv, pos::Integer)
    pos < 1 && (return tapeenv.base)
    try
        return tapeenv.input_data[pos]
    catch BoundsError
        return tapeenv.base
    end
end

function _get_obs(gridenv::GridAlgorithmicEnv, pos::Array)
    x, y = pos
    any(idx < 1 for idx in pos) && (return gridenv.base)
    try
        return gridenv.input_data[x][y]
    catch BoundsError
        return gridenv.base
    end
end
