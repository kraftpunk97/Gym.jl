using Gym
using StatsBase: sample, Weights
using DataStructures: CircularBuffer
using Statistics: mean

env = make("Copy-v0")
reset!(env)

max_episodes = 75000
α = 3f-1
γ = 9f-1
ϵ = 1f-1
ϵ_schedule = 10000
# Copy is considered solved when the average reward for 100 continuous episodes >=25
goal = 25f0

num_dims = [length(space) for space in env.action_space.spaces]
num_actions = prod(num_dims)
num_states = length(env.observation_space)

Q = zeros(Float32, num_states, num_actions)
P = zeros(Float32, num_actions)

function policy(state)
    fill!(P, ϵ/num_actions)
    P[argmax(Q[state[1], :])] += 1 - ϵ
    return sample(1:num_actions, Weights(P))
end

function actioninttotup(int)
  @assert int <= prod(num_dims)
  res = []
  int -= 1
  for dim in reverse(num_dims)
    push!(res, (int%dim)+1)
    int = div(int, dim)
  end
  return reverse(res)
end

total_reward = CircularBuffer{Float32}(100)
for e in 1:max_episodes
    global α, γ, ϵ_schedule, ϵ, goal
    if e % 50 == 0
        println("Episode: $e | Average reward of the last 100 episodes: $(mean(total_reward))")
        if mean(total_reward) > goal
            println("Copy Solved  at episode = $e")
            break
        end
    end
    e % ϵ_schedule == 0 && (ϵ /= 2)
    current_state = reset!(env)
    done = false
    push!(total_reward, 0f0)
    while !done
        a = policy(current_state)
        decoded_a = actioninttotup(a)
        next_state, reward, done, _ = step!(env, decoded_a)
        #println("a = $a\tcurrent_state[1] = $(current_state[1])\tnext_state[1] = $(next_state[1])")
        Q[current_state[1], a] += α * ((reward + γ * maximum(Q[next_state[1], :])) - Q[current_state[1], a])
        current_state = next_state
        total_reward[end] += reward
    end
end
