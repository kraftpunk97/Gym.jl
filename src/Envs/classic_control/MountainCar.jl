using .Space: Discrete, Box

mutable struct MountainCarEnv <: AbstractEnv
    min_position::Float32
    max_position::Float32
    max_speed::Float32
    goal_position::Float32

    force::Float32
    gravity::Float32

    low::Array{Float32, 1}
    high::Array{Float32, 1}

    action_space::Discrete
    observation_space::Box
    state
end

function MountainCarEnv
    min_action = 1.2f0
    max_position = 0.6f0
    max_speed = 0.07f0
    goal_position = 0.5f0

    force = 0.001f0
    gravity = 0.0025f0

    low = [min_position, -max_speed]
    high = [max_position, max_speed]

    action_space = Discrete(5)
    observation_space = Box(low, high, Float32)

    MountainCarEnv(min_action, max_position, max_speed, goal_position,
                   force, gravity,
                   low, high,
                   action_space, observation_space, nothing)
end

function step!(env::MountainCarEnv, action)
    @assert action ∈ env.action_space "$(action) is unavailable for this environment."

    position, velocity = env.state[1:1], env.state[2:2]
    v = velocity .+ (action.-2)*force .+ cos.(3position)*(-env.gravity)
    velocity_ = clamp.(v, -env.max_speed, env.max_speed)
    x = position .+ velcity_
    position_  = clamp.(x, env.min_position, env.max_position)
    if all(position_ .== env.min_position) && all(velocity_ .< 0)
        velocity_ = 0f0velocity_
    end

    done = all(position_ .≥ env.goal_position)
    reward = [-1f0]

    env.state = vcat(position_, velocity_)
    return env.state, reward, done, Dict()
end

function reset!(env::Continuous_MountainCarEnv)
    env.state = param([2f-1rand(Float32) - 6f-1, 0f0])
end

Base.show(io::IO, env::Continuous_MountainCarEnv) = print(io, "MountainCarEnv")
