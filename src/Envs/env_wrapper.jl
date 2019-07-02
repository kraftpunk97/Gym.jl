import Flux.testmode!

abstract type AbstractEnv end
abstract type AbstractCtx end

IntOrNothing  = Union{Integer,  Nothing}
RealOrNothing = Union{Real, Nothing}

mutable struct EnvWrapper
    done::Bool
    total_reward::RealOrNothing
    steps_since_reset::Int
	total_steps::Int
    train::Bool
	reward_threshold::RealOrNothing
	max_episode_steps::IntOrNothing
	action_space::GymSpaces.AbstractSpace
	observation_space::GymSpaces.AbstractSpace
    _env::AbstractEnv
    _ctx::AbstractCtx
	reset_called_first::Bool
end

EnvWrapper(env::AbstractEnv, ctx::AbstractCtx, train::Bool=true;
		   reward_threshold=nothing, max_episode_steps=nothing) =
EnvWrapper(false, 0, 0, 0, train, reward_threshold, max_episode_steps, env.action_space,
 			env.observation_space, env, ctx, false)

function step!(env::EnvWrapper, a)
	@assert env.reset_called_first "Cannot call step!(::EnvWrapper, ::Any) before calling reset!(::EnvWrapper)"
    s′, r, done, dict = step!(env._env, a)
    env.total_reward = env.total_reward .+ r
    env.steps_since_reset += 1
	env.total_steps += 1
    env.done = done
	if !isnothing(env.max_episode_steps)
		env.done |= env.steps_since_reset ≥ env.max_episode_steps
	end
    return s′, r, env.done, dict
end

function reset!(env::EnvWrapper)
	env.reset_called_first = true
    env.done = false
    env.total_reward = 0
    env.steps_since_reset= 0
    reset!(env._env)
end

render!(env::EnvWrapper, ctx::AbstractCtx) = render!(env._env, ctx)
render!(env::EnvWrapper) = render!(env, env._ctx)

Ctx(env::EnvWrapper) = Ctx(env._env, env.render_mode)

"""
Returns the observational state of the environment. The original state can
be accessed by `\`env._env.state\``.
"""
function state(env::EnvWrapper)
	try
		return _get_obs(env._env)
	catch y
		if isa(y, UndefVarError) || isa(y, MethodError)
			return env._env.state
		end
	end
end


function testmode!(env::EnvWrapper, val::Bool=true)
    env.train = !val
end

trainable(env::EnvWrapper) = env.train
game_over(env::EnvWrapper) = env.done
