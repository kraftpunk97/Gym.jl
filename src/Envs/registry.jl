include("registration.jl")

# Classic Control
#=============================================================================#
register("CartPole-v0",
         :CartPoleEnv,
         "/classic_control/CartPole.jl",
	     max_episode_steps=200,
		 reward_threshold=195.0)

register("CartPole-v1",
         :CartPoleEnv,
         "/classic_control/CartPole.jl",
	     max_episode_steps=500,
		 reward_threshold=475.0)

register("Pendulum-v0",
         :PendulumEnv,
         "/classic_control/Pendulum.jl",
		 max_episode_steps=200)

register("MountainCarContinuous-v0",
         :Continuous_MountainCarEnv,
         "/classic_control/Continuous-MountainCar.jl",
		 max_episode_steps=999,
		 reward_threshold=90.0)

register("MountainCar-v0",
		 :MountainCarEnv,
		 "/classic_control/MountainCar.jl",
		 max_episode_steps=200,
		 reward_threshold=110.0)

# Algorithmic environments
#=============================================================================#

register("Copy-v0",
		 :CopyEnv,
		 "/algorithmic/Copy.jl",
		 max_episode_steps=200,
		 reward_threshold=25.0)

register("RepeatCopy-v0",
		 :RepeatCopyEnv,
		 "/algorithmic/RepeatCopy.jl",
		 max_episode_steps=200,
		 reward_threshold=75.0)

register("ReversedAddition-v0",
		 :ReversedAdditionEnv,
		 "/algorithmic/ReversedAddition.jl",
		 max_episode_steps=200,
		 reward_threshold=25.0)

register("DuplicatedInput-v0",
		 :DuplicatedInputEnv,
		 "/algorithmic/DuplicatedInput.jl",
		 max_episode_steps=200,
		 reward_threshold=25.0)

register("Reverse-v0",
		 :ReverseEnv,
		 "/algorithmic/Reverse.jl",
		 max_episode_steps=200,
		 reward_threshold=25.0)
