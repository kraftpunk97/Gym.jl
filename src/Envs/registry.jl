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

register("Acrobot-v0",
		 :AcrobotEnv,
		 "/classic_control/Acrobot.jl",
		 max_episode_steps=500,
		 reward_threshold=100.0)


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


# Atari environments
#=============================================================================#

games_list = ["adventure", "air_raid", "alien", "amidar", "assault", "asterisk", "atlantis",
			  "bank_heist", "battle_zone", "beam_rider", "berzerk", "bowling", "boxing", "breakout", "carnival",
			  "centipede", "chopper_command", "crazy_climber", "defender", "demon_attack", "double_dunk",
			  "elevator_action", "enduro", "fishing_derby", "freeway", "frostbite", "gopher", "gravitar",
			  "hero", "ice_hockey", "jamesbond", "journey_escape", "kangaroo", "krull", "kung_fu_master",
			  "montezuma_revenge", "ms_pacman", "name_this_game", "pheonix", "pitfall", "pong", "pooyan",
			  "private_eye", "qbert", "riverraid", "road_runner", "robotank", "seaquest", "skiing",
			  "solaris", "space_invaders", "star_gunner", "tennis", "time_pilot", "tutankham",
			  "up_n_down", "venture", "video_pinball", "wizard_of_wor", "yars_revenge", "zaxxon"]

for game ∈ games_list
	for obs_type ∈ (:ram, :color, :grey)
		# space_invaders should yield SpaceInvaders-v0 and SpaceInvaders-ram-v0
		name = game |> (g) -> split(g, '_') .|> titlecase |> join
		obs_type == :ram && (name = "$(name)-ram")
		obs_type == :grey && (name = "$(name)-grey")

		# ElevatorAction-ram-v0 seems to yield slightly
		# non-deterministic observations about 10% of the time. We
		# should track this down eventually, but for now we just
		# mark it as nondeterministic.
		nondeterministic = game=="elevator_action" && obs_type==:ram

		register("$(name)-v0",
				 :AtariEnv,
				 "/atari/Atari.jl",
				 max_episode_steps = 10000,
				 nondeterministic = nondeterministic,
				 kwargs = Dict(
				 	:game_path => game,
				 	:obs_type  => obs_type,
				 	:repeat_action_probability => 0.25f0
				 	)
			)

		register("$(name)-v4",
				 :AtariEnv,
			 	 "/atari/Atari.jl",
				 max_episode_steps = 100000,
				 nondeterministic = nondeterministic,
				 kwargs = Dict(
				 	:game_path => game,
				 	:obs_type  => obs_type,
				 	)
			)

		frameskip = game=="space_invaders" ? 3 : 4

		register("$(name)Deterministic-v0",
				 :AtariEnv,
				 "/atari/Atari.jl",
				 max_episode_steps = 100000,
				 nondeterministic = nondeterministic,
				 kwargs = Dict(
				 	:game_path => game,
					:obs_type  => obs_type,
					:frameskip => frameskip
				 )
			)

		register("$(name)Deterministic-v4",
				 :AtariEnv,
				 "/atari/Atari.jl",
				 max_episode_steps = 100000,
				 nondeterministic = nondeterministic,
				 kwargs = Dict(
				 	:game_path => game,
					:obs_type  => obs_type,
					:frameskip => frameskip
				 )
			)

		register("$(name)NoFrameskip-v0",
				 :AtariEnv,
				 "/atari/Atari.jl",
				 max_episode_steps = frameskip * 100000,
				 nondeterministic = nondeterministic,
				 kwargs = Dict(
				 	:game_path => game,
					:obs_type  => obs_type,
					:frameskip => 1,
					:repeat_action_probability => 25f-2
				 )
			)


		# No frameskip. (Atari has no entropy source, so these are
	    # deterministic environments.)
		register("$(name)NoFrameskip-v4",
				 :AtariEnv,
				 "/atari/Atari.jl",
				 max_episode_steps = frameskip * 100000,
				 nondeterministic = nondeterministic,
				 kwargs = Dict(
				 	:game_path => game,
					:obs_type  => obs_type,
					:frameskip => 1
				 )
			)
	end
end
