module Gym

using Flux
#using Flux.Tracker


# GymSpaces exports
using GymSpaces
export sample

include("Envs/registry.jl")
export make, register,        	     		# Registry functions
       EnvWrapper, reset!, step!, seed!, state,
       trainable, game_over, render!, testmode!  # Environment interaction functions

end #module
