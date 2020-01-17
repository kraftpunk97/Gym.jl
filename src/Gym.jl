module Gym

using Flux
#using Flux.Tracker


# GymSpaces exports
using GymSpaces
export sample, seed!

using Requires

include("Envs/registry.jl")
export make, register, speclist,          	     # Registry functions
       EnvWrapper, reset!, step!, seed!, state,
       trainable, is_over, render!, testmode!    # Environment interaction functions

end #module
