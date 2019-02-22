# Gym.jl
Gym environments in Julia

# Installation
```julia
julia> ] add https://github.com/tejank10/Gym.jl
```
To use Gym.jl on GPU, run
```julia
julia> ] add https://github.com/tejank10/Gym.jl#GPU
```

## Usage

```julia
env = CartPoleEnv()
ctx = Ctx(env)

display(ctx.s)

# using Blink # when not on Juno
# body!(Blink.Window(), ctx.s)

actions = rand(1:2, 1000)
i = 1
done = false
reset!(env)
while i <= length(actions) && !done
    global i, done
    a, b, done, d = step!(env, actions[i])
    render(env, ctx)
    i += 1
    # sleep(0.4) # to see an animation
end
```
