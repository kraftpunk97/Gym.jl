# Gym.jl
Gym environments in Julia

**`Gym.jl` is a work in progress and in active development. Expect breaking changes for some time.**

**`Gym.jl` requires Julia v1.1**

# Installation
First, we need to install a dependency `GymSpaces.jl`, which is currently unregistered (hopefully not for long).

```julia
julia> ] add https://github.com/kraftpunk97/GymSpaces.jl
```
Next we proceed to install `Gym.jl`
```julia
julia> ] add https://github.com/kraftpunk97/Gym.jl#toytext
```

## Usage

```julia
env = make("CartPole-v0", :human_pane)

actions = [sample(env._env.action_space) for i=1:1000]
i = 1
done = false
reset!(env)
while i <= length(actions) && !done
    global i, done
    a, b, done, d = step!(env, actions[i])
    render!(env)
    i += 1
end
```
## Currently available environments
* Continous Control Problems
* Algorithmic Environments
* Atari Environments
* Toy Text Problems
