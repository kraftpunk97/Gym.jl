using .Space: Box, Discrete
using ArcadeLearningEnvironment

setLoggerMode!(:error)  # Only log errors

struct AtariEnv <: AbstractEnv
    game_path::String
    ale::Ptr{Nothing}
    action_set::Array{UInt, 1}
    action_space::Discrete
    observation_space::Box
    frameskip::Union{Int, UnitRange{Int}}
    obs_type::Symbol
end

include("vis/utils.jl")

function AtariEnv( ; game_path::String, obs_type::Symbol=:ram, frameskip::Union{Int, UnitRange{Int}}=2:5,
    repeat_action_probability::AbstractFloat=0f0, full_action_space::Bool=false)

    @assert obs_type ∈ (:ram, :image) "Invalid observation type. Choose either :ram or :image"
    @assert isa(repeat_action_probability, AbstractFloat) || isa(repeat_action_probability, Integer) "Invalid repeat_action_probability"

    ale = ALE_new()
    loadROM(ale, game_path)  # This is will also check if the loaded file is valid or not.
    action_set = full_action_space ? getLegalActionSet(ale) : getMinimalActionSet(ale)
    action_set .+= 1  # Converting to one indexed array
    action_space = full_action_space ? Discrete(getLegalActionSize(ale)) : Discrete(getMinimalActionSize(ale))

    setFloat(ale, "repeat_action_probability", repeat_action_probability)

    screen_width, screen_height = getScreenWidth(ale), getScreenHeight(ale)

    observation_space = obs_type==:ram ? Box(0, 255, (128,), UInt8) : Box(0, 255, (screen_height, screen_width, 3), UInt8)

    AtariEnv(game_path, ale, action_set, action_space, observation_space, frameskip, obs_type)
end


function step!(env::AtariEnv, action)
    @assert action ∈ env.action_space "Action $action is invalid."
    reward = 0f0

    num_steps = isa(env.frameskip, UnitRange) ? rand(env.frameskip) : env.frameskip

    for _ in 1:num_steps
        reward += act(env.ale, action)
    end
    ob = _get_obs(env)
    return ob, reward, ArcadeLearningEnvironment.game_over(env.ale), Dict(:ale_lives => lives(env.ale))
end

_get_obs(env::AtariEnv) = env.obs_type == :ram ? getRAM(env.ale) : getScreenRGB(env.ale)

function reset!(env::AtariEnv)
    reset_game(env.ale)
    _get_obs(env)
end


function clone_state(env::AtariEnv)
    state_ref = cloneState(env.ale)
    state = encodeState(state_ref)
    deleteState(state_ref)
    return state
end

function restore_state(env::AtariEnv, state::Array{Int8, 1})
    state_ref = decodeState(state)
    restoreState(env.ale, state_ref)
    deleteState(state_ref)
end


ACTION_MEANING = [
    (:NOOP,),
    (:FIRE,),
    (:UP,),
    (:RIGHT,),
    (:LEFT,),
    (:DOWN,),
    (:UP, :RIGHT),
    (:UP, :LEFT),
    (:DOWN, :RIGHT),
    (:DOWN, :LEFT),
    (:UP, :FIRE),
    (:RIGHT, :FIRE),
    (:LEFT, :FIRE),
    (:DOWN, :FIRE),
    (:UP, :RIGHT, :FIRE),
    (:UP, :LEFT, :FIRE),
    (:DOWN, :RIGHT, :FIRE),
    (:DOWN, :LEFT, :FIRE)
]

get_action_meanings(env::AtariEnv) = [ACTION_MEANING[i] for i in env.action_set]


function get_keys_to_action(env::AtariEnv)
    KEYWORD_TO_KEY = Dict(
        :UP    => Int('w'),
        :DOWN  => Int('s'),
        :LEFT  => Int('a'),
        :RIGHT => Int('d'),
        :FIRE  => Int(' ')
    )

    keystoaction = Dict()

    for (action_id, action_meaning) ∈ enumerate(get_action_meanings(env))
        keys_ = [key for (keyword, key) ∈ collect(zip(keys(KEYWORD_TO_KEY), values(KEYWORD_TO_KEY)))
                    if keyword ∈ action_meaning] |> sort |> Tuple
        keystoaction[keys_] = action_id
    end
    keystoaction
end

Base.show(io::IO, env::AtariEnv) = print(io, "AtariEnv($(env.game_path), $(env.obs_type))")
export get_keys_to_action, restore_state, clone_state
