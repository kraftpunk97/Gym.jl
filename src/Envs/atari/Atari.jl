using ArcadeLearningEnvironment

setLoggerMode!(:error)  # Only log errors

include("atari_roms.jl")

struct AtariEnv <: AbstractEnv
    game_path::String
    ale::Ptr{Nothing}
    action_set::Array{UInt, 1}
    action_space::Discrete
    observation_space::Box
    frameskip
end

function AtariEnv(game_path::String, obs_type::Symbol=:ram, frameskip=(2, 5), repeat_action_probability::AbstractFloat=0f0, full_action_space::Bool=false)
    @assert isfile(game_path) "$game_path is not the path to a valid file "
    @assert obs_type ∈ (:ram, :image) "Invalid observation type. Choose either :ram or :image"
    @assert isa(repeat_action_probability, AbstractFloat) || isa(repeat_action_probability, Integer) "Invalid repeat_action_probability"

    ale = ALE_new()
    loadROM(ale, game_path)
    action_set = full_action_space ? getLegalActionSet(ale) : getMinimalActionSet(ale)
    action_space = full_action_space ? getLegalActionSize(ale) : getMinimalActionSize(ale)

    screen_width, screen_height = getScreenWidth(ale), getScreenHeight(ale)

    observation_space = obs_type==:ram ? Box(0, 255, (128,), UInt8) : Box(0, 255, (screen_height, screen_width, 3), UInt8)

    AtariEnv(game_path, ale, action_set, action_space, observation_space, frameskip)
end
