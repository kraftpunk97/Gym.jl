include("utils.jl")

function Ctx(env::AlgorithmicEnv, mode::Symbol=:human)
    if mode == :human
        HumanCtx()
    elseif mode == :ansi
        ANSICtx()
    elseif mode == :no_render
        NoCtx()
    else
        error("Unrecognized mode in Ctx(): $(mode)")
    end
end

function colorize_bold(string::Union{AbstractString, Char}, color::Symbol)
    colormap = Dict(
    :gray => 30, # highlights to black
    :red => 31,
    :green => 32,
    :yellow => 34,
    :magenta => 35,
    :cyan => 36,
    :white => 37)

    num = colormap[color]
    num += 10 # For highlighting
    return "\x1b[$num;1m" * string * "\x1b[0m"
end

_get_str_target(env::AlgorithmicEnv, pos) =
    pos < 1 || length(env.target) < pos ? " " : env.charmap[env.target[pos]]

function render_observation(env::TapeAlgorithmicEnv)
    x = env.read_head_position
    x_str = "Observation Tape    : "
    for i=-1:length(env.input_data)+2
        x_str *= i == x ? colorize_bold(env.charmap[_get_obs(env, i)[1]], :green) :
                          env.charmap[_get_obs(env, i)[1]]
    end
    x_str *= "\n"
    return x_str
end

function render_observation(env::GridAlgorithmicEnv)
    x = env.read_head_position
    label = "Observation Grid    : "
    x_str = ""
    for j=0:env.rows
        if j != 0
            for _=1:length(label)
                x_str *= " "
            end
        end
        for i=0:length(env.input_data)
            println("Okay for i=$i and j=$j, _get_obs=$(_get_obs(env, [i, j]))")
            x_str *= i == x[1] && j == x[2] ? colorize_bold(env.charmap[_get_obs(env, [i, j])[1]], :green) :
                                              env.charmap[_get_obs(env, [i, j])[1]]
        end
        x_str *= "\n"
    end
    x_str = label * x_str
    return x_str
end

function drawcanvas!(env::AlgorithmicEnv)
    inp = "Total length of input instance: $(length(env.input_data)), step: $(env.time)\n"
    output = ""
    output *= inp
    x = env.read_head_position
    y = env.write_head_position
    action = env.last_action
    !isnothing(action) &&
        ((inp_act, out_act, pred) = action)
    equal_sign = ""
    for i=1:length(inp)+1
        equal_sign *= '='
    end
    output *= equal_sign * '\n'
    y_str =      "Output Tape         : "
    target_str = "Targets             : "
    !isnothing(action) &&
        (pred_str = env.charmap[pred])
    x_str = render_observation(env)
    for i=-1:length(env.target)+2
        target_str *= _get_str_target(env, i)
        if i < y - 1
            y_str *= _get_str_target(env, i)
        elseif i == (y - 1)
            if !isnothing(action) && out_act == 2
                color = pred == env.target[i] ? :green : :red
                y_str *= colorize_bold(pred_str, color)
            else
                y_str *= _get_str_target(env, i)
            end
        end
    end
    output *= x_str * y_str * '\n' * target_str * "\n\n"

    if !isnothing(action)
        output *= "Current reward      :   $(env.last_reward)\n"
        output *= "Cumulative reward   :   $(env.episode_total_reward)\n"
        output *= "Action              :   Tuple(move over input: $(String(env.MOVEMENTS[inp_act])),\n"
        output *= "                              write to the output tape: $(out_act==2)\n"
        output *= "                              prediction: $pred_str)\n"
    else
        output *= "\n\n\n\n\n"
    end
    return output
end
