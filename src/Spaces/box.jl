# TODO: seed, copy

"""
A box in R^n, i.e., each coordinate is bounded.

Two kinds of valid input:
    Box(-1.0, 1.0, (3,4)) # low and high are scalars, and shape is provided
    Box([-1.0,-2.0], [2.0,4.0]) # low and high are arrays of the same shape
"""
mutable struct Box <: AbstractSpace
    low::Array
    high::Array
    shape::Tuple
    dtype::DataType
    #seed::Int
end

function Box(low::Number, high::Number, shape::Union{Tuple, Array{Int64, 1}}, dtype::Union{DataType, Nothing}=nothing)
    if isnothing(dtype)
        dtype = high == 255 ? UInt8 : Float32
        @warn "dtype was autodetected as $(dtype). Please provide explicit data type."
    end

    if low > high
        @warn "low  > high. Swapping values to preserve sanity"
        (low, high) = (high, low)  # Preserves sanity if low  high
    end

    if dtype <: Integer
        if !isa(low, Integer) || !isa(high, Integer)
            @warn "dtype is an Integer, but the values are floating points. Using ceiling of lower bound and floor of upper bound"
        end
        low = ceil(dtype, low)
        high = floor(dtype, high)
    end

    Low = dtype(low) .+ zeros(dtype, shape)
    High = dtype(high) .+ zeros(dtype, shape)
    return Box(Low, High, shape, dtype)
end

function Box(low::Array, high::Array, dtype::Union{DataType, Nothing}=nothing)
    @assert size(low) == size(high) "Dimension mismatch between low and high arrays."
    shape = size(low)
    @assert all(low .< high) "elements of low must be lesser than their respective counterparts in high"

    if isnothing(dtype)
        dtype = all(high .== 255) ? UInt8 : Float32
        @warn "dtype was autodetected as $(dtype). Please provide explicit data type."
    end
    if dtype <: Integer
        if !all(isa.(low, Integer)) || !all(isa(high, Integer))
            @warn "dtype is an Integer, but the values are floating points. Using ceiling of lower bound and floor of upper bound"
        end
        low = ceil.(dtype, low)
        high = floor.(dtype, high)
    else
        low = dtype.(low)
        high = dtype.(high)
    end
    return Box(low, high, shape, dtype)
end
#=
function seed!(box_obj::Box, seed::Int)
    box_obj.seed = seed
end
=#
function sample(box_obj::Box)
    box_obj.dtype <: AbstractFloat ?
        rand(box_obj.dtype, box_obj.shape) .* (box_obj.high .- box_obj.low) .+ box_obj.low :
        rand.(UnitRange.(box_obj.low, box_obj.high))
end

function contains(x, box_obj::Box)
    isa(x, Number) && box_obj.shape == (1,) && (x = [x])
    size(x) == box_obj.shape && all(box_obj.low .<= x .<= box_obj.high)
end

checkvalidtypes(box_obj1::Box, box_obj2::Box) =
    box_obj1.dtype == box_obj2.dtype ||                            # If the dtypes of both boxes are not the same...
        box_obj1.dtype <: Integer && box_obj2.dtype <: Integer &&  # then check if they're both integers...
            (box_obj1.dtype <: Unsigned && box_obj2.dtype <: Unsigned) || (box_obj1.dtype <: Signed && box_obj2.dtype <: Signed)  # And then check if they're both signed or both unsigned.

Base.:(==)(box_obj::Box, other::Box) = isapprox(box_obj.low, other.low) && isapprox(box_obj.high, other.high) && checkvalidtypes(box_obj, other)
