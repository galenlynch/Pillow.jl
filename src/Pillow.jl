module Pillow

using PyCall

import Base: size, getindex, setindex!, IndexStyle, @propagate_inbounds

export TiffStack

const MODE_BITSTYPE_LOOKUP = Dict(
    "I;16" => (UInt16, 3),
    "L" => (UInt8, 3),
    "RGB" => (UInt8, 4)
)

const PIM = PyNULL() # PIL.Image
const NP = PyNULL() # numpy

function __init__()
    copy!(PIM, pyimport("PIL.Image"))
    copy!(NP, pyimport("numpy"))
end

mutable struct TiffStack{T, M, N} <: AbstractArray{T, M}
    img::PyObject
    current_page::PyArray{T, N}
    current_pageno::Int
    dims::NTuple{M, Int}

    function TiffStack{T, M, N}(img::PyObject, dims::NTuple{M, Int}) where {T, M, N}
        N == M - 1 || throw(ArgumentError("Invalid dimensions"))
        all(dims .>= 0) || throw(ArgumentError("Invalid size"))
        dims[end] > 0 || throw(ArgumentError("Must have at least one frame"))
        current_page = load_frame(img, 1)::PyArray{T, N}
        new(img, current_page, 1, dims)
    end
end

function TiffStack{T, 3}(img::PyObject, nx, ny, nf) where T
    TiffStack{T, 3, 2}(img, (nx, ny, nf))
end

function TiffStack{T, 4}(img::PyObject, nx, ny, nf) where T
    nc = length(img.getbands())::Int
    TiffStack{T, 4, 3}(img, (nc, nx, ny, nf))
end

function TiffStack{T, N}(img::PyObject) where {T, N}
    nx, ny = img.size::NTuple{2, Int}
    nf = img.n_frames::Int
    TiffStack{T, N}(img, nx, ny, nf)
end

TiffStack{T, N}(pathname::AbstractString) where {T, N} =
    TiffStack{T, N}(PIM.open(pathname)::PyObject)

function TiffStack(img::PyObject)
    elt, nd = MODE_BITSTYPE_LOOKUP[img.mode::String]
    TiffStack{elt, nd}(img)
end

TiffStack(pathname::AbstractString) = TiffStack(PIM.open(pathname)::PyObject)

function load_frame(o::PyObject, pageno::Integer)
    o.seek(pageno - 1)
    pycall(NP.array, PyArray, o)
end

function load_frame!(t::TiffStack{T, 3}, pageno::Integer) where T
    t.current_pageno = pageno
    t.current_page = load_frame(t.img, pageno)::PyArray{T, 2}
end

IndexStyle(::Type{<:TiffStack}) = IndexCartesian()

size(t::TiffStack) = t.dims

@inline @propagate_inbounds function getindex(
    t::TiffStack{<:Any, 3, 2}, i::Vararg{<:Integer, 3}
)
    @boundscheck checkbounds(t, i...)
    if t.current_pageno != i[3]
        load_frame!(t, i[3])
    end
    t.current_page[i[2], i[1]]
end

setindex!(::TiffStack, ::Any, ::Any...) = throw(ReadOnlyMemoryError())

end # module
