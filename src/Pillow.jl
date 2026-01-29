module Pillow

export TiffStack, frame


using PythonCall: Py, PyArray, pybuiltins, pyconvert, pyimport, pynew, pycopy!
import Base: close, size, getindex, setindex!, IndexStyle
using Base: front, @propagate_inbounds

# Pillow image mode -> (Julia eltype, stack ndims, NumPy dtype attribute symbol)
# Stack ndims here refers to the exposed array dimensionality:
#   ndims==3  => (nx, ny, nf)
#   ndims==4  => (nc, nx, ny, nf)
const MODE_INFO = Dict(
    "I;16" => (UInt16, 3, :uint16),
    "L"    => (UInt8,  3, :uint8),
    "RGB"  => (UInt8,  4, :uint8),
)

const PIM = pynew()  # PIL.Image module
const NP  = pynew()  # numpy module

function __init__()
    pycopy!(PIM ,pyimport("PIL.Image"))
    pycopy!(NP, pyimport("numpy"))
end

"""
    TiffStack(path::AbstractString)
    TiffStack(img::Py)

Read-only cached view of a multi-frame TIFF backed by Pillow + NumPy.

Axis semantics (kept compatible with the original PyCall version):
- Grayscale: `size(t) == (nx, ny, nf)` and indexing `t[x, y, k]`
- RGB:       `size(t) == (nc, nx, ny, nf)` and indexing `t[c, x, y, k]`

Implementation notes:
- Each frame is decoded to a NumPy array and then *transposed as a view* so that the
  cached page matches the axis semantics above:
    - grayscale: NumPy (ny, nx)    -> cached page (nx, ny) via transpose(1,0)
    - RGB:      NumPy (ny,nx,nc)   -> cached page (nc,nx,ny) via transpose(2,1,0)
- Only one page is cached at a time; changing the last index loads a new frame.
"""
mutable struct TiffStack{T, M, N, P<:AbstractArray{T, N}} <: AbstractArray{T, M}
    img::Py
    dtype::Py              # cached NumPy dtype object, e.g. np.uint8
    current_page::P        # cached page in our axis order: (nx,ny) or (nc,nx,ny)
    current_pageno::Int    # 1-based frame index
    dims::NTuple{M, Int}   # (nx,ny,nf) or (nc,nx,ny,nf)
    closed::Bool
end

IndexStyle(::Type{<:TiffStack}) = IndexCartesian()
size(t::TiffStack) = t.dims

function close(t::TiffStack)
    t.closed && return nothing # Exit early if already closed

    try
        if PythonCall.C.Py_IsInitialized() != 0
            t.img.close()
        end
    catch ex
        ex isa UndefVarError || rethrow()
        # Module bindings may be torn down during atexit; nothing to do.
    end

    t.closed = true
    return nothing
end

function TiffStack(f::Function, pathname::AbstractString)
    stack = TiffStack(pathname)
    try
        f(stack)
    finally
        close(stack)
    end
end

# -----------------------
# Internal: page loading
# -----------------------
@inline function _load_page(img::Py, pageno::Int, dtype::Py, ::Val{M}) where {M}
    img.seek(pageno - 1)                   # Pillow uses 0-based frame indexing
    a = NP.asarray(img, dtype=dtype)     # decode -> NumPy array (often allocates once)

    if M == 3
        # (ny,nx) -> (nx,ny)
        return PyArray(a.transpose(1, 0))
    elseif M == 4
        # (ny,nx,nc) -> (nc,nx,ny)
        return PyArray(a.transpose(2, 1, 0))
    else
        throw(ArgumentError("Unsupported stack dimensionality M=$M"))
    end
end

@inline function _load_page!(t::TiffStack{T, M, N, P}, pageno::Int) where {T, M, N, P}
    t.closed && error("TiffStack is already closed and cannot load new frames.")
    t.current_pageno = pageno
    t.current_page = _load_page(t.img, pageno, t.dtype, Val(M))::P
    return t
end

# -----------------------
# Public: frame accessor
# -----------------------
@inline frame(t::TiffStack, k::Integer) = _frame(t, Int(k))

@inline @propagate_inbounds function _frame(t::TiffStack{<:Any, M}, k::Int) where {M}
    # Frame index is always the last axis in this API.
    @boundscheck checkbounds(t, ntuple(_ -> 1, M - 1)..., k)
    if k != t.current_pageno
        _load_page!(t, k)
    end
    return t.current_page
end

# -----------------------
# Constructors
# -----------------------
TiffStack(pathname::AbstractString) = TiffStack(PIM.open(pathname))

function TiffStack(img::Py)
    mode = pyconvert(String, img.mode)
    info = get(MODE_INFO, mode, nothing)
    info === nothing && throw(ArgumentError("Unsupported Pillow mode: $(repr(mode))"))
    elt, nd, dtypesym = info

    dtype = getproperty(NP, dtypesym)

    # PIL.Image.size is (nx, ny)
    nx, ny = pyconvert(NTuple{2, Int}, img.size)
    nf = pyconvert(Int, img.n_frames)
    nf > 0 || throw(ArgumentError("Must have at least one frame"))

    if nd == 3
        dims = (nx, ny, nf)
        page1 = _load_page(img, 1, dtype, Val(3))  # (nx, ny)
        obj =  TiffStack{elt, 3, 2, typeof(page1)}(img, dtype, page1, 1, dims, false)
    elseif nd == 4
        nc = pyconvert(Int, pybuiltins.len(img.getbands()))
        dims = (nc, nx, ny, nf)
        page1 = _load_page(img, 1, dtype, Val(4))  # (nc, nx, ny)
        obj = TiffStack{elt, 4, 3, typeof(page1)}(img, dtype, page1, 1, dims, false)
    else
        throw(ArgumentError("Unsupported nd=$nd for mode=$(repr(mode))"))
    end

    finalizer(close, obj)

    return obj
end

# -----------------------
# Indexing
# -----------------------

# (1) Fast scalar path: all Int indices
@inline @propagate_inbounds function getindex(
    t::TiffStack{<:Any, M}, i::Vararg{Int, M}
) where {M}
    @boundscheck checkbounds(t, i...)
    k = i[M] # Access the last element directly via M
    if k != t.current_pageno
        _load_page!(t, k)
    end
    return t.current_page[front(i)...]
end

# (2) CartesianIndex forwarding
@inline @propagate_inbounds function getindex(
    t::TiffStack{<:Any, M}, I::CartesianIndex{M}
) where {M}
    return t[I.I...]
end

# (B) Single-frame slicing where the last index is an Int frame number:
#     t[:, :, k] or t[:, :, :, k]
@inline @propagate_inbounds function getindex(
    t::TiffStack{<:Any, M},
    inds::Vararg{Any, M},
) where {M}
    @boundscheck checkbounds(t, inds...)

    k = inds[M]
    k isa Int || throw(ArgumentError("Last index must be an Int frame number, got $(typeof(k))"))

    if k != t.current_pageno
        _load_page!(t, k)
    end

    return t.current_page[front(inds)...]
end

setindex!(::TiffStack, ::Any, ::Any...) = throw(ReadOnlyMemoryError())

end # module
