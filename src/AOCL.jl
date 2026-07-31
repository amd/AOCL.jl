# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

module AOCL

using AOCL_jll
using LinearAlgebra

function __init__()
    lbt_forward_to_aocl()
end

function lbt_forward_to_aocl()
    if !AOCL_jll.is_available()
        isinteractive() && @warn "AOCL is not available/installed."
        return
    end

    if Base.USE_BLAS64
        # Load ILP64 forwards used by Julia.
        BLAS.lbt_forward(AOCL_jll.libaocl64; clear=true)
        # Also expose LP64 forwards.
        BLAS.lbt_forward(AOCL_jll.libaocl)
    else
        BLAS.lbt_forward(AOCL_jll.libaocl; clear=true)
    end
end

end # module
