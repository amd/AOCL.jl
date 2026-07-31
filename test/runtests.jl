# ------------------------------------------------------------------------------
# This file includes tests adapted from the following open-source projects:
#
#   1. Project A - https://github.com/JuliaLinearAlgebra/BLISBLAS.jl
#   2. Project B - https://github.com/JuliaLinearAlgebra/MKL.jl
#
# Both are licensed under the MIT License. Their license texts are reproduced
# below in accordance with the terms of the license.
#
# Modifications are licensed under the MIT License.
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Refer to the top-level LICENSE file in this repository for details.
# ------------------------------------------------------------------------------
# License (Project A - https://github.com/JuliaLinearAlgebra/BLISBLAS.jl)
#
# MIT License
#
# Copyright (c) 2022 Carsten Bauer <carsten.bauer@uni-paderborn.de> and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ------------------------------------------------------------------------------
# License (Project B: https://github.com/JuliaLinearAlgebra/MKL.jl)
#
# MKL.jl is licensed under the MIT license.
#
# Copyright (c) 2018-22 Julia Computing, Inc. and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ------------------------------------------------------------------------------

using Test
using LinearAlgebra
using Random
using Libdl

# Set up a debugging fallback function that prints out a stacktrace if the LinearAlgebra
# tests end up calling a function that we don't have forwarded.
function debug_missing_function()
    println("Missing BLAS/LAPACK function!")
    display(stacktrace())
end
LinearAlgebra.BLAS.lbt_set_default_func(@cfunction(debug_missing_function, Cvoid, ()))

function blas(interface=Base.USE_BLAS64 ? :ilp64 : :lp64)
    libs = filter(lib -> lib.interface == interface, BLAS.get_config().loaded_libs)
    isempty(libs) && return :unknown
    lib = lowercase(basename(first(libs).libname))
    if contains(lib, "openblas")
        return :openblas
    elseif contains(lib, "libaocl64")
        return :aocl_ilp64
    elseif lib == "libaocl.so"
        return :aocl_lp64
    else
        return :unknown
    end
end

@testset "AOCL.jl" begin
    @testset "Sanity Tests" begin
        @test blas() == :openblas
        using AOCL
        @test blas(:lp64) == :aocl_lp64
        Base.USE_BLAS64 && @test blas(:ilp64) == :aocl_ilp64
        @test LinearAlgebra.peakflops() > 0
    end

    @testset "AOCL-BLAS threads" begin
        @test isnothing(BLAS.set_num_threads(1))
        @test BLAS.get_num_threads() == 1
        @test isnothing(BLAS.set_num_threads(2))
        @test BLAS.get_num_threads() == 2
        @test isnothing(BLAS.set_num_threads(3))
        @test BLAS.get_num_threads() == 3
    end
end

@testset "CBLAS dot test" begin
    a = ComplexF64[
        1 + 1im,
        2 - 2im,
        3 + 3im
    ]
    @test LinearAlgebra.BLAS.dotc(a, a) ≈ ComplexF64(28)
    @test LinearAlgebra.BLAS.dotu(a, a) ≈ ComplexF64(12im)

    a = Float32[1, 2, 3]
    @test LinearAlgebra.BLAS.dot(a, a) ≈ 14f0
end

@testset "Threading nondeterminism test" begin
    A = [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -5 -5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0; 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -5 -5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0; 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -5 -5 0 0 0 0 0 0 0 0 0 0 0 0 0 0; 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -5 -5 0 0 0 0 0 0 0 0 0 0 0 0 0; 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -5 -5 0 0 0 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -5 0 0 0 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 -5 0 0 0 0 0 -5 -5 0 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 -5 0 0 0 0 0 -5 -5 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 -5 0 0 0 0 0 -5 -5 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 -5 0 0 0 0 0 -5 -5 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 -5 0 0 0 0 0 -5 -5 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 -5 0 0 0 0 0 -5 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 -5 0 0 0 0 0 -5 -5 0 0 0 0; 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 -5 0 0 0 0 0 -5 -5 0 0 0; 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 -5 0 0 0 0 0 -5 -5 0 0; 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 -5 0 0 0 0 0 -5 -5 0; 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 -5 0 0 0 0 0 -5 -5; 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 -5 0 0 0 0 0 -5; -5 0 0 0 0 0 -5 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0; -5 -5 0 0 0 0 0 -5 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0; 0 -5 -5 0 0 0 0 0 -5 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0; 0 0 -5 -5 0 0 0 0 0 -5 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0; 0 0 0 -5 -5 0 0 0 0 0 -5 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0; 0 0 0 0 -5 -5 0 0 0 0 0 -5 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 -5 0 0 0 0 0 -5 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 -5 -5 0 0 0 0 0 -5 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 -5 -5 0 0 0 0 0 -5 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 -5 -5 0 0 0 0 0 -5 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 0 -5 -5 0 0 0 0 0 -5 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 0 0 -5 -5 0 0 0 0 0 -5 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 0 0 0 0 -5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0; 0 0 0 0 0 0 0 0 0 0 0 0 -5 -5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0; 0 0 0 0 0 0 0 0 0 0 0 0 0 -5 -5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0; 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -5 -5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0; 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -5 -5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0; 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -5 -5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1];

    B = [1 0 0 0 0; 0 1 0 0 0; 0 0 1 0 0; 0 0 0 1 0; 0 0 0 0 1; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0];

    for i=1:10
        @test sum(abs, A \ B) ≈ 24.772054506344045
    end
end

# Run the BLAS and LAPACK tests of the LinearAlgebra stdlib against AOCL, using
# Julia's own test driver to get multithreaded, distributed execution and to
# restart workers that trip over the testing RSS limit. The workers only pick up
# AOCL if the test sources load it, so prepend the import to `testdefs.jl`, which
# every worker includes.
julia_test_dir = joinpath(Sys.BINDIR, Base.DATAROOTDIR, "julia", "test")
if !isdir(julia_test_dir)
    @warn "Julia's test sources are not installed, skipping the stdlib tests." julia_test_dir
else
    mktempdir() do dir
        cp(julia_test_dir, dir; force=true)

        testdefs_path = joinpath(dir, "testdefs.jl")
        chmod(testdefs_path, 0o644)
        testdefs_content = read(testdefs_path, String)
        open(testdefs_path, write=true) do io
            println(io, "using AOCL")
            println(io, testdefs_content)
        end

        # Run against a copy of the LinearAlgebra tests, so that the known
        # failure below can be marked without touching the Julia installation.
        # `test_path` in `choosetests.jl` resolves every name that is not a
        # stdlib relative to the directory holding the test driver.
        linalg_dir = joinpath(dir, "aocl_linalg")
        cp(joinpath(dirname(dirname(Base.find_package("LinearAlgebra"))), "test"), linalg_dir)

        # AOCL's `dlacpy` leaves the destination untouched when the destination
        # has fewer rows than the source. Reference LAPACK does no argument
        # checking there and copies rows `1:min(j, m)` of each column `j`, so the
        # call is well defined for `uplo = 'U'`. The single and complex variants
        # are unaffected. Julia tests this since 1.12.
        lapack_path = joinpath(linalg_dir, "lapack.jl")
        lapack_content = read(lapack_path, String)
        chmod(lapack_path, 0o644)
        write(lapack_path, replace(lapack_content,
            "@test B == view(triu(A), 1:n, 1:n)" =>
                "@test B == view(triu(A), 1:n, 1:n) broken = (elty === Float64)"))

        # Every worker loads its own copy of AOCL, and BLIS defaults to one
        # thread per core, so leaving threading alone oversubscribes the machine
        # badly for tests that gain nothing from it.
        cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) $(dir)/runtests.jl aocl_linalg/blas aocl_linalg/lapack`
        run(addenv(cmd, "BLIS_NUM_THREADS" => "1"))
    end
end
