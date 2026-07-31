# AOCL.jl

## AOCL
AOCL is a set of numerical libraries optimized for AMD processors based on the
AMD “Zen” core architecture and generations. Supported processor families are
AMD EPYC™, AMD Ryzen™, and AMD Ryzen™ Threadripper™ processors. The tuned
implementations of industry-standard math libraries enable rapid development of
scientific and high-performance computing applications.

## Using Julia with AOCL
AOCL.jl is a Julia package that allows users to use the AMD AOCL libraries for
Julia's underlying BLAS and LAPACK, instead of OpenBLAS (default).

This is made possible by libblastrampoline, which enables picking a BLAS and
LAPACK library at runtime.

## Install
Adding the package will replace the system BLAS and LAPACK with AOCL provided
ones at runtime. Note that the AOCL package has to be loaded in every new Julia
process. Upon quitting and restarting, Julia will start with the default
OpenBLAS.

The recommended installation uses the package from Julia's General registry:

```julia
julia> using Pkg
julia> Pkg.add("AOCL")
```

If installing the AMD-hosted version use its repository URL explicitly:

```julia
julia> using Pkg;
julia> Pkg.add(url="https://github.com/amd/AOCL.jl");
```

`AOCL_jll` is available from Julia's General registry and will be installed
automatically as a dependency.

Note: The package is only available on Linux at the moment.

## Checking the Installation
```Julia
julia> using LinearAlgebra

julia> BLAS.get_config()
LinearAlgebra.BLAS.LBTConfig
Libraries:
└ [ILP64] libopenblas64_.so

julia> using AOCL

julia> BLAS.get_config()
LinearAlgebra.BLAS.LBTConfig
Libraries:
├ [ILP64] libaocl64.so
└ [ LP64] libaocl.so
```

## Support
Please contact [toolchainsupport@amd.com](mailto:toolchainsupport@amd.com) for
questions, feature requests, or issues.
