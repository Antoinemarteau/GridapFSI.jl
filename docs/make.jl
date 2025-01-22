using Documenter, GridapFSI

makedocs(;
    modules=[GridapFSI],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/oriolcg/GridapFSI.jl/blob/{commit}{path}#L{line}",
    sitename="GridapFSI.jl",
    authors="Oriol Colomes",
    warnonly = [:cross_references,:missing_docs],
#    assets=String[],
)

deploydocs(;
    repo="github.com/oriolcg/GridapFSI.jl",
)
