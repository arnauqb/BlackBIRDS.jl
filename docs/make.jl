using BlackBIRDS
using Documenter

DocMeta.setdocmeta!(BlackBIRDS, :DocTestSetup, :(using BlackBIRDS); recursive=true)

makedocs(;
    modules=[BlackBIRDS],
    authors="Arnau Quera-Bofarull",
    sitename="BlackBIRDS.jl",
    format=Documenter.HTML(;
        canonical="https://arnauqb.github.io/BlackBIRDS.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/arnauqb/BlackBIRDS.jl",
    devbranch="main",
)
