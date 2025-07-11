using Quantus
using Documenter

DocMeta.setdocmeta!(Quantus, :DocTestSetup, :(using Quantus); recursive=true)

makedocs(;
    modules=[Quantus],
    authors="Karim Shawky <k.shawky@campus.tu-berlin.de>",
    sitename="Quantus.jl",
    format=Documenter.HTML(;
        canonical="https://KarimHShawky.github.io/Quantus.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Getting Started" => "guide.md",
    ],
)

deploydocs(;
    repo="github.com/KarimHShawky/Quantus.jl",
    devbranch="main",
)
