using QuantusJL
using Documenter

DocMeta.setdocmeta!(QuantusJL, :DocTestSetup, :(using QuantusJL); recursive=true)

makedocs(;
    modules=[QuantusJL],
    authors="Group D",
    sitename="QuantusJL.jl",
    format=Documenter.HTML(;
        canonical="https://juliuswa.github.io/QuantusJL.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Getting Started" => "guide.md",
    ],
)

deploydocs(;
    repo="github.com/juliuswa/QuantusJL.jl",
    devbranch="main",
)
