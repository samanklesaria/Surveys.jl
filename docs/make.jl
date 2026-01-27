using Documenter
using Surveys

makedocs(
    sitename="Surveys",
    format=Documenter.HTML(sidebar_sitename=false),
    pages=[],
    modules=[Surveys]
)

# deploydocs(repo="github.com/samanklesaria/Surveys.jl.git")
