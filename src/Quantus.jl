module Quantus
    using PythonCall
    const quantus = pyimport("quantus")
    export evaluate_faithfulness

    include("metrics/python_metrics.jl")
    include("metrics/RandomLogitJL.jl")
    include("metrics/MaxSensitivityJL.jl")
    include("metrics/SparsenessJL.jl")

    using .PythonMetrics: evaluate_faithfulness
    using .RandomLogitJL
    using .MaxSensitivityJL
    using .SparsenessJL
    using .PythonMetrics: evaluate_faithfulness

    export evaluate_faithfulness, RandomLogit, MaxSensitivity, Sparseness
end