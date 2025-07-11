module Quantus
    using PythonCall
    const quantus = pyimport("quantus")
    export evaluate_faithfulness

    include("metrics/python_metrics.jl")
    using .PythonMetrics: evaluate_faithfulness
end