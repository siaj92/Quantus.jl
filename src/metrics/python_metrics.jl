module PythonMetrics
export evaluate_faithfulness

"""
    evaluate_faithfulness(; kwargs...) -> Vector{Float64}

Julia wrapper for `quantus.FaithfulnessCorrelation`. Returns one score per input sample.

### Required keyword arguments:
- model: PyObject (e.g., a PyTorch model)
- x_batch: input batch (NumPy-compatible)
- y_batch: ground-truth labels
- a_batch: attribution maps
- device: "cpu" or "cuda"

### All other keyword arguments are passed directly to Quantus.
"""
function evaluate_faithfulness(; model, x_batch, y_batch, a_batch, device=nothing, kwargs...)
    metric_fn = quantus.FaithfulnessCorrelation(; kwargs...)
    return metric_fn(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        device=device
    )
end

end # module
