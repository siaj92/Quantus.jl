module MaxSensitivityJL2

export MaxSensitivity, evaluate_batch

using Statistics, Random

# Struct definition
struct MaxSensitivity
    nr_runs::Int
    radius::Float32
    perturb_func::Function
    explain_func::Function
    changed_prediction_indices_func::Function
    max_func::Function
end

# Default constructor
function MaxSensitivity(; nr_runs=10, radius=0.05f0, perturb_func=nothing, explain_func=nothing, changed_prediction_indices_func=nothing, max_func=maximum)
    return MaxSensitivity(nr_runs, radius, perturb_func, explain_func, changed_prediction_indices_func, max_func)
end

# Evaluation function
function evaluate_batch(metric::MaxSensitivity, model, x_batch, y_batch, a_batch)
    batch_size = size(x_batch, 1)
    n_features = prod(size(x_batch)[2:end])

    a_batch = reshape(a_batch, batch_size, n_features)
    similarities = fill(NaN32, batch_size, metric.nr_runs)

    for run in 1:metric.nr_runs
        indices = [collect(1:n_features) for _ in 1:batch_size]
        x_perturbed = metric.perturb_func(x_batch, indices)

        changed_prediction_indices = metric.changed_prediction_indices_func(model, x_batch, x_perturbed)

        a_perturbed = reshape(metric.explain_func(model, x_perturbed, y_batch), batch_size, n_features)

        for i in 1:batch_size
            similarities[i, run] = metric.similarity_func(a_batch[i, :], a_perturbed[i, :])
        end

        for i in changed_prediction_indices
            similarities[i, run] = NaN32
        end
    end

    return metric.max_func(similarities, dims=2)[:]
end

end # module
