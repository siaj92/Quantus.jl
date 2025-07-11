module MaxSensitivityJL

export MaxSensitivity, evaluate_batch

using Statistics

"""
    MaxSensitivity(nr_runs::Int, radius::Float32, similarity_func::Function,
                   perturb_func::Function, explain_func::Function; max_func::Function=maximum)

Construct a MaxSensitivity metric instance.

# Arguments
- `nr_runs`: Number of perturbation runs.
- `radius`: Radius or intensity of the perturbation (not used directly here, passed to `perturb_func` if needed).
- `similarity_func`: Function to compare original vs. perturbed attributions (e.g., cosine similarity).
- `perturb_func`: Function that perturbs input data.
- `explain_func`: Function that generates attributions (explanations) from the model.
- `max_func`: Aggregation function over runs (default: `maximum`).

Returns a `MaxSensitivity` struct.
"""
struct MaxSensitivity
    nr_runs::Int
    radius::Float32
    perturb_func::Function
    explain_func::Function
    similarity_func::Function
    max_func::Function
end

"""
    evaluate_batch(metric::MaxSensitivity, model, x_batch, y_batch, a_batch)

Evaluate the MaxSensitivity metric on a batch.

# Arguments
- `metric`: An instance of `MaxSensitivity`.
- `model`: The model to evaluate.
- `x_batch`: Batch of input samples, shaped (H, W, C, B).
- `y_batch`: Corresponding true labels.
- `a_batch`: Attribution maps (explanations) for the true class, shaped (H, W, C, B).

# Returns
- A vector of sensitivity scores for each sample in the batch.
"""
function evaluate_batch(metric::MaxSensitivity, model, x_batch, y_batch, a_batch)
    batch_size = size(x_batch, 4)                       # Number of samples in batch
    n_features = prod(size(x_batch)[1:3])               # Total number of input features per sample

    a_flat = reshape(a_batch, batch_size, n_features)   # Flatten attribution maps for similarity calculation
    similarities = fill(NaN32, batch_size, metric.nr_runs)  # Store similarity scores per run

    for run in 1:metric.nr_runs
        # Create a list of indices to perturb (here: all features)
        indices = [collect(1:n_features) for _ in 1:batch_size]
    
        # Apply perturbation function to inputs
        x_perturbed = metric.perturb_func(x_batch, indices)
    
        # Compute attributions on perturbed inputs
        a_perturbed = reshape(metric.explain_func(model, x_perturbed, y_batch), batch_size, n_features)
    
        # Compare original and perturbed attributions
        for i in 1:batch_size
            similarities[i, run] = metric.similarity_func(
                reshape(a_flat[i, :], 1, :),        # reshape vector to 1-row matrix
                reshape(a_perturbed[i, :], 1, :)
            )
        end
    end
    

    # Aggregate similarity values over runs (e.g., take max per sample)
    return metric.max_func(similarities, dims=2)[:]
end

end # module
