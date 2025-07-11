include("MNIST_ExAI.jl")
include("metrics/RandomLogitJL.jl")
include("metrics/MaxSensitivityJL.jl")
include("metrics/SparsenessJL.jl")

using .RandomLogitJL
using .MaxSensitivityJL
#using .FaithfulnessCorrelationJL
using Distances
using LinearAlgebra

# Cosine_batch an example of a similarity_func
function cosine_batch(a::AbstractMatrix, b::AbstractMatrix)
    dists = pairwise(CosineDist(), eachrow(a), eachrow(b))
    return 1 .- diag(dists)
end

# perturb_noise an example of a perturb_func
function perturb_noise(arr::AbstractArray, indices::Vector{Vector{Int}})
    arr_copy = copy(arr)  # Make a copy so original isn't changed
    batch_size = size(arr, 1)
    feature_dim = prod(size(arr)[2:end])  # Flattened feature space
    reshaped_arr = reshape(arr_copy, batch_size, feature_dim)

    for i in 1:batch_size
        n_masked = length(indices[i])
        reshaped_arr[i, indices[i]] .= randn(Float32, n_masked)  # Gaussian noise
    end

    return reshape(reshaped_arr, size(arr))
end


# Batch-level wrapper for ExplainableAI — required by `evaluate_batch`
function explain_batch(model, x_batch, y_target_batch, analyzer)
    batch_size = size(x_batch, 4)
    out = Vector{Array{Float32, 3}}(undef, batch_size)

    for i in 1:batch_size
        x_sample = reshape(x_batch[:, :, :, i], size(x_batch, 1), size(x_batch, 2), size(x_batch, 3), 1)

        # Get the Explanation object
        explanation = analyze(x_sample, analyzer1; target=y_target_batch[i])

        # Extract the explanation values
        # explanation.val is expected to be Array{Float32, 4} => reshape to 3D
        explanation_data = dropdims(explanation.val, dims=4)  # drops singleton dimension to get Array{Float32, 3}

        out[i] = explanation_data
    end

    return cat(out...; dims=4)
end

# Prepare a_batch (original explanations for true classes)
batch_size = size(x_batch, 4)
a_batch = Vector{Array{Float32, 3}}(undef, batch_size)

for i in 1:batch_size
    x_sample = reshape(x_batch[:, :, :, i], size(x_batch, 1), size(x_batch, 2), size(x_batch, 3), 1)

    explanation = analyze(x_sample, analyzer1; target=y_batch[i])

    # Extract and reshape explanation values
    explanation_data = dropdims(explanation.val, dims=4)

    a_batch[i] = explanation_data
end

a_batch = cat(a_batch...; dims=4)

# Instantiate metric
metric = RandomLogitJL.RandomLogit(10, 45, cosine_batch)
#metric1= FaithfulnessCorrelationJL.FaithfulnessCorrelation(100, 224, cosine_batch, perturb_noise)
metric2 = SparsenessJL.Sparseness()
#metric3=MaxSensitivityJL.MaxSensitivity(10,0.05f0, perturb_noise, changed_prediction_indices, cosine_batch)

# Evaluate
scores = RandomLogitJL.evaluate_batch(metric, fluxModel, x_batch, y_batch, a_batch; explain_batch=explain_batch)
#scores1 = FaithfulnessCorrelationJL.evaluate_batch(metric1, fluxModel, x_batch, y_batch, a_batch)
scores2 = SparsenessJL.evaluate_batch(metric2, a_batch)
#println("RandomLogit Scores: ", scores)
println("RandomLogit Test for this MNIST image: ", scores)
#println("FaithfulnessCorrelation for this MNIST image: ", scores1)
println("Sparseness for this MNIST image: ", scores2)