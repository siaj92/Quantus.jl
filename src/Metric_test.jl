include("MNIST_ExAI.jl")
include("RandomLogitJL.jl")

using .RandomLogitJL
using Distances

# Define cosine similarity as 1 - cosine distance between flattened rows
function cosine_batch(a::AbstractMatrix, b::AbstractMatrix)
    dists = pairwise(CosineDist(), eachrow(a), eachrow(b))
    return 1 .- diag(dists)
end

# Batch-level wrapper for ExplainableAI — required by `evaluate_batch`
function explain_batch(model, x_batch, y_target_batch)
    batch_size = size(x_batch, 4)
    out = Vector{Array{Float32, 3}}(undef, batch_size)

    for i in 1:batch_size
        x_sample = reshape(x_batch[:, :, :, i], size(x_batch, 1), size(x_batch, 2), size(x_batch, 3), 1)
        out[i] = analyze(x_sample, analyzer1; target=y_target_batch[i])
    end

    return cat(out...; dims=4)
end

# Prepare a_batch (original explanations for true classes)
batch_size = size(x_batch, 4)
a_batch = Vector{Array{Float32, 3}}(undef, batch_size)

for i in 1:batch_size
    x_sample = reshape(x_batch[:, :, :, i], size(x_batch, 1), size(x_batch, 2), size(x_batch, 3), 1)
    a_batch[i] = analyze(x_sample, analyzer1; target=y_batch[i])
end

a_batch = cat(a_batch...; dims=4)

# Instantiate metric
#metric = RandomLogit(n_classes=10, seed=45, similarity_func=cosine_batch)
metric= MaxSensitivity(; nsamples=10, radius=0.05f0, normtype=2)

# Evaluate
scores = evaluate_batch(metric, model, x_batch, y_batch, a_batch; explain_batch=explain_batch)

#println("RandomLogit Scores: ", scores)
println("MaxSensitivity for this MNIST image: ", scores)