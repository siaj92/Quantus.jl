using Flux, MLDatasets
include("src/MaxSensitivityJL.jl")
using .MaxSensitivityJL
using Flux: flatten

model = Chain(
    Conv((3, 3), 1=>8, relu),
    MaxPool((2, 2)),
    Conv((3, 3), 8=>16, relu),
    MaxPool((2, 2)),
    flatten,
    Dense(400, 10),  
    softmax
)


# Load sample input
# Load data
train_x, _ = MLDatasets.MNIST.traindata()
x = Float32.(train_x[:, :, 1]) ./ 255.0
x = reshape(x, 28, 28, 1, 1)  # Add batch dimension

# Evaluate
metric = MaxSensitivity(nsamples=10, radius=0.02f0, normtype=2)
sensitivity = evaluate(metric, model, gradient_explainer, x)

println("MaxSensitivity for this MNIST image: ", sensitivity)