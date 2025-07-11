using ExplainableAI
using Flux, MLDatasets
using Flux: flatten
using PythonCall

# Define model
fluxModel = Chain(
    Conv((3, 3), 1=>8, relu),
    MaxPool((2, 2)),
    Conv((3, 3), 8=>16, relu),
    MaxPool((2, 2)),
    flatten,
    Dense(400, 10),
    softmax
)
weights=Flux.params(fluxModel)

# Load data
# Entire batch for model use
train_x, train_y = MLDatasets.MNIST.traindata()
x_batch = Float32.(train_x[:, :, 1:32]) ./ 255.0
x_batch = reshape(x_batch, 28, 28, 1, 32)

# Optional: true labels for the batch
y_batch = train_y[1:32] .- 1  # Convert from 1-based to 0-based if needed

# For per-sample analysis (explanations, etc.), still use `x`
x = x_batch[:, :, :, 1]  # This is a single 28x28x1 image
x = reshape(x, 28, 28, 1, 1)
# Create analyzers
analyzer1 = SmoothGrad(fluxModel)
analyzer2 = Gradient(fluxModel)
analyzer3 = IntegratedGradients(fluxModel)
analyzer4 = InputTimesGradient(fluxModel)