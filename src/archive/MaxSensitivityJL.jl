module MaxSensitivityJL

export MaxSensitivity, evaluate, gradient_explainer

using Statistics, Random, Zygote, Flux
using LinearAlgebra: norm

# =============================================================================
# MaxSensitivity Metric
# -----------------------------------------------------------------------------
# This metric evaluates the *robustness* of explanation methods for ML models.
# So, how stable an explanation is. MaxSensitivity measures how much an explanation (e.g., gradient or attribution) changes
# when small random perturbations are added to the input.
#
#
#   * A stable explanation should not change drastically for small noise.
#   * We add noise to the input, recompute the explanation, and calculate
#     the distance (norm) between the original and perturbed explanations.
#   * The final score is the maximum of these distances across multiple samples.
#
# =============================================================================

# Struct definition for the metric configuration
struct MaxSensitivity
    nsamples::Int        # Number of random perturbation samples
    radius::Float32      # Magnitude (standard deviation) of perturbations
    normtype::Int        # Norm type used to compare explanations (e.g., 1, 2, ∞)
end

# Outer constructor with default values
MaxSensitivity(; nsamples=10, radius=0.05f0, normtype=2) = 
    MaxSensitivity(nsamples, radius, normtype)

# =============================================================================
# Evaluate Function
# -----------------------------------------------------------------------------
# Computes the maximum difference between original and perturbed explanations.
# Inputs:
#   - metric:      MaxSensitivity object with parameters
#   - model:       The neural network or classifier
#   - explainer:   A function that computes explanation/saliency for a given input
#   - x:           Input sample (e.g., image or vector)
# Output:
#   - A single Float32 value representing the max deviation in explanations
# =============================================================================
function evaluate(metric::MaxSensitivity, model, explainer, x)
    original_expl = explainer(model, x)     # Explanation for unperturbed input
    diffs = Float32[]                       # To store norm differences

    for _ in 1:metric.nsamples
        δ = metric.radius * randn(Float32, size(x)...)    # Random perturbation
        x_perturbed = clamp.(x .+ δ, 0f0, 1f0)            # Add noise and clamp to valid range
        perturbed_expl = explainer(model, x_perturbed)    # Explanation for perturbed input
        diff = norm(perturbed_expl .- original_expl, metric.normtype)  # Difference in explanations
        push!(diffs, diff)                                # Store the result
    end

    return maximum(diffs)  # Return the worst-case difference
end

# =============================================================================
# Gradient-based Explainer
# -----------------------------------------------------------------------------
# Computes the gradient of the model's output (predicted class) w.r.t. the input.
# Saliency method in explainable AI.
# =============================================================================
function gradient_explainer(f, x)
    ŷ, back = Zygote.pullback(f, x)                       # Forward + backward pass
    class_index = only(Flux.onecold(ŷ))                   # Get predicted class index
    grad = first(back(Flux.onehot(class_index, 1:10)))    # Compute gradient for this class
    return grad
end

# =============================================================================
# Simple Perturbation-based Explainer
# -----------------------------------------------------------------------------
# A basic method for attribution: compares input against a zero baseline.
# Approximation of influence using finite differences.
# =============================================================================
function perturbation_explainer(f, x; ε=1f-3)
    baseline = zero(x)  # Zero baseline input
    return (f(x) - f(baseline)) .* (x - baseline) / ε
end

end # module
