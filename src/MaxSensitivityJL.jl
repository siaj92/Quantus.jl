module MaxSensitivityJL

export MaxSensitivity, evaluate, gradient_explainer

using Statistics, Random, Zygote, Flux
using LinearAlgebra: norm

# Struct definition
struct MaxSensitivity
    nsamples::Int
    radius::Float32
    normtype::Int
end

# Constructor with defaults
MaxSensitivity(; nsamples=10, radius=0.05f0, normtype=2) = 
    MaxSensitivity(nsamples, radius, normtype)

# Evaluation function
function evaluate(metric::MaxSensitivity, model, explainer, x)
    original_expl = explainer(model, x)
    diffs = Float32[]
    for _ in 1:metric.nsamples
        δ = metric.radius * randn(Float32, size(x)...)
        x_perturbed = clamp.(x .+ δ, 0f0, 1f0)
        perturbed_expl = explainer(model, x_perturbed)
        diff = norm(perturbed_expl .- original_expl, metric.normtype)
        push!(diffs, diff)
    end
    return maximum(diffs)
end

# 1. Gradient-based explainer 
function gradient_explainer(f, x)
    ŷ, back = Zygote.pullback(f, x)
    class_index = only(Flux.onecold(ŷ))  # extract scalar
    grad = first(back(Flux.onehot(class_index, 1:10)))
    return grad
end

# 2. Simple perturbation-based explainer 
function perturbation_explainer(f, x; ε=1f-3)
    baseline = zero(x)
    (f(x) - f(baseline)) .* (x - baseline) / ε
end

end # module
