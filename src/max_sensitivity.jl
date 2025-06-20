
# MaxSensitivity.jl
module MaxSensitivityJL
    struct MaxSensitivity
        nsamples::Int       # Anzahl der Perturbationen
        radius::Float32     # Radius für Eingabestörungen
        normtype::Int       # Normtyp (z.B. 2 = L2-Norm)
    end

    # Optionaler Konstruktor mit Defaultwerten
    MaxSensitivity(; nsamples=10, radius=0.05f0, normtype=2) = 
        MaxSensitivity(nsamples, radius, normtype)

    # Evaluation der Sensitivität
    function evaluate(metric::MaxSensitivity, f, Φ, x)
        original_expl = Φ(f, x)
        diffs = Float32[]

        for _ in 1:metric.nsamples
            δ = metric.radius * randn(Float32, size(x)...)
            x_perturbed = clamp.(x .+ δ, 0f0, 1f0)
            perturbed_expl = Φ(f, x_perturbed)
            diff = norm(perturbed_expl .- original_expl, metric.normtype)
            push!(diffs, diff)
        end
        return maximum(diffs)
    end
    using MLDatasets
    using Flux
    using Flux: onehot, onecold, logitcrossentropy
    using Statistics
    using Random
    using Zygote

    # === 2. Gradient-basierte Erklärungsmethode ===
    function gradient_explainer(f, x)
        ŷ, back = Zygote.pullback(f, x)
        # Nehme höchste logit-Output-Klasse
        class_index = Flux.onecold(ŷ)
        grad = first(back(Flux.onehot(class_index, 1:10)))
        return grad
    end

    # === 3. CNN-Modell (wie in Vorlesung 7) ===
    model = Chain(
        Conv((3, 3), 1=>8, relu),
        MaxPool((2, 2)),
        Conv((3, 3), 8=>16, relu),
        MaxPool((2, 2)),
        flatten,
        Dense(256, 10),
        softmax
    )

    # === 5. Lade ein Bild aus MNIST ===
    train_x, train_y = MNIST.traindata()
    x = Float32.(train_x[:, :, 1]) ./ 255.0        # Normalisieren
    x = reshape(x, (28, 28, 1))                    # (H, W, C)
    # x ist das Input-Bild für evaluate()

    # === 6. Sensitivität berechnen ===
    metric = MaxSensitivity(nsamples=10, radius=0.02f0, normtype=2)

    sensitivity = evaluate(metric, model, gradient_explainer, x)

    println("MaxSensitivity für dieses MNIST-Bild: ", sensitivity)
end