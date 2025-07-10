module FaithfulnessCorrelationJL

using Random
export FaithfulnessCorrelation, evaluate_batch



struct FaithfulnessCorrelation
  nr_runs::Int
  subset_size::Int
  similarity_func::Function
  perturb_func::Function 
end


function evaluate_batch(metric::FaithfulnessCorrelation, model, x_batch, y_batch, a_batch)
    # --- Initial setup ---
    batch_size = size(x_batch, 1)
    n_features = prod(size(x_batch)[2:end])  # flatten everything except batch dimension

    # --- Filter invalid labels ---
    logits_ref = model(x_batch)
    num_classes = size(logits_ref, 1)
    valid_idx = [i for i in 1:batch_size if 0 <= y_batch[i] < num_classes]
    if length(valid_idx) < batch_size
        @warn "Filtering out $(batch_size - length(valid_idx)) samples with invalid labels."
        x_batch = x_batch[valid_idx, :, :, :]
        y_batch = y_batch[valid_idx]
        a_batch = a_batch[valid_idx, :, :, :]
        batch_size = length(valid_idx)
        n_features = prod(size(x_batch)[2:end])
        logits_ref = model(x_batch)
    end

    # --- Flatten attributions ---
    a_flat = reshape(a_batch, batch_size, n_features)

    # --- Helper: extract target logits (0-based labels) ---
    function extract_logits(logits)
        preds = zeros(Float32, batch_size)
        for i in 1:batch_size
            idx = y_batch[i] + 1
            @assert 1 <= idx <= num_classes "Label index out of bounds: $idx"
            preds[i] = logits[idx, i]
        end
        return preds
    end

    # --- Original predictions ---
    y_orig = extract_logits(logits_ref)

    # --- Prepare accumulators ---
    deltas = zeros(Float32, batch_size, metric.nr_runs)
    sums   = zeros(Float32, batch_size, metric.nr_runs)

    # --- Main loop: perturb and predict ---
    for run in 1:metric.nr_runs
        # sample random feature subsets
        indices = [randperm(n_features)[1:metric.subset_size] for _ in 1:batch_size]

        # perturb inputs
        x_pert = metric.perturb_func(x_batch, indices)

        # predict on perturbed
        logits_pert = model(x_pert)
        y_pert = extract_logits(logits_pert)

        # record changes
        deltas[:, run] = y_orig .- y_pert
        for i in 1:batch_size
            sums[i, run] = sum(a_flat[i, indices[i]])
        end
    end

    # --- Compute and return similarity score ---
    return metric.similarity_func(sums, deltas)
end




end
