module SparsenessJL

export Sparseness



struct Sparseness
# empty, no need for particular inputs/intialization
end

function evaluate_batch(metric::Sparseness, a_batch)
    batch_size = size(a_batch, 1)
    n_features = prod(size(a_batch)[2:end])

    # Flatten the attribution maps
    a_flat = reshape(a_batch, batch_size, n_features)
    a_flat .+= 1e-7

    # Sorting
    a_sorted = sort(a_flat, dims=2)

    # Create an array of feature ranks [1, 2, ..., n_features] for each batch element
    ranks = repeat(reshape(1:n_features, (1, n_features)), batch_size, 1)

    # Sparseness score calculation
    numerator = sum((2 .* ranks .- n_features .- 1) .* a_sorted, dims=2)
    denominator = n_features .* sum(a_sorted, dims=2)

    scores = vec(numerator ./ denominator)

    return scores
end




end
