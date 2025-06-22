module RandomLogitJL

using Random
export RandomLogit, evaluate_batch

struct RandomLogit
  n_classes::Int
  seed::Int
  similarity_func::Function
end

function evaluate_batch(metric::RandomLogit, model, x_batch, y_batch, a_batch)
    rand_seed=randn(Xoshiro(seed),Int) # generating a random number on the base of a seed, making it pseudorandom if the user desires to reproduce tests
    batch_size = size(x_batch, 1)

    y_classes = collect(0:metric.n_classes-1)
    y_off = Vector{}
    for i in 1:batch_size
        y_filtered_classes = setdiff(y_classes, [y_batch[i]])
        y_off[i] = rand(rand_seed, y_filtered_classes)
    end
    a_perturbed=explain_batch(model,x_batch, y_off)

    score= metric.similarity_func(
        reshape(a_batch, batch_size, :)
        reshape(a_perturbed, batch_size, :)
    )
    return score 
end

end