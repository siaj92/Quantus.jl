include("MNIST_ExAI.jl")
include("Preliminaries.jl")

#1: FaithfulnessCorrelation setup
metric=quantus.FaithfulnessCorrelation(
    nr_runs=100,  
    subset_size=224,  
    perturb_baseline="black",
    perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
    similarity_func=quantus.similarity_func.correlation_pearson,
    abs=false,  
    return_aggregate=false,
)

# FaithfulnessCorrelation with Python Model: 
a_batch=quantus.explain(model,x_batch, y_batch, method="Gradient")
scores=metric(
    model=model, 
    x_batch=x_batch_np, 
    y_batch=y_batch_np,
    a_batch= a_batch,
    device=device
)
@info scores

#= Faild Try to use FaithfulnessCorrelation with 
a_batch = quantus.explain(fluxModel, x_batch, y_batch, method="Gradient")
scores=metric(
    model=fluxModel, 
    x_batch=x_batch_np, 
    y_batch=y_batch_np,
    a_batch= a_batch,
    device=device
)
@info scores =#


#2: 
metric2=quantus.Sufficiency(
    threshold=0.6,
    return_aggregate=false,)
scores2=(model=model,
   x_batch=x_batch,
   y_batch=y_batch,
   a_batch=a_batch,
   device=device,)
