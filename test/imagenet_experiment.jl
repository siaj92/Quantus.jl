
include("Quantus.jl")
include("helpers.jl")
include("plotting.jl")

using .Quantus
using .Helpers

using PythonCall

const torch = pyimport("torch")
const torchvision = pyimport("torchvision")
const quantus = pyimport("quantus")

# === Load model and data ===
model = torchvision.models.resnet18(pretrained=true)
is_cuda = pyconvert(Bool, torch.cuda.is_available())  # zuerst in Bool konvertieren
device = is_cuda ? "cuda:0" : "cpu"
model = model.to(device)

x_batch = torch.load("src/assets/imagenet_samples/x_batch.pt").to(device)
y_batch = torch.load("src/assets/imagenet_samples/y_batch.pt").to(device)
s_batch = torch.load("src/assets/imagenet_samples/s_batch.pt").to(device)

x_batch_np = Helpers.to_numpy(x_batch)
y_batch_np = Helpers.to_numpy(y_batch)
s_batch_np = Helpers.to_numpy(s_batch)
s_batch_np = s_batch_np.reshape(length(x_batch_np), 1, 224, 224)

# === Get attributions ===
a_batch = quantus.explain(model, x_batch, y_batch, method="Gradient")


# === Evaluate Faithfulness ===
scores = evaluate_faithfulness(
    model=model,
    x_batch=x_batch_np,
    y_batch=y_batch_np,
    a_batch=a_batch,
    device=device,
    nr_runs=50,
    subset_size=224,
    perturb_baseline="black",
    perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
    similarity_func=quantus.similarity_func.correlation_pearson,
    abs=false,
    return_aggregate=false
)

@info "Faithfulness scores: $scores"

