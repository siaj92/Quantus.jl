module MaxSensitivityPC
    using PythonCall
    const Py = PythonCall

    # Import necessary Python modules
    const torch = pyimport("torch")
    const plt = pyimport("matplotlib.pyplot")
    const np = pyimport("numpy")
    const quantus = pyimport("quantus")
    const torchvision = pyimport("torchvision")
    const pybuiltin = pyimport("builtins")
    random = pyimport("random")

    # === Adjust this path ===
    path_to_files = "src/assets/imagenet_samples"

    # === Load tensors ===
    x_batch = torch.load("$(path_to_files)/x_batch.pt")
    y_batch = torch.load("$(path_to_files)/y_batch.pt")
    s_batch = torch.load("$(path_to_files)/s_batch.pt")

    # === Move tensors to device (e.g., CUDA or CPU) ===
    is_cuda = pyconvert(Bool, torch.cuda.is_available())  # zuerst in Bool konvertieren
    device = is_cuda ? "cuda:0" : "cpu"
    x_batch = x_batch.to(device)
    s_batch = s_batch.to(device)
    y_batch = y_batch.to(device)


# === Define the normalization function ===
    function normalize_image(arr)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        if pyconvert(Bool, pybuiltin.isinstance(arr, torch.Tensor))
            arr_copy = arr.clone().cpu().numpy()
        else
            arr_copy = arr.copy()
        end

        arr_copy = quantus.normalise_func.denormalise(arr_copy, mean=mean, std=std)
        arr_copy = np.moveaxis(arr_copy, 0, -1)
        arr_copy = (arr_copy * 255.0).astype(np.uint8)

        return arr_copy
    end

   #=
    nr_images = 5
    figsize = (nr_images * 3, round(Int, nr_images * 2 / 3))
    fig, axes = plt.subplots(nrows=1, ncols=nr_images, figsize=figsize)

    for i in 0:(nr_images - 1)
        img = normalize_image(x_batch[i])
        axes[i].imshow(img, vmin=0.0, vmax=1.0, cmap="gray")
        axes[i].set_title("ImageNet class - $(y_batch[i].item())")
        axes[i].axis("off")
    end

    plt.show()
    =#
    model = torchvision.models.resnet18(pretrained=true)
    model = model.to(device)
    a_batch = quantus.explain(model, x_batch, y_batch, method="IntegratedGradients")

    function to_numpy(x)
        if pyisinstance(x, torch.Tensor)
            return x.cpu().numpy()
        else
            return x
        end
    end
    
    # Konvertiere alle Batchs sicher
    x_batch_np = to_numpy(x_batch)
    s_batch_np = to_numpy(s_batch)
    y_batch_np = to_numpy(y_batch)
    a_batch_np = to_numpy(a_batch)

    # === Reshape s_batch to (batch_size, 1, 224, 224) ===
    s_batch_np = s_batch_np.reshape(length(x_batch_np), 1, 224, 224)

        # index zufällig auswählen
    index = random.randint(0, length(x_batch) - 1)

    # Beispielhafte Erklärungen plotten
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 5))

    #= Bild normalisieren und anzeigen
    axes[0].imshow(normalize_image(x_batch[index]), vmin=0.0, vmax=1.0)
    axes[0].title.set_text("ImageNet class $(pyconvert(Int, y_batch[index].item()))")
    # Attribution anzeigen (reshape wie in Python)
    converted = pyconvert(Array, a_batch[index])
    reshaped = reshape(converted, 224, 224)
    exp = axes[1].imshow(reshaped, cmap="seismic")
    # Farbskala hinzufügen
    fig.colorbar(exp, fraction=0.03, pad=0.05)

    # Achsen ausblenden
    axes[0].axis("off")
    axes[1].axis("off")

    plt.show() =#


    # Build the metric instance
    metric = quantus.MaxSensitivity(
        nr_samples=10,
        lower_bound=0.2,
        norm_numerator=quantus.norm_func.fro_norm,
        norm_denominator=quantus.norm_func.fro_norm,
        perturb_func=quantus.perturb_func.uniform_noise,
        similarity_func=quantus.similarity_func.difference
    )
    
    dict = pyimport("builtins").dict
    explain_func_kwargs = dict([("method", "Gradient")])

    @info typeof(device)
    @info typeof(quantus.norm_func.fro_norm)
    scores = metric(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch = Py.Py(nothing),
        device=device,
        explain_func=quantus.explain,
        explain_func_kwargs=explain_func_kwargs
    )

end
