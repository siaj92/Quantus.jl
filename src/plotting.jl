module Plotting

using PythonCall
const plt = pyimport("matplotlib.pyplot")
const np = pyimport("numpy")
const random = pyimport("random")

include("helpers.jl")
using ..Helpers: normalize_image, to_numpy

"""
    plot_image_batch(x_batch, y_batch; nr_images=5)

Display a batch of input images with their predicted labels.

- `x_batch`: A batch of image tensors (e.g., from Torch).
- `y_batch`: Corresponding class labels.
- `nr_images`: Number of images to display (default: 5).
"""
function plot_image_batch(x_batch, y_batch; nr_images=5)
    figsize = (nr_images * 3, round(Int, nr_images * 2 / 3))
    fig, axes = plt.subplots(nrows=1, ncols=nr_images, figsize=figsize)

    for i in 0:(nr_images - 1)
        img = normalize_image(x_batch[i])
        axes[i].imshow(img, vmin=0.0, vmax=1.0, cmap="gray")
        label = pyconvert(Int, y_batch[i].item())
        axes[i].set_title("Label: $label")
        axes[i].axis("off")
    end

    plt.tight_layout()
    plt.show()
end

"""
    plot_example_explanation(x_batch, y_batch, a_batch; index=nothing)

Displays one image and its corresponding explanation heatmap side-by-side.

- `x_batch`: Image input batch.
- `y_batch`: Class labels.
- `a_batch`: Attribution maps.
- `index`: Optional index to specify which sample to plot. Random if not given.
"""
function plot_example_explanation(x_batch, y_batch, a_batch; index=nothing)
    if index === nothing
        index = random.randint(0, length(x_batch) - 1)
    end

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 5))

    # Normalize and plot image
    img = normalize_image(x_batch[index])
    axes[0].imshow(img, vmin=0.0, vmax=1.0)
    label = pyconvert(Int, y_batch[index].item())
    axes[0].set_title("Class $label")

    # Attribution map (convert to numpy and reshape to image)
    attr = pyconvert(Array, a_batch[index])
    reshaped = reshape(attr, 224, 224)  # Adjust if resolution differs
    heatmap = axes[1].imshow(reshaped, cmap="seismic")

    fig.colorbar(heatmap, fraction=0.03, pad=0.05)

    axes[0].axis("off")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
end

export plot_image_batch, plot_example_explanation

end # module

