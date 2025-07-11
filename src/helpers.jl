module Helpers
using PythonCall

const torch = pyimport("torch")
const np = pyimport("numpy")
const quantus = pyimport("quantus")
const pybuiltin = pyimport("builtins")

function normalize_image(arr)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    arr_copy = pyconvert(Bool, pybuiltin.isinstance(arr, torch.Tensor)) ? arr.clone().cpu().numpy() : arr.copy()
    arr_copy = quantus.normalise_func.denormalise(arr_copy, mean=mean, std=std)
    arr_copy = np.moveaxis(arr_copy, 0, -1)
    return (arr_copy * 255.0).astype(np.uint8)
end

function to_numpy(x)
    pyisinstance(x, torch.Tensor) ? x.cpu().numpy() : x
end

export normalize_image, to_numpy
end
