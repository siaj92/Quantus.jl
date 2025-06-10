module QuantusJL
    using PythonCall
    
    const abs_difference(x, y) = begin
        x_np_array = pyimport("numpy").array(x);
        y_np_array = pyimport("numpy").array(y);
        similarity_score = pyimport("quantus").functions.similarity_func.abs_difference(x_np_array, y_np_array);
        return pyconvert(Float64, similarity_score)
    end

    export abs_difference;
end

using QuantusJL
a = abs_difference([3.5, 3.0, 3.2], [2.5, 2.0, 2.2])

# typeof(a)