# Getting started

Welcome to QuantusJL! Please install the package by running the following:

using Pkg
Pkg.add(url="https://github.com/siaj92/Quantus.jl")

## Overview

Explainable AI (XAI) refers to methods and techniques in artificial intelligence designed to make the decisions and internal workings of AI systems understandable to humans. This package aims to implement select methods from the Python Quantus toolkit and produce metrics for different evaluation categories. 

## Supported metrics

# evaluate_faithfulness()
Faithfulness correlation is a metric used to evaluate how well an explanation method highlights the truly important features influencing a model’s prediction. It works by systematically perturbing input features based on their attribution scores and measuring how much the model’s output changes. A high correlation between feature importance and output change means the explanation is faithful to the model’s decision process.

Reference: [Quantus documentation](https://quantus.readthedocs.io/en/latest/docs_api/quantus.metrics.faithfulness.faithfulness_correlation.html#quantus.metrics.faithfulness.faithfulness_correlation.FaithfulnessCorrelation) and [“Sanity Checks for Saliency Maps”](https://arxiv.org/abs/1810.03292) paper.

# MaxSensitivityJL.evaluate_batch()
Max-Sensitivity measures the maximum change in an attribution map when the input is perturbed slightly. It is used to assess the robustness of explanation methods — robust explanations should not change much under small input perturbations.

Reference: [Quantus documentation](https://quantus.readthedocs.io/en/latest/docs_api/quantus.metrics.robustness.max_sensitivity.html).

# RandomLogitJL.evaluate_batch()
RandomLogit metric checks whether attribution maps for the true class are different from those for random classes. If they’re too similar, the explanation method may not depend on class. This is a sanity check for attribution methods.

Reference: [Quantus documentation](https://quantus.readthedocs.io/en/latest/docs_api/quantus.metrics.randomisation.random_logit.html).

# SparsenessJL.evaluate_batch()
The Sparseness metric quantifies how much the attribution is focused on a small subset of features. High sparseness means the explanation highlights only a few features as important.

Reference: [Quantus documentation](https://quantus.readthedocs.io/en/latest/docs_api/quantus.metrics.complexity.sparseness.html#quantus.metrics.complexity.sparseness.Sparseness.name).


## Other features

- **INCOMPLETE** [imagenet_experiment.jl](test/imagenet_experiment.jl) - attempt at a test, we only managed to get it to work on a ResNet18 Python model
- **INCOMPLETE** [plotting.jl](src/plotting.jl) - generating saliency maps, we only managed to get it to work with results from imagenet_experiment.jl
- [Metric_test.jl](src/Metric_test.jl) - script that shows how to set up and run your metrics on a batch of MNIST images, using custom similarity and perturbation functions, and prints the results for inspection. It’s a practical example of how an end user would use your Quantus.jl metrics..



