# Getting started

Welcome to QuantusJL! As the work on the package is still in progress please install it by running the following:

using Pkg
Pkg.add(url="https://github.com/juliuswa/QuantusJL.jl")

## Overview

Explainable AI (XAI) refers to methods and techniques in artificial intelligence designed to make the decisions and internal workings of AI systems understandable to humans. This package aims to implement select methods from the Python Quantus toolkit and produce metrics for different evaluation categories. At the moment we implemented metrics for Robustness and Faithfulness - unfortunately the debugging process for those is still ongoing, so these aren't fully functional yet.

## Contents

- [MNIST_ExAI.jl](../../src/MNIST_ExAI.jl) **NOT WORKING** - training the MNIST model from class 7 and preparing it for evaluation
- [Preliminaries.jl](../../src/Preliminaries.jl) **WORKING** - importing a pre-trained ResNet-18 model for evaluation
- [MetricsEval.jl](../../src/MetricsEval.jl) **PARTIALLY WORKING** - implementing Faithfulness evaluation metrics, works on model created by Preliminaries.jl ("model", not "fluxModel")
- [MaxSensitivityJL.jl](../../src/MaxSensitivityJL.jl) **NOT WORKING** - implementing Robustness evaluation metrics
- [Metric_test.jl](../../src/Metric_test.jl), [RandomLogitJL.jl](../../src/RandomLogitJL.jl) **NOT WORKING** - used for testing.

## Instructions

Run the code in MetricsEval.jl](../../src/MetricsEval.jl) omitting the parts pertaining to [MNIST_ExAI.jl](../../src/MNIST_ExAI.jl) and "fluxModel".



