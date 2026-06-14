## Overview

We are developing benchmarks and methods for developing "models of
models"---specifically, trying to build predictors that can look at a target
neural networks behavior (and possibly pattern of activation) on some training
distribution, and predict how it will perform (either at the dataset level or
the instance level).

The codebase has three components:

1. Dataset construction
2. Subject model construction
3. Predictor construction and evaluation

See `docs/DATASETS.md`, `docs/MODELS.md` and `docs/PREDICTORS.md` for a
description of each component.

## Environment

You have access to a machine with 8 A100 GPUs at align-3.csail.mit.edu. If at
any point you are not able to SSH into the machine, stop and wait for further
instructions rather than repeatedly trying to log in. Only run your job on free
GPUs; if there are no GPUs available you should again stop and wait.

Develop code locally and upload it to /raid/lingo/jda/code/darpa3 only when you
need to train/evaluate neural models. Write Python and place libraries etc. in a
virtualenv.

## Documentation

Whenever more details about the target implementation become available (e.g. you
ask a clarifying question, or something is underspecified in the documentation
and you make a specific implementation decision) you should update this file or
the relevant file in `docs`. 
