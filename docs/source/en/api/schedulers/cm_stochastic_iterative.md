# Consistency Model Multistep Scheduler

## Overview

Multistep and onestep scheduler (Algorithm 1) introduced alongside consistency models in the paper [Consistency Models](https://arxiv.org/abs/2303.01469) by Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever.
Based on the [original consistency models implementation](https://github.com/openai/consistency_models).
Should generate good samples from [`ConsistencyModelPipeline`] in one or a small number of steps.

## CMStochasticIterativeScheduler
[[autodoc]] CMStochasticIterativeScheduler

