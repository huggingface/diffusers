# Latent Consistency Model Multistep Scheduler

## Overview

Multistep and onestep scheduler (Algorithm 3) introduced alongside latent consistency models in the paper [Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference](https://arxiv.org/abs/2310.04378) by Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao.
This scheduler should be able to generate good samples from [`LatentConsistencyModelPipeline`] in 1-8 steps.

## LCMScheduler
[[autodoc]] LCMScheduler
