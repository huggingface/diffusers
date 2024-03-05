# UFOGen Multistep and Single-Step Scheduler

## Overview

Multistep and onestep scheduler introduced with the UFOGen model in the paper [UFOGen: You Forward Once Large Scale Text-to-Image Generation via Diffusion GANs](https://arxiv.org/abs/2311.09257) by Yanwu Xu, Yang Zhao, Zhisheng Xiao, and Tingbo Hou.
This scheduler should be able to generate good samples from a UFOGen model in 1-4 steps.

<Tip warning={true}>

Multistep sampling support is currently experimental.

</Tip>

## UFOGenScheduler
[[autodoc]] UFOGenScheduler
