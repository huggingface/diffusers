<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# How to contribute to Diffusers 🧨

We ❤️ contributions from the open-source community! Everyone is welcome, and all types of participation –not just code– are valued and appreciated. Answering questions, helping others, reaching out, and improving the documentation are all immensely valuable to the community, so don't be afraid and get involved if you're up for it!

Everyone is encouraged to start by saying 👋 in our public Discord channel. We discuss the latest trends in diffusion models, ask questions, show off personal projects, help each other with contributions, or just hang out ☕. <a href="https://Discord.gg/G7tWnz98XR"><img alt="Join us on Discord" src="https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white"></a>

Whichever way you choose to contribute, we strive to be part of an open, welcoming, and kind community. Please, read our [code of conduct](https://github.com/huggingface/diffusers/blob/main/CODE_OF_CONDUCT.md) and be mindful to respect it during your interactions. We also recommend you become familiar with the [ethical guidelines](https://huggingface.co/docs/diffusers/conceptual/ethical_guidelines) that guide our project and ask you to adhere to the same principles of transparency and responsibility.

We enormously value feedback from the community, so please do not be afraid to speak up if you believe you have valuable feedback that can help improve the library - every message, comment, issue, and pull request (PR) is read and considered.

## Overview

You can contribute in many ways ranging from answering questions on issues and discussions to adding new diffusion models to the core library.

In the following, we give an overview of different ways to contribute, ranked by difficulty in ascending order. All of them are valuable to the community.

* 1. Asking and answering questions on [the Diffusers discussion forum](https://discuss.huggingface.co/c/discussion-related-to-httpsgithubcomhuggingfacediffusers) or on [Discord](https://discord.gg/G7tWnz98XR).
* 2. Opening new issues on [the GitHub Issues tab](https://github.com/huggingface/diffusers/issues/new/choose) or new discussions on [the GitHub Discussions tab](https://github.com/huggingface/diffusers/discussions/new/choose).
* 3. Answering issues on [the GitHub Issues tab](https://github.com/huggingface/diffusers/issues) or discussions on [the GitHub Discussions tab](https://github.com/huggingface/diffusers/discussions).
* 4. Fix a simple issue, marked by the "Good first issue" label, see [here](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22).
* 5. Contribute to the [documentation](https://github.com/huggingface/diffusers/tree/main/docs/source).
* 6. Contribute a [Community Pipeline](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3Acommunity-examples).
* 7. Contribute to the [examples](https://github.com/huggingface/diffusers/tree/main/examples).
* 8. Fix a more difficult issue, marked by the "Good second issue" label, see [here](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22Good+second+issue%22).
* 9. Add a new pipeline, model, or scheduler, see ["New Pipeline/Model"](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+pipeline%2Fmodel%22) and ["New scheduler"](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+scheduler%22) issues. For this contribution, please have a look at [Design Philosophy](https://github.com/huggingface/diffusers/blob/main/PHILOSOPHY.md).

As said before, **all contributions are valuable to the community**.
In the following, we will explain each contribution a bit more in detail.

For all contributions 4 - 9, you will need to open a PR. It is explained in detail how to do so in [Opening a pull request](#how-to-open-a-pr).

### 1. Asking and answering questions on the Diffusers discussion forum or on the Diffusers Discord

Any question or comment related to the Diffusers library can be asked on the [discussion forum](https://discuss.huggingface.co/c/discussion-related-to-httpsgithubcomhuggingfacediffusers/) or on [Discord](https://discord.gg/G7tWnz98XR). Such questions and comments include (but are not limited to):
- Reports of training or inference experiments in an attempt to share knowledge
- Presentation of personal projects
- Questions to non-official training examples
- Project proposals
- General feedback
- Paper summaries
- Asking for help on personal projects that build on top of the Diffusers library
- General questions
- Ethical questions regarding diffusion models
- ...

Every question that is asked on the forum or on Discord actively encourages the community to publicly
share knowledge and might very well help a beginner in the future who has the same question you're
having. Please do pose any questions you might have.
In the same spirit, you are of immense help to the community by answering such questions because this way you are publicly documenting knowledge for everybody to learn from.

**Please** keep in mind that the more effort you put into asking or answering a question, the higher
the quality of the publicly documented knowledge. In the same way, well-posed and well-answered questions create a high-quality knowledge database accessible to everybody, while badly posed questions or answers reduce the overall quality of the public knowledge database.
In short, a high quality question or answer is *precise*, *concise*, *relevant*, *easy-to-understand*, *accessible*, and *well-formatted/well-posed*. For more information, please have a look through the [How to write a good issue](#how-to-write-a-good-issue) section.

**NOTE about channels**:
[*The forum*](https://discuss.huggingface.co/c/discussion-related-to-httpsgithubcomhuggingfacediffusers/63) is much better indexed by search engines, such as Google. Posts are ranked by popularity rather than chronologically. Hence, it's easier to look up questions and answers that we posted some time ago.
In addition, questions and answers posted in the forum can easily be linked to.
In contrast, *Discord* has a chat-like format that invites fast back-and-forth communication.
While it will most likely take less time for you to get an answer to your question on Discord, your
question won't be visible anymore over time. Also, it's much harder to find information that was posted a while back on Discord. We therefore strongly recommend using the forum for high-quality questions and answers in an attempt to create long-lasting knowledge for the community. If discussions on Discord lead to very interesting answers and conclusions, we recommend posting the results on the forum to make the information more available for future readers.

### 2. Opening new issues on the GitHub issues tab

The 🧨 Diffusers library is robust and reliable thanks to the users who notify us of
the problems they encounter. So thank you for reporting an issue.

Remember, GitHub issues are reserved for technical questions directly related to the Diffusers library, bug reports, feature requests, or feedback on the library design.

In a nutshell, this means that everything that is **not** related to the **code of the Diffusers library** (including the documentation) should **not** be asked on GitHub, but rather on either the [forum](https://discuss.huggingface.co/c/discussion-related-to-httpsgithubcomhuggingfacediffusers/63) or [Discord](https://discord.gg/G7tWnz98XR).

**Please consider the following guidelines when opening a new issue**:
- Make sure you have searched whether your issue has already been asked before (use the search bar on GitHub under Issues).
- Please never report a new issue on another (related) issue. If another issue is highly related, please
open a new issue nevertheless and link to the related issue.
- Make sure your issue is written in English. Please use one of the great, free online translation services, such as [DeepL](https://www.deepl.com/translator) to translate from your native language to English if you are not comfortable in English.
- Check whether your issue might be solved by updating to the newest Diffusers version. Before posting your issue, please make sure that `python -c "import diffusers; print(diffusers.__version__)"` is higher or matches the latest Diffusers version.
- Remember that the more effort you put into opening a new issue, the higher the quality of your answer will be and the better the overall quality of the Diffusers issues.

New issues usually include the following.

#### 2.1. Reproducible, minimal bug reports

A bug report should always have a reproducible code snippet and be as minimal and concise as possible.
This means in more detail:
- Narrow the bug down as much as you can, **do not just dump your whole code file**.
- Format your code.
- Do not include any external libraries except for Diffusers depending on them.
- **Always** provide all necessary information about your environment; for this, you can run: `diffusers-cli env` in your shell and copy-paste the displayed information to the issue.
- Explain the issue. If the reader doesn't know what the issue is and why it is an issue, (s)he cannot solve it.
- **Always** make sure the reader can reproduce your issue with as little effort as possible. If your code snippet cannot be run because of missing libraries or undefined variables, the reader cannot help you. Make sure your reproducible code snippet is as minimal as possible and can be copy-pasted into a simple Python shell.
- If in order to reproduce your issue a model and/or dataset is required, make sure the reader has access to that model or dataset. You can always upload your model or dataset to the [Hub](https://huggingface.co) to make it easily downloadable. Try to keep your model and dataset as small as possible, to make the reproduction of your issue as effortless as possible.

For more information, please have a look through the [How to write a good issue](#how-to-write-a-good-issue) section.

You can open a bug report [here](https://github.com/huggingface/diffusers/issues/new?assignees=&labels=bug&projects=&template=bug-report.yml).

#### 2.2. Feature requests

A world-class feature request addresses the following points:

1. Motivation first:
* Is it related to a problem/frustration with the library? If so, please explain
why. Providing a code snippet that demonstrates the problem is best.
* Is it related to something you would need for a project? We'd love to hear
about it!
* Is it something you worked on and think could benefit the community?
Awesome! Tell us what problem it solved for you.
2. Write a *full paragraph* describing the feature;
3. Provide a **code snippet** that demonstrates its future use;
4. In case this is related to a paper, please attach a link;
5. Attach any additional information (drawings, screenshots, etc.) you think may help.

You can open a feature request [here](https://github.com/huggingface/diffusers/issues/new?assignees=&labels=&template=feature_request.md&title=).

#### 2.3 Feedback

Feedback about the library design and why it is good or not good helps the core maintainers immensely to build a user-friendly library. To understand the philosophy behind the current design philosophy, please have a look [here](https://huggingface.co/docs/diffusers/conceptual/philosophy). If you feel like a certain design choice does not fit with the current design philosophy, please explain why and how it should be changed. If a certain design choice follows the design philosophy too much, hence restricting use cases, explain why and how it should be changed.
If a certain design choice is very useful for you, please also leave a note as this is great feedback for future design decisions.

You can open an issue about feedback [here](https://github.com/huggingface/diffusers/issues/new?assignees=&labels=&template=feedback.md&title=).

#### 2.4 Technical questions

Technical questions are mainly about why certain code of the library was written in a certain way, or what a certain part of the code does. Please make sure to link to the code in question and please provide details on
why this part of the code is difficult to understand.

You can open an issue about a technical question [here](https://github.com/huggingface/diffusers/issues/new?assignees=&labels=bug&template=bug-report.yml).

#### 2.5 Proposal to add a new model, scheduler, or pipeline

If the diffusion model community released a new model, pipeline, or scheduler that you would like to see in the Diffusers library, please provide the following information:

* Short description of the diffusion pipeline, model, or scheduler and link to the paper or public release.
* Link to any of its open-source implementation(s).
* Link to the model weights if they are available.

If you are willing to contribute to the model yourself, let us know so we can best guide you. Also, don't forget
to tag the original author of the component (model, scheduler, pipeline, etc.) by GitHub handle if you can find it.

You can open a request for a model/pipeline/scheduler [here](https://github.com/huggingface/diffusers/issues/new?assignees=&labels=New+model%2Fpipeline%2Fscheduler&template=new-model-addition.yml).

### 3. Answering issues on the GitHub issues tab

Answering issues on GitHub might require some technical knowledge of Diffusers, but we encourage everybody to give it a try even if you are not 100% certain that your answer is correct.
Some tips to give a high-quality answer to an issue:
- Be as concise and minimal as possible.
- Stay on topic. An answer to the issue should concern the issue and only the issue.
- Provide links to code, papers, or other sources that prove or encourage your point.
- Answer in code. If a simple code snippet is the answer to the issue or shows how the issue can be solved, please provide a fully reproducible code snippet.

Also, many issues tend to be simply off-topic, duplicates of other issues, or irrelevant. It is of great
help to the maintainers if you can answer such issues, encouraging the author of the issue to be
more precise, provide the link to a duplicated issue or redirect them to [the forum](https://discuss.huggingface.co/c/discussion-related-to-httpsgithubcomhuggingfacediffusers/63) or [Discord](https://discord.gg/G7tWnz98XR).

If you have verified that the issued bug report is correct and requires a correction in the source code,
please have a look at the next sections.

For all of the following contributions, you will need to open a PR. It is explained in detail how to do so in the [Opening a pull request](#how-to-open-a-pr) section.

### 4. Fixing a "Good first issue"

*Good first issues* are marked by the [Good first issue](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22) label. Usually, the issue already
explains how a potential solution should look so that it is easier to fix.
If the issue hasn't been closed and you would like to try to fix this issue, you can just leave a message "I would like to try this issue.". There are usually three scenarios:
- a.) The issue description already proposes a fix. In this case and if the solution makes sense to you, you can open a PR or draft PR to fix it.
- b.) The issue description does not propose a fix. In this case, you can ask what a proposed fix could look like and someone from the Diffusers team should answer shortly. If you have a good idea of how to fix it, feel free to directly open a PR.
- c.) There is already an open PR to fix the issue, but the issue hasn't been closed yet. If the PR has gone stale, you can simply open a new PR and link to the stale PR. PRs often go stale if the original contributor who wanted to fix the issue suddenly cannot find the time anymore to proceed. This often happens in open-source and is very normal. In this case, the community will be very happy if you give it a new try and leverage the knowledge of the existing PR. If there is already a PR and it is active, you can help the author by giving suggestions, reviewing the PR or even asking whether you can contribute to the PR.


### 5. Contribute to the documentation

A good library **always** has good documentation! The official documentation is often one of the first points of contact for new users of the library, and therefore contributing to the documentation is a **highly
valuable contribution**.

Contributing to the library can have many forms:

- Correcting spelling or grammatical errors.
- Correct incorrect formatting of the docstring. If you see that the official documentation is weirdly displayed or a link is broken, we would be very happy if you take some time to correct it.
- Correct the shape or dimensions of a docstring input or output tensor.
- Clarify documentation that is hard to understand or incorrect.
- Update outdated code examples.
- Translating the documentation to another language.

Anything displayed on [the official Diffusers doc page](https://huggingface.co/docs/diffusers/index) is part of the official documentation and can be corrected, adjusted in the respective [documentation source](https://github.com/huggingface/diffusers/tree/main/docs/source).

Please have a look at [this page](https://github.com/huggingface/diffusers/tree/main/docs) on how to verify changes made to the documentation locally.

### 6. Contribute a community pipeline

> [!TIP]
> Read the [Community pipelines](../using-diffusers/custom_pipeline_overview#community-pipelines) guide to learn more about the difference between a GitHub and Hugging Face Hub community pipeline. If you're interested in why we have community pipelines, take a look at GitHub Issue [#841](https://github.com/huggingface/diffusers/issues/841) (basically, we can't maintain all the possible ways diffusion models can be used for inference but we also don't want to prevent the community from building them).

Contributing a community pipeline is a great way to share your creativity and work with the community. It lets you build on top of the [`DiffusionPipeline`] so that anyone can load and use it by setting the `custom_pipeline` parameter. This section will walk you through how to create a simple pipeline where the UNet only does a single forward pass and calls the scheduler once (a "one-step" pipeline).

1. Create a one_step_unet.py file for your community pipeline. This file can contain whatever package you want to use as long as it's installed by the user. Make sure you only have one pipeline class that inherits from [`DiffusionPipeline`] to load model weights and the scheduler configuration from the Hub. Add a UNet and scheduler to the `__init__` function.

    You should also add the `register_modules` function to ensure your pipeline and its components can be saved with [`~DiffusionPipeline.save_pretrained`].

```py
from diffusers import DiffusionPipeline
import torch

class UnetSchedulerOneForwardPipeline(DiffusionPipeline):
    def __init__(self, unet, scheduler):
        super().__init__()

        self.register_modules(unet=unet, scheduler=scheduler)
```

1. In the forward pass (which we recommend defining as `__call__`), you can add any feature you'd like. For the "one-step" pipeline, create a random image and call the UNet and scheduler once by setting `timestep=1`.

```py
  from diffusers import DiffusionPipeline
  import torch

  class UnetSchedulerOneForwardPipeline(DiffusionPipeline):
      def __init__(self, unet, scheduler):
          super().__init__()

          self.register_modules(unet=unet, scheduler=scheduler)

      def __call__(self):
          image = torch.randn(
              (1, self.unet.config.in_channels, self.unet.config.sample_size, self.unet.config.sample_size),
          )
          timestep = 1

          model_output = self.unet(image, timestep).sample
          scheduler_output = self.scheduler.step(model_output, timestep, image).prev_sample

          return scheduler_output
```

Now you can run the pipeline by passing a UNet and scheduler to it or load pretrained weights if the pipeline structure is identical.

```py
from diffusers import DDPMScheduler, UNet2DModel

scheduler = DDPMScheduler()
unet = UNet2DModel()

pipeline = UnetSchedulerOneForwardPipeline(unet=unet, scheduler=scheduler)
output = pipeline()
# load pretrained weights
pipeline = UnetSchedulerOneForwardPipeline.from_pretrained("google/ddpm-cifar10-32", use_safetensors=True)
output = pipeline()
```

You can either share your pipeline as a GitHub community pipeline or Hub community pipeline.

<hfoptions id="pipeline type">
<hfoption id="GitHub pipeline">

Share your GitHub pipeline by opening a pull request on the Diffusers [repository](https://github.com/huggingface/diffusers) and add the one_step_unet.py file to the [examples/community](https://github.com/huggingface/diffusers/tree/main/examples/community) subfolder.

</hfoption>
<hfoption id="Hub pipeline">

Share your Hub pipeline by creating a model repository on the Hub and uploading the one_step_unet.py file to it.

</hfoption>
</hfoptions>

### 7. Contribute to training examples

Diffusers examples are a collection of training scripts that reside in [examples](https://github.com/huggingface/diffusers/tree/main/examples).

We support two types of training examples:

- Official training examples
- Research training examples

Research training examples are located in [examples/research_projects](https://github.com/huggingface/diffusers/tree/main/examples/research_projects) whereas official training examples include all folders under [examples](https://github.com/huggingface/diffusers/tree/main/examples) except the `research_projects` and `community` folders.
The official training examples are maintained by the Diffusers' core maintainers whereas the research training examples are maintained by the community.
This is because of the same reasons put forward in [6. Contribute a community pipeline](#6-contribute-a-community-pipeline) for official pipelines vs. community pipelines: It is not feasible for the core maintainers to maintain all possible training methods for diffusion models.
If the Diffusers core maintainers and the community consider a certain training paradigm to be too experimental or not popular enough, the corresponding training code should be put in the `research_projects` folder and maintained by the author.

Both official training and research examples consist of a directory that contains one or more training scripts, a `requirements.txt` file, and a `README.md` file. In order for the user to make use of the
training examples, it is required to clone the repository:

```bash
git clone https://github.com/huggingface/diffusers
```

as well as to install all additional dependencies required for training:

```bash
cd diffusers
pip install -r examples/<your-example-folder>/requirements.txt
```

Therefore when adding an example, the `requirements.txt` file shall define all pip dependencies required for your training example so that once all those are installed, the user can run the example's training script. See, for example, the [DreamBooth `requirements.txt` file](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/requirements.txt).

Training examples of the Diffusers library should adhere to the following philosophy:
- All the code necessary to run the examples should be found in a single Python file.
- One should be able to run the example from the command line with `python <your-example>.py --args`.
- Examples should be kept simple and serve as **an example** on how to use Diffusers for training. The purpose of example scripts is **not** to create state-of-the-art diffusion models, but rather to reproduce known training schemes without adding too much custom logic. As a byproduct of this point, our examples also strive to serve as good educational materials.

To contribute an example, it is highly recommended to look at already existing examples such as [dreambooth](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py) to get an idea of how they should look like.
We strongly advise contributors to make use of the [Accelerate library](https://github.com/huggingface/accelerate) as it's tightly integrated
with Diffusers.
Once an example script works, please make sure to add a comprehensive `README.md` that states how to use the example exactly. This README should include:
- An example command on how to run the example script as shown [here](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth#running-locally-with-pytorch).
- A link to some training results (logs, models, etc.) that show what the user can expect as shown [here](https://api.wandb.ai/report/patrickvonplaten/xm6cd5q5).
- If you are adding a non-official/research training example, **please don't forget** to add a sentence that you are maintaining this training example which includes your git handle as shown [here](https://github.com/huggingface/diffusers/tree/main/examples/research_projects/intel_opts#diffusers-examples-with-intel-optimizations).

If you are contributing to the official training examples, please also make sure to add a test to its folder such as [examples/dreambooth/test_dreambooth.py](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/test_dreambooth.py). This is not necessary for non-official training examples.

### 8. Fixing a "Good second issue"

*Good second issues* are marked by the [Good second issue](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22Good+second+issue%22) label. Good second issues are
usually more complicated to solve than [Good first issues](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22).
The issue description usually gives less guidance on how to fix the issue and requires
a decent understanding of the library by the interested contributor.
If you are interested in tackling a good second issue, feel free to open a PR to fix it and link the PR to the issue. If you see that a PR has already been opened for this issue but did not get merged, have a look to understand why it wasn't merged and try to open an improved PR.
Good second issues are usually more difficult to get merged compared to good first issues, so don't hesitate to ask for help from the core maintainers. If your PR is almost finished the core maintainers can also jump into your PR and commit to it in order to get it merged.

### 9. Adding pipelines, models, schedulers

Pipelines, models, and schedulers are the most important pieces of the Diffusers library.
They provide easy access to state-of-the-art diffusion technologies and thus allow the community to
build powerful generative AI applications.

By adding a new model, pipeline, or scheduler you might enable a new powerful use case for any of the user interfaces relying on Diffusers which can be of immense value for the whole generative AI ecosystem.

Diffusers has a couple of open feature requests for all three components - feel free to gloss over them
if you don't know yet what specific component you would like to add:
- [Model or pipeline](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+pipeline%2Fmodel%22)
- [Scheduler](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+scheduler%22)

Before adding any of the three components, it is strongly recommended that you give the [Philosophy guide](philosophy) a read to better understand the design of any of the three components. Please be aware that we cannot merge model, scheduler, or pipeline additions that strongly diverge from our design philosophy
as it will lead to API inconsistencies. If you fundamentally disagree with a design choice, please open a [Feedback issue](https://github.com/huggingface/diffusers/issues/new?assignees=&labels=&template=feedback.md&title=) instead so that it can be discussed whether a certain design pattern/design choice shall be changed everywhere in the library and whether we shall update our design philosophy. Consistency across the library is very important for us.

Please make sure to add links to the original codebase/paper to the PR and ideally also ping the original author directly on the PR so that they can follow the progress and potentially help with questions.

If you are unsure or stuck in the PR, don't hesitate to leave a message to ask for a first review or help.

#### Copied from mechanism

A unique and important feature to understand when adding any pipeline, model or scheduler code is the `# Copied from` mechanism. You'll see this all over the Diffusers codebase, and the reason we use it is to keep the codebase easy to understand and maintain. Marking code with the `# Copied from` mechanism forces the marked code to be identical to the code it was copied from. This makes it easy to update and propagate changes across many files whenever you run `make fix-copies`.

For example, in the code example below, [`~diffusers.pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is the original code and `AltDiffusionPipelineOutput` uses the `# Copied from` mechanism to copy it. The only difference is changing the class prefix from `Stable` to `Alt`.

```py
# Copied from diffusers.pipelines.stable_diffusion.pipeline_output.StableDiffusionPipelineOutput with Stable->Alt
class AltDiffusionPipelineOutput(BaseOutput):
    """
    Output class for Alt Diffusion pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`.
        nsfw_content_detected (`List[bool]`)
            List indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content or
            `None` if safety checking could not be performed.
    """
```

To learn more, read this section of the [~Don't~ Repeat Yourself*](https://huggingface.co/blog/transformers-design-philosophy#4-machine-learning-models-are-static) blog post.

## How to write a good issue

**The better your issue is written, the higher the chances that it will be quickly resolved.**

1. Make sure that you've used the correct template for your issue. You can pick between *Bug Report*, *Feature Request*, *Feedback about API Design*, *New model/pipeline/scheduler addition*, *Forum*, or a blank issue. Make sure to pick the correct one when opening [a new issue](https://github.com/huggingface/diffusers/issues/new/choose).
2. **Be precise**: Give your issue a fitting title. Try to formulate your issue description as simple as possible. The more precise you are when submitting an issue, the less time it takes to understand the issue and potentially solve it. Make sure to open an issue for one issue only and not for multiple issues. If you found multiple issues, simply open multiple issues. If your issue is a bug, try to be as precise as possible about what bug it is - you should not just write "Error in diffusers".
3. **Reproducibility**: No reproducible code snippet == no solution. If you encounter a bug, maintainers **have to be able to reproduce** it. Make sure that you include a code snippet that can be copy-pasted into a Python interpreter to reproduce the issue. Make sure that your code snippet works, *i.e.* that there are no missing imports or missing links to images, ... Your issue should contain an error message **and** a code snippet that can be copy-pasted without any changes to reproduce the exact same error message. If your issue is using local model weights or local data that cannot be accessed by the reader, the issue cannot be solved. If you cannot share your data or model, try to make a dummy model or dummy data.
4. **Minimalistic**: Try to help the reader as much as you can to understand the issue as quickly as possible by staying as concise as possible. Remove all code / all information that is irrelevant to the issue. If you have found a bug, try to create the easiest code example you can to demonstrate your issue, do not just dump your whole workflow into the issue as soon as you have found a bug. E.g., if you train a model and get an error at some point during the training, you should first try to understand what part of the training code is responsible for the error and try to reproduce it with a couple of lines. Try to use dummy data instead of full datasets.
5. Add links. If you are referring to a certain naming, method, or model make sure to provide a link so that the reader can better understand what you mean. If you are referring to a specific PR or issue, make sure to link it to your issue. Do not assume that the reader knows what you are talking about. The more links you add to your issue the better.
6. Formatting. Make sure to nicely format your issue by formatting code into Python code syntax, and error messages into normal code syntax. See the [official GitHub formatting docs](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax) for more information.
7. Think of your issue not as a ticket to be solved, but rather as a beautiful entry to a well-written encyclopedia. Every added issue is a contribution to publicly available knowledge. By adding a nicely written issue you not only make it easier for maintainers to solve your issue, but you are helping the whole community to better understand a certain aspect of the library.

## How to write a good PR

1. Be a chameleon. Understand existing design patterns and syntax and make sure your code additions flow seamlessly into the existing code base. Pull requests that significantly diverge from existing design patterns or user interfaces will not be merged.
2. Be laser focused. A pull request should solve one problem and one problem only. Make sure to not fall into the trap of "also fixing another problem while we're adding it". It is much more difficult to review pull requests that solve multiple, unrelated problems at once.
3. If helpful, try to add a code snippet that displays an example of how your addition can be used.
4. The title of your pull request should be a summary of its contribution.
5. If your pull request addresses an issue, please mention the issue number in
the pull request description to make sure they are linked (and people
consulting the issue know you are working on it);
6. To indicate a work in progress please prefix the title with `[WIP]`. These
are useful to avoid duplicated work, and to differentiate it from PRs ready
to be merged;
7. Try to formulate and format your text as explained in [How to write a good issue](#how-to-write-a-good-issue).
8. Make sure existing tests pass;
9. Add high-coverage tests. No quality testing = no merge.
- If you are adding new `@slow` tests, make sure they pass using
`RUN_SLOW=1 python -m pytest tests/test_my_new_model.py`.
CircleCI does not run the slow tests, but GitHub Actions does every night!
10. All public methods must have informative docstrings that work nicely with markdown. See [`pipeline_latent_diffusion.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/latent_diffusion/pipeline_latent_diffusion.py) for an example.
11. Due to the rapidly growing repository, it is important to make sure that no files that would significantly weigh down the repository are added. This includes images, videos, and other non-text files. We prefer to leverage a hf.co hosted `dataset` like
[`hf-internal-testing`](https://huggingface.co/hf-internal-testing) or [huggingface/documentation-images](https://huggingface.co/datasets/huggingface/documentation-images) to place these files.
If an external contribution, feel free to add the images to your PR and ask a Hugging Face member to migrate your images
to this dataset.

## How to open a PR

Before writing code, we strongly advise you to search through the existing PRs or
issues to make sure that nobody is already working on the same thing. If you are
unsure, it is always a good idea to open an issue to get some feedback.

You will need basic `git` proficiency to be able to contribute to
🧨 Diffusers. `git` is not the easiest tool to use but it has the greatest
manual. Type `git --help` in a shell and enjoy. If you prefer books, [Pro
Git](https://git-scm.com/book/en/v2) is a very good reference.

Follow these steps to start contributing ([supported Python versions](https://github.com/huggingface/diffusers/blob/83bc6c94eaeb6f7704a2a428931cf2d9ad973ae9/setup.py#L270)):

1. Fork the [repository](https://github.com/huggingface/diffusers) by
clicking on the 'Fork' button on the repository's page. This creates a copy of the code
under your GitHub user account.

2. Clone your fork to your local disk, and add the base repository as a remote:

 ```bash
 $ git clone git@github.com:<your GitHub handle>/diffusers.git
 $ cd diffusers
 $ git remote add upstream https://github.com/huggingface/diffusers.git
 ```

3. Create a new branch to hold your development changes:

 ```bash
 $ git checkout -b a-descriptive-name-for-my-changes
 ```

**Do not** work on the `main` branch.

4. Set up a development environment by running the following command in a virtual environment:

 ```bash
 $ pip install -e ".[dev]"
 ```

If you have already cloned the repo, you might need to `git pull` to get the most recent changes in the
library.

5. Develop the features on your branch.

As you work on the features, you should make sure that the test suite
passes. You should run the tests impacted by your changes like this:

 ```bash
 $ pytest tests/<TEST_TO_RUN>.py
 ```

Before you run the tests, please make sure you install the dependencies required for testing. You can do so
with this command:

 ```bash
 $ pip install -e ".[test]"
 ```

You can also run the full test suite with the following command, but it takes
a beefy machine to produce a result in a decent amount of time now that
Diffusers has grown a lot. Here is the command for it:

 ```bash
 $ make test
 ```

🧨 Diffusers relies on `black` and `isort` to format its source code
consistently. After you make changes, apply automatic style corrections and code verifications
that can't be automated in one go with:

 ```bash
 $ make style
 ```

🧨 Diffusers also uses `ruff` and a few custom scripts to check for coding mistakes. Quality
control runs in CI, however, you can also run the same checks with:

 ```bash
 $ make quality
 ```

Once you're happy with your changes, add changed files using `git add` and
make a commit with `git commit` to record your changes locally:

 ```bash
 $ git add modified_file.py
 $ git commit -m "A descriptive message about your changes."
 ```

It is a good idea to sync your copy of the code with the original
repository regularly. This way you can quickly account for changes:

 ```bash
 $ git pull upstream main
 ```

Push the changes to your account using:

 ```bash
 $ git push -u origin a-descriptive-name-for-my-changes
 ```

6. Once you are satisfied, go to the
webpage of your fork on GitHub. Click on 'Pull request' to send your changes
to the project maintainers for review.

7. It's OK if maintainers ask you for changes. It happens to core contributors
too! So everyone can see the changes in the Pull request, work in your local
branch and push the changes to your fork. They will automatically appear in
the pull request.

### Tests

An extensive test suite is included to test the library behavior and several examples. Library tests can be found in
the [tests folder](https://github.com/huggingface/diffusers/tree/main/tests).

We like `pytest` and `pytest-xdist` because it's faster. From the root of the
repository, here's how to run tests with `pytest` for the library:

```bash
$ python -m pytest -n auto --dist=loadfile -s -v ./tests/
```

In fact, that's how `make test` is implemented!

You can specify a smaller set of tests in order to test only the feature
you're working on.

By default, slow tests are skipped. Set the `RUN_SLOW` environment variable to
`yes` to run them. This will download many gigabytes of models — make sure you
have enough disk space and a good Internet connection, or a lot of patience!

```bash
$ RUN_SLOW=yes python -m pytest -n auto --dist=loadfile -s -v ./tests/
```

`unittest` is fully supported, here's how to run tests with it:

```bash
$ python -m unittest discover -s tests -t . -v
$ python -m unittest discover -s examples -t examples -v
```

### Syncing forked main with upstream (HuggingFace) main

To avoid pinging the upstream repository which adds reference notes to each upstream PR and sends unnecessary notifications to the developers involved in these PRs,
when syncing the main branch of a forked repository, please, follow these steps:
1. When possible, avoid syncing with the upstream using a branch and PR on the forked repository. Instead, merge directly into the forked main.
2. If a PR is absolutely necessary, use the following steps after checking out your branch:
```bash
$ git checkout -b your-branch-for-syncing
$ git pull --squash --no-commit upstream main
$ git commit -m '<your message without GitHub references>'
$ git push --set-upstream origin your-branch-for-syncing
```

### Style guide

For documentation strings, 🧨 Diffusers follows the [Google style](https://google.github.io/styleguide/pyguide.html).