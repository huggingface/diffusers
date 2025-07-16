<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# AutoPipelineBlocks

<Tip warning={true}>

🧪 **Experimental Feature**: Modular Diffusers is an experimental feature we are actively developing. The API may be subject to breaking changes.

</Tip>

`AutoPipelineBlocks` is a subclass of `ModularPipelineBlocks`. It is a multi-block that automatically selects which sub-blocks to run based on the inputs provided at runtime, creating conditional workflows that adapt to different scenarios. The main purpose is convenience and portability - for developers, you can package everything into one workflow, making it easier to share and use.

In this tutorial, we will show you how to create an `AutoPipelineBlocks` and learn more about how the conditional selection works.

<Tip>

Other types of multi-blocks include [SequentialPipelineBlocks](sequential_pipeline_blocks.md) (for linear workflows) and [LoopSequentialPipelineBlocks](loop_sequential_pipeline_blocks.md) (for iterative workflows). For information on creating individual blocks, see the [PipelineBlock guide](pipeline_block.md).

Additionally, like all `ModularPipelineBlocks`, `AutoPipelineBlocks` are definitions/specifications, not runnable pipelines. You need to convert them into a `ModularPipeline` to actually execute them. For information on creating and running pipelines, see the [Modular Pipeline guide](modular_pipeline.md).

</Tip>

For example, you might want to support text-to-image and image-to-image tasks. Instead of creating two separate pipelines, you can create an `AutoPipelineBlocks` that automatically chooses the workflow based on whether an `image` input is provided.

Let's see an example. We'll use the helper function from the [PipelineBlock guide](./pipeline_block.md) to create our blocks:

**Helper Function**

```py
from diffusers.modular_pipelines import PipelineBlock, InputParam, OutputParam
import torch

def make_block(inputs=[], intermediate_inputs=[], intermediate_outputs=[], block_fn=None, description=None):
    class TestBlock(PipelineBlock):
        model_name = "test"
        
        @property
        def inputs(self):
            return inputs
            
        @property
        def intermediate_inputs(self):
            return intermediate_inputs
            
        @property
        def intermediate_outputs(self):
            return intermediate_outputs
            
        @property
        def description(self):
            return description if description is not None else ""
            
        def __call__(self, components, state):
            block_state = self.get_block_state(state)
            if block_fn is not None:
                block_state = block_fn(block_state, state)
            self.set_block_state(state, block_state)
            return components, state
    
    return TestBlock
```

Now let's create a dummy `AutoPipelineBlocks` that includes dummy text-to-image, image-to-image, and inpaint pipelines.


```py
from diffusers.modular_pipelines import AutoPipelineBlocks 

# These are dummy blocks and we only focus on "inputs" for our purpose
inputs = [InputParam(name="prompt")]
# block_fn prints out which workflow is running so we can see the execution order at runtime
block_fn = lambda x, y: print("running the text-to-image workflow")
block_t2i_cls = make_block(inputs=inputs, block_fn=block_fn, description="I'm a text-to-image workflow!")

inputs = [InputParam(name="prompt"), InputParam(name="image")]
block_fn = lambda x, y: print("running the image-to-image workflow")
block_i2i_cls = make_block(inputs=inputs, block_fn=block_fn, description="I'm a image-to-image workflow!")

inputs = [InputParam(name="prompt"), InputParam(name="image"), InputParam(name="mask")]
block_fn = lambda x, y: print("running the inpaint workflow")
block_inpaint_cls = make_block(inputs=inputs, block_fn=block_fn, description="I'm a inpaint workflow!")

class AutoImageBlocks(AutoPipelineBlocks):
    # List of sub-block classes to choose from
    block_classes = [block_inpaint_cls, block_i2i_cls, block_t2i_cls]
    # Names for each block in the same order
    block_names = ["inpaint", "img2img", "text2img"]
    # Trigger inputs that determine which block to run
    # - "mask" triggers inpaint workflow
    # - "image" triggers img2img workflow (but only if mask is not provided) 
    # - if none of above, runs the text2img workflow (default)
    block_trigger_inputs = ["mask", "image", None]
    # Description is extremely important for AutoPipelineBlocks
    @property
    def description(self):
        return (
            "Pipeline generates images given different types of conditions!\n"
            + "This is an auto pipeline block that works for text2img, img2img and inpainting tasks.\n"
            + " - inpaint workflow is run when `mask` is provided.\n"
            + " - img2img workflow is run when `image` is provided (but only when `mask` is not provided).\n"
            + " - text2img workflow is run when neither `image` nor `mask` is provided.\n"
        )

# Create the blocks
auto_blocks = AutoImageBlocks()
# convert to pipeline
auto_pipeline = auto_blocks.init_pipeline()
```

Now we have created an `AutoPipelineBlocks` that contains 3 sub-blocks. Notice the warning message at the top - this automatically appears in every `ModularPipelineBlocks` that contains `AutoPipelineBlocks` to remind end users that dynamic block selection happens at runtime. 

```py
AutoImageBlocks(
  Class: AutoPipelineBlocks

  ====================================================================================================
  This pipeline contains blocks that are selected at runtime based on inputs.
  Trigger Inputs: ['mask', 'image']
  ====================================================================================================


  Description: Pipeline generates images given different types of conditions!
      This is an auto pipeline block that works for text2img, img2img and inpainting tasks.
       - inpaint workflow is run when `mask` is provided.
       - img2img workflow is run when `image` is provided (but only when `mask` is not provided).
       - text2img workflow is run when neither `image` nor `mask` is provided.
      


  Sub-Blocks:
    • inpaint [trigger: mask] (TestBlock)
       Description: I'm a inpaint workflow!

    • img2img [trigger: image] (TestBlock)
       Description: I'm a image-to-image workflow!

    • text2img [default] (TestBlock)
       Description: I'm a text-to-image workflow!

)
```

Check out the documentation with `print(auto_pipeline.doc)`:

```py
>>> print(auto_pipeline.doc)
class AutoImageBlocks

  Pipeline generates images given different types of conditions!
  This is an auto pipeline block that works for text2img, img2img and inpainting tasks.
   - inpaint workflow is run when `mask` is provided.
   - img2img workflow is run when `image` is provided (but only when `mask` is not provided).
   - text2img workflow is run when neither `image` nor `mask` is provided.

  Inputs:

      prompt (`None`, *optional*):

      image (`None`, *optional*):

      mask (`None`, *optional*):
```

There is a fundamental trade-off of AutoPipelineBlocks: it trades clarity for convenience. While it is really easy for packaging multiple workflows, it can become confusing without proper documentation. e.g. if we just throw a pipeline at you and tell you that it contains 3 sub-blocks and takes 3 inputs `prompt`, `image` and `mask`, and ask you to run an image-to-image workflow: if you don't have any prior knowledge on how these pipelines work, you would be pretty clueless, right?

This pipeline we just made though, has a docstring that shows all available inputs and workflows and explains how to use each with different inputs. So it's really helpful for users. For example, it's clear that you need to pass `image` to run img2img. This is why the description field is absolutely critical for AutoPipelineBlocks. We highly recommend you to explain the conditional logic very well for each `AutoPipelineBlocks` you would make. We also recommend to always test individual pipelines first before packaging them into AutoPipelineBlocks. 

Let's run this auto pipeline with different inputs to see if the conditional logic works as described. Remember that we have added `print` in each `PipelineBlock`'s `__call__` method to print out its workflow name, so it should be easy to tell which one is running:

```py
>>> _ = auto_pipeline(image="image", mask="mask")
running the inpaint workflow
>>> _ = auto_pipeline(image="image")
running the image-to-image workflow
>>> _ = auto_pipeline(prompt="prompt")
running the text-to-image workflow
>>> _ = auto_pipeline(image="prompt", mask="mask")
running the inpaint workflow
```

However, even with documentation, it can become very confusing when AutoPipelineBlocks are combined with other blocks. The complexity grows quickly when you have nested AutoPipelineBlocks or use them as sub-blocks in larger pipelines.

Let's make another `AutoPipelineBlocks` - this one only contains one block, and it does not include `None` in its `block_trigger_inputs` (which corresponds to the default block to run when none of the trigger inputs are provided). This means this block will be skipped if the trigger input (`ip_adapter_image`) is not provided at runtime.

```py
from diffusers.modular_pipelines import SequentialPipelineBlocks, InsertableDict
inputs = [InputParam(name="ip_adapter_image")]
block_fn = lambda x, y: print("running the ip-adapter workflow")
block_ipa_cls = make_block(inputs=inputs, block_fn=block_fn, description="I'm a IP-adapter workflow!")

class AutoIPAdapter(AutoPipelineBlocks):
    block_classes = [block_ipa_cls]
    block_names = ["ip-adapter"]
    block_trigger_inputs = ["ip_adapter_image"]
    @property
    def description(self):
        return "Run IP Adapter step if `ip_adapter_image` is provided."
```

Now let's combine these 2 auto blocks together into a `SequentialPipelineBlocks`:

```py
auto_ipa_blocks = AutoIPAdapter()
blocks_dict = InsertableDict()
blocks_dict["ip-adapter"] = auto_ipa_blocks
blocks_dict["image-generation"] = auto_blocks
all_blocks = SequentialPipelineBlocks.from_blocks_dict(blocks_dict)
pipeline = all_blocks.init_pipeline()
```

Let's take a look: now things get more confusing. In this particular example, you could still try to explain the conditional logic in the `description` field here - there are only 4 possible execution paths so it's doable. However, since this is a `SequentialPipelineBlocks` that could contain many more blocks, the complexity can quickly get out of hand as the number of blocks increases.

```py
>>> all_blocks
SequentialPipelineBlocks(
  Class: ModularPipelineBlocks

  ====================================================================================================
  This pipeline contains blocks that are selected at runtime based on inputs.
  Trigger Inputs: ['image', 'mask', 'ip_adapter_image']
  Use `get_execution_blocks()` with input names to see selected blocks (e.g. `get_execution_blocks('image')`).
  ====================================================================================================


  Description: 


  Sub-Blocks:
    [0] ip-adapter (AutoIPAdapter)
       Description: Run IP Adapter step if `ip_adapter_image` is provided.
                   

    [1] image-generation (AutoImageBlocks)
       Description: Pipeline generates images given different types of conditions!
                   This is an auto pipeline block that works for text2img, img2img and inpainting tasks.
                    - inpaint workflow is run when `mask` is provided.
                    - img2img workflow is run when `image` is provided (but only when `mask` is not provided).
                    - text2img workflow is run when neither `image` nor `mask` is provided.
                   

)

```

This is when the `get_execution_blocks()` method comes in handy - it basically extracts a `SequentialPipelineBlocks` that only contains the blocks that are actually run based on your inputs.

Let's try some examples:

`mask`: we expect it to skip the first ip-adapter since `ip_adapter_image` is not provided, and then run the inpaint for the second block.

```py
>>> all_blocks.get_execution_blocks('mask')
SequentialPipelineBlocks(
  Class: ModularPipelineBlocks

  Description: 


  Sub-Blocks:
    [0] image-generation (TestBlock)
       Description: I'm a inpaint workflow!

)
```

Let's also actually run the pipeline to confirm:

```py
>>> _ = pipeline(mask="mask")
skipping auto block: AutoIPAdapter
running the inpaint workflow
```

Try a few more:

```py
print(f"inputs: ip_adapter_image:")
blocks_select = all_blocks.get_execution_blocks('ip_adapter_image')
print(f"expected_execution_blocks: {blocks_select}")
print(f"actual execution blocks:")
_ = pipeline(ip_adapter_image="ip_adapter_image", prompt="prompt")
# expect to see ip-adapter + text2img

print(f"inputs: image:")
blocks_select = all_blocks.get_execution_blocks('image')
print(f"expected_execution_blocks: {blocks_select}")
print(f"actual execution blocks:")
_ = pipeline(image="image", prompt="prompt")
# expect to see img2img

print(f"inputs: prompt:")
blocks_select = all_blocks.get_execution_blocks('prompt')
print(f"expected_execution_blocks: {blocks_select}")
print(f"actual execution blocks:")
_ = pipeline(prompt="prompt")
# expect to see text2img (prompt is not a trigger input so fallback to default)

print(f"inputs: mask + ip_adapter_image:")
blocks_select = all_blocks.get_execution_blocks('mask','ip_adapter_image')
print(f"expected_execution_blocks: {blocks_select}")
print(f"actual execution blocks:")
_ = pipeline(mask="mask", ip_adapter_image="ip_adapter_image")
# expect to see ip-adapter + inpaint
```

In summary, `AutoPipelineBlocks` is a good tool for packaging multiple workflows into a single, convenient interface and it can greatly simplify the user experience. However, always provide clear descriptions explaining the conditional logic, test individual pipelines first before combining them, and use `get_execution_blocks()` to understand runtime behavior in complex compositions.