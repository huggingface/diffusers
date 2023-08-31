# 여러 GPU를 사용한 분산 추론

분산 설정에서는 여러 개의 프롬프트를 동시에 생성할 때 유용한 🤗 [Accelerate](https://huggingface.co/docs/accelerate/index) 또는 [PyTorch Distributed](https://pytorch.org/tutorials/beginner/dist_overview.html)를 사용하여 여러 GPU에서 추론을 실행할 수 있습니다.

이 가이드에서는 분산 추론을 위해 🤗 Accelerate와 PyTorch Distributed를 사용하는 방법을 보여드립니다.

## 🤗 Accelerate

🤗 [Accelerate](https://huggingface.co/docs/accelerate/index)는 분산 설정에서 추론을 쉽게 훈련하거나 실행할 수 있도록 설계된 라이브러리입니다. 분산 환경 설정 프로세스를 간소화하여 PyTorch 코드에 집중할 수 있도록 해줍니다.

시작하려면 Python 파일을 생성하고 [`accelerate.PartialState`]를 초기화하여 분산 환경을 생성하면, 설정이 자동으로 감지되므로 `rank` 또는 `world_size`를 명시적으로 정의할 필요가 없습니다. ['DiffusionPipeline`]을 `distributed_state.device`로 이동하여 각 프로세스에 GPU를 할당합니다.

이제 컨텍스트 관리자로 [`~accelerate.PartialState.split_between_processes`] 유틸리티를 사용하여 프로세스 수에 따라 프롬프트를 자동으로 분배합니다.


```py
from accelerate import PartialState
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
distributed_state = PartialState()
pipeline.to(distributed_state.device)

with distributed_state.split_between_processes(["a dog", "a cat"]) as prompt:
    result = pipeline(prompt).images[0]
    result.save(f"result_{distributed_state.process_index}.png")
```

Use the `--num_processes` argument to specify the number of GPUs to use, and call `accelerate launch` to run the script:

```bash
accelerate launch run_distributed.py --num_processes=2
```

<Tip>자세한 내용은 [🤗 Accelerate를 사용한 분산 추론](https://huggingface.co/docs/accelerate/en/usage_guides/distributed_inference#distributed-inference-with-accelerate) 가이드를 참조하세요.

</Tip>

## Pytoerch 분산

PyTorch는 데이터 병렬 처리를 가능하게 하는 [`DistributedDataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)을 지원합니다.

시작하려면 Python 파일을 생성하고 `torch.distributed` 및 `torch.multiprocessing`을 임포트하여 분산 프로세스 그룹을 설정하고 각 GPU에서 추론용 프로세스를 생성합니다. 그리고 [`DiffusionPipeline`]도 초기화해야 합니다:

확산 파이프라인을 `rank`로 이동하고 `get_rank`를 사용하여 각 프로세스에 GPU를 할당하면 각 프로세스가 다른 프롬프트를 처리합니다:

```py
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from diffusers import DiffusionPipeline

sd = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
```

사용할 백엔드 유형, 현재 프로세스의 `rank`, `world_size` 또는 참여하는 프로세스 수로 분산 환경 생성을 처리하는 함수[`init_process_group`]를 만들어 추론을 실행해야 합니다.

2개의 GPU에서 추론을 병렬로 실행하는 경우 `world_size`는 2입니다.

```py
def run_inference(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    sd.to(rank)

    if torch.distributed.get_rank() == 0:
        prompt = "a dog"
    elif torch.distributed.get_rank() == 1:
        prompt = "a cat"

    image = sd(prompt).images[0]
    image.save(f"./{'_'.join(prompt)}.png")
```

분산 추론을 실행하려면 [`mp.spawn`](https://pytorch.org/docs/stable/multiprocessing.html#torch.multiprocessing.spawn)을 호출하여 `world_size`에 정의된 GPU 수에 대해 `run_inference` 함수를 실행합니다:

```py
def main():
    world_size = 2
    mp.spawn(run_inference, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
```

추론 스크립트를 완료했으면 `--nproc_per_node` 인수를 사용하여 사용할 GPU 수를 지정하고 `torchrun`을 호출하여 스크립트를 실행합니다:

```bash
torchrun run_distributed.py --nproc_per_node=2
```