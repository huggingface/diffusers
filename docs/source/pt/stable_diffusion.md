<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

[[open-in-colab]]

# Desempenho básico

Difusão é um processo aleatório que demanda muito processamento. Você pode precisar executar o [`DiffusionPipeline`] várias vezes antes de obter o resultado desejado. Por isso é importante equilibrar cuidadosamente a velocidade de geração e o uso de memória para iterar mais rápido.

Este guia recomenda algumas dicas básicas de desempenho para usar o [`DiffusionPipeline`]. Consulte a seção de documentação sobre Otimização de Inferência, como [Acelerar inferência](./optimization/fp16) ou [Reduzir uso de memória](./optimization/memory) para guias de desempenho mais detalhados.

## Uso de memória

Reduzir a quantidade de memória usada indiretamente acelera a geração e pode ajudar um modelo a caber no dispositivo.

O método [`~DiffusionPipeline.enable_model_cpu_offload`] move um modelo para a CPU quando não está em uso para economizar memória da GPU.

```py
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
  "stabilityai/stable-diffusion-xl-base-1.0",
  torch_dtype=torch.bfloat16,
  device_map="cuda"
)
pipeline.enable_model_cpu_offload()

prompt = """
cinematic film still of a cat sipping a margarita in a pool in Palm Springs, California
highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain
"""
pipeline(prompt).images[0]
print(f"Memória máxima reservada: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
```

## Velocidade de inferência

O processo de remoção de ruído é o mais exigente computacionalmente durante a difusão. Métodos que otimizam este processo aceleram a velocidade de inferência. Experimente os seguintes métodos para acelerar.

- Adicione `device_map="cuda"` para colocar o pipeline em uma GPU. Colocar um modelo em um acelerador, como uma GPU, aumenta a velocidade porque realiza computações em paralelo.
- Defina `torch_dtype=torch.bfloat16` para executar o pipeline em meia-precisão. Reduzir a precisão do tipo de dado aumenta a velocidade porque leva menos tempo para realizar computações em precisão mais baixa.

```py
import torch
import time
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

pipeline = DiffusionPipeline.from_pretrained(
  "stabilityai/stable-diffusion-xl-base-1.0",
  torch_dtype=torch.bfloat16,
  device_map="cuda"
)
```

- Use um agendador mais rápido, como [`DPMSolverMultistepScheduler`], que requer apenas ~20-25 passos.
- Defina `num_inference_steps` para um valor menor. Reduzir o número de passos de inferência reduz o número total de computações. No entanto, isso pode resultar em menor qualidade de geração.

```py
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

prompt = """
cinematic film still of a cat sipping a margarita in a pool in Palm Springs, California
highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain
"""

start_time = time.perf_counter()
image = pipeline(prompt).images[0]
end_time = time.perf_counter()

print(f"Geração de imagem levou {end_time - start_time:.3f} segundos")
```

## Qualidade de geração

Muitos modelos de difusão modernos entregam imagens de alta qualidade imediatamente. No entanto, você ainda pode melhorar a qualidade de geração experimentando o seguinte.

- Experimente um prompt mais detalhado e descritivo. Inclua detalhes como o meio da imagem, assunto, estilo e estética. Um prompt negativo também pode ajudar, guiando um modelo para longe de características indesejáveis usando palavras como baixa qualidade ou desfocado.

    ```py
    import torch
    from diffusers import DiffusionPipeline

    pipeline = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )

    prompt = """
    cinematic film still of a cat sipping a margarita in a pool in Palm Springs, California
    highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain
    """
    negative_prompt = "low quality, blurry, ugly, poor details"
    pipeline(prompt, negative_prompt=negative_prompt).images[0]
    ```

    Para mais detalhes sobre como criar prompts melhores, consulte a documentação sobre [Técnicas de prompt](./using-diffusers/weighted_prompts).

- Experimente um agendador diferente, como [`HeunDiscreteScheduler`] ou [`LMSDiscreteScheduler`], que sacrifica velocidade de geração por qualidade.

    ```py
    import torch
    from diffusers import DiffusionPipeline, HeunDiscreteScheduler

    pipeline = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )
    pipeline.scheduler = HeunDiscreteScheduler.from_config(pipeline.scheduler.config)

    prompt = """
    cinematic film still of a cat sipping a margarita in a pool in Palm Springs, California
    highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain
    """
    negative_prompt = "low quality, blurry, ugly, poor details"
    pipeline(prompt, negative_prompt=negative_prompt).images[0]
    ```

## Próximos passos

Diffusers oferece otimizações mais avançadas e poderosas, como [group-offloading](./optimization/memory#group-offloading) e [compilação regional](./optimization/fp16#regional-compilation). Para saber mais sobre como maximizar o desempenho, consulte a seção sobre Otimização de Inferência.
