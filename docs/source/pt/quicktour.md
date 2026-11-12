<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

[[open-in-colab]]

# Tour rápido

Modelos de difusão são treinados para remover o ruído Gaussiano aleatório passo a passo para gerar uma amostra de interesse, como uma imagem ou áudio. Isso despertou um tremendo interesse em IA generativa, e você provavelmente já viu exemplos de imagens geradas por difusão na internet. 🧨 Diffusers é uma biblioteca que visa tornar os modelos de difusão amplamente acessíveis a todos.

Seja você um desenvolvedor ou um usuário, esse tour rápido irá introduzir você ao 🧨 Diffusers e ajudar você a começar a gerar rapidamente! Há três componentes principais da biblioteca para conhecer:

- O [`DiffusionPipeline`] é uma classe de alto nível de ponta a ponta desenhada para gerar rapidamente amostras de modelos de difusão pré-treinados para inferência.
- [Modelos](./api/models) pré-treinados populares e módulos que podem ser usados como blocos de construção para criar sistemas de difusão.
- Vários [Agendadores](./api/schedulers/overview) diferentes - algoritmos que controlam como o ruído é adicionado para treinamento, e como gerar imagens sem o ruído durante a inferência.

Esse tour rápido mostrará como usar o [`DiffusionPipeline`] para inferência, e então mostrará como combinar um modelo e um agendador para replicar o que está acontecendo dentro do [`DiffusionPipeline`].

> [!TIP]
> Esse tour rápido é uma versão simplificada da introdução 🧨 Diffusers [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/diffusers_intro.ipynb) para ajudar você a começar rápido. Se você quer aprender mais sobre o objetivo do 🧨 Diffusers, filosofia de design, e detalhes adicionais sobre a API principal, veja o notebook!

Antes de começar, certifique-se de ter todas as bibliotecas necessárias instaladas:

```py
# uncomment to install the necessary libraries in Colab
#!pip install --upgrade diffusers accelerate transformers
```

- [🤗 Accelerate](https://huggingface.co/docs/accelerate/index) acelera o carregamento do modelo para geração e treinamento.
- [🤗 Transformers](https://huggingface.co/docs/transformers/index) é necessário para executar os modelos mais populares de difusão, como o [Stable Diffusion](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/overview).

## DiffusionPipeline

O [`DiffusionPipeline`] é a forma mais fácil de usar um sistema de difusão pré-treinado para geração. É um sistema de ponta a ponta contendo o modelo e o agendador. Você pode usar o [`DiffusionPipeline`] pronto para muitas tarefas. Dê uma olhada na tabela abaixo para algumas tarefas suportadas, e para uma lista completa de tarefas suportadas, veja a tabela [Resumo do 🧨 Diffusers](./api/pipelines/overview#diffusers-summary).

| **Tarefa**                             | **Descrição**                                                                                                             | **Pipeline**                                                                       |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| Unconditional Image Generation         | gera uma imagem a partir do ruído Gaussiano                                                                               | [unconditional_image_generation](./using-diffusers/unconditional_image_generation) |
| Text-Guided Image Generation           | gera uma imagem a partir de um prompt de texto                                                                            | [conditional_image_generation](./using-diffusers/conditional_image_generation)     |
| Text-Guided Image-to-Image Translation | adapta uma imagem guiada por um prompt de texto                                                                           | [img2img](./using-diffusers/img2img)                                               |
| Text-Guided Image-Inpainting           | preenche a parte da máscara da imagem, dado a imagem, a máscara e o prompt de texto                                       | [inpaint](./using-diffusers/inpaint)                                               |
| Text-Guided Depth-to-Image Translation | adapta as partes de uma imagem guiada por um prompt de texto enquanto preserva a estrutura por estimativa de profundidade | [depth2img](./using-diffusers/depth2img)                                           |

Comece criando uma instância do [`DiffusionPipeline`] e especifique qual checkpoint do pipeline você gostaria de baixar.
Você pode usar o [`DiffusionPipeline`] para qualquer [checkpoint](https://huggingface.co/models?library=diffusers&sort=downloads) armazenado no Hugging Face Hub.
Nesse quicktour, você carregará o checkpoint [`stable-diffusion-v1-5`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) para geração de texto para imagem.

> [!WARNING]
> Para os modelos de [Stable Diffusion](https://huggingface.co/CompVis/stable-diffusion), por favor leia cuidadosamente a [licença](https://huggingface.co/spaces/CompVis/stable-diffusion-license) primeiro antes de rodar o modelo. 🧨 Diffusers implementa uma verificação de segurança: [`safety_checker`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/safety_checker.py) para prevenir conteúdo ofensivo ou nocivo, mas as capacidades de geração de imagem aprimorada do modelo podem ainda produzir conteúdo potencialmente nocivo.

Para carregar o modelo com o método [`~DiffusionPipeline.from_pretrained`]:

```python
>>> from diffusers import DiffusionPipeline

>>> pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", use_safetensors=True)
```

O [`DiffusionPipeline`] baixa e armazena em cache todos os componentes de modelagem, tokenização, e agendamento. Você verá que o pipeline do Stable Diffusion é composto pelo [`UNet2DConditionModel`] e [`PNDMScheduler`] entre outras coisas:

```py
>>> pipeline
StableDiffusionPipeline {
  "_class_name": "StableDiffusionPipeline",
  "_diffusers_version": "0.13.1",
  ...,
  "scheduler": [
    "diffusers",
    "PNDMScheduler"
  ],
  ...,
  "unet": [
    "diffusers",
    "UNet2DConditionModel"
  ],
  "vae": [
    "diffusers",
    "AutoencoderKL"
  ]
}
```

Nós fortemente recomendamos rodar o pipeline em uma placa de vídeo, pois o modelo consiste em aproximadamente 1.4 bilhões de parâmetros.
Você pode mover o objeto gerador para uma placa de vídeo, assim como você faria no PyTorch:

```python
>>> pipeline.to("cuda")
```

Agora você pode passar o prompt de texto para o `pipeline` para gerar uma imagem, e então acessar a imagem sem ruído. Por padrão, a saída da imagem é embrulhada em um objeto [`PIL.Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html?highlight=image#the-image-class).

```python
>>> image = pipeline("An image of a squirrel in Picasso style").images[0]
>>> image
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/image_of_squirrel_painting.png"/>
</div>

Salve a imagem chamando o `save`:

```python
>>> image.save("image_of_squirrel_painting.png")
```

### Pipeline local

Você também pode utilizar o pipeline localmente. A única diferença é que você precisa baixar os pesos primeiro:

```bash
!git lfs install
!git clone https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
```

Assim carregue os pesos salvos no pipeline:

```python
>>> pipeline = DiffusionPipeline.from_pretrained("./stable-diffusion-v1-5", use_safetensors=True)
```

Agora você pode rodar o pipeline como você faria na seção acima.

### Troca dos agendadores

Agendadores diferentes tem diferentes velocidades de retirar o ruído e compensações de qualidade. A melhor forma de descobrir qual funciona melhor para você é testar eles! Uma das principais características do 🧨 Diffusers é permitir que você troque facilmente entre agendadores. Por exemplo, para substituir o [`PNDMScheduler`] padrão com o [`EulerDiscreteScheduler`], carregue ele com o método [`~diffusers.ConfigMixin.from_config`]:

```py
>>> from diffusers import EulerDiscreteScheduler

>>> pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", use_safetensors=True)
>>> pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
```

Tente gerar uma imagem com o novo agendador e veja se você nota alguma diferença!

Na próxima seção, você irá dar uma olhada mais de perto nos componentes - o modelo e o agendador - que compõe o [`DiffusionPipeline`] e aprender como usar esses componentes para gerar uma imagem de um gato.

## Modelos

A maioria dos modelos recebe uma amostra de ruído, e em cada _timestep_ ele prevê o _noise residual_ (outros modelos aprendem a prever a amostra anterior diretamente ou a velocidade ou [`v-prediction`](https://github.com/huggingface/diffusers/blob/5e5ce13e2f89ac45a0066cb3f369462a3cf1d9ef/src/diffusers/schedulers/scheduling_ddim.py#L110)), a diferença entre uma imagem menos com ruído e a imagem de entrada. Você pode misturar e combinar modelos para criar outros sistemas de difusão.

Modelos são inicializados com o método [`~ModelMixin.from_pretrained`] que também armazena em cache localmente os pesos do modelo para que seja mais rápido na próxima vez que você carregar o modelo. Para o tour rápido, você irá carregar o [`UNet2DModel`], um modelo básico de geração de imagem incondicional com um checkpoint treinado em imagens de gato:

```py
>>> from diffusers import UNet2DModel

>>> repo_id = "google/ddpm-cat-256"
>>> model = UNet2DModel.from_pretrained(repo_id, use_safetensors=True)
```

Para acessar os parâmetros do modelo, chame `model.config`:

```py
>>> model.config
```

A configuração do modelo é um dicionário 🧊 congelado 🧊, o que significa que esses parâmetros não podem ser mudados depois que o modelo é criado. Isso é intencional e garante que os parâmetros usados para definir a arquitetura do modelo no início permaneçam os mesmos, enquanto outros parâmetros ainda podem ser ajustados durante a geração.

Um dos parâmetros mais importantes são:

- `sample_size`: a dimensão da altura e largura da amostra de entrada.
- `in_channels`: o número de canais de entrada da amostra de entrada.
- `down_block_types` e `up_block_types`: o tipo de blocos de downsampling e upsampling usados para criar a arquitetura UNet.
- `block_out_channels`: o número de canais de saída dos blocos de downsampling; também utilizado como uma order reversa do número de canais de entrada dos blocos de upsampling.
- `layers_per_block`: o número de blocks ResNet presentes em cada block UNet.

Para usar o modelo para geração, crie a forma da imagem com ruído Gaussiano aleatório. Deve ter um eixo `batch` porque o modelo pode receber múltiplos ruídos aleatórios, um eixo `channel` correspondente ao número de canais de entrada, e um eixo `sample_size` para a altura e largura da imagem:

```py
>>> import torch

>>> torch.manual_seed(0)

>>> noisy_sample = torch.randn(1, model.config.in_channels, model.config.sample_size, model.config.sample_size)
>>> noisy_sample.shape
torch.Size([1, 3, 256, 256])
```

Para geração, passe a imagem com ruído para o modelo e um `timestep`. O `timestep` indica o quão ruidosa a imagem de entrada é, com mais ruído no início e menos no final. Isso ajuda o modelo a determinar sua posição no processo de difusão, se está mais perto do início ou do final. Use o método `sample` para obter a saída do modelo:

```py
>>> with torch.no_grad():
...     noisy_residual = model(sample=noisy_sample, timestep=2).sample
```

Para geração de exemplos reais, você precisará de um agendador para guiar o processo de retirada do ruído. Na próxima seção, você irá aprender como acoplar um modelo com um agendador.

## Agendadores

Agendadores gerenciam a retirada do ruído de uma amostra ruidosa para uma amostra menos ruidosa dado a saída do modelo - nesse caso, é o `noisy_residual`.

> [!TIP]
> 🧨 Diffusers é uma caixa de ferramentas para construir sistemas de difusão. Enquanto o [`DiffusionPipeline`] é uma forma conveniente de começar com um sistema de difusão pré-construído, você também pode escolher seus próprios modelos e agendadores separadamente para construir um sistema de difusão personalizado.

Para o tour rápido, você irá instanciar o [`DDPMScheduler`] com o método [`~diffusers.ConfigMixin.from_config`]:

```py
>>> from diffusers import DDPMScheduler

>>> scheduler = DDPMScheduler.from_config(repo_id)
>>> scheduler
DDPMScheduler {
  "_class_name": "DDPMScheduler",
  "_diffusers_version": "0.13.1",
  "beta_end": 0.02,
  "beta_schedule": "linear",
  "beta_start": 0.0001,
  "clip_sample": true,
  "clip_sample_range": 1.0,
  "num_train_timesteps": 1000,
  "prediction_type": "epsilon",
  "trained_betas": null,
  "variance_type": "fixed_small"
}
```

> [!TIP]
> 💡 Perceba como o agendador é instanciado de uma configuração. Diferentemente de um modelo, um agendador não tem pesos treináveis e é livre de parâmetros!

Um dos parâmetros mais importante são:

- `num_train_timesteps`: o tamanho do processo de retirar ruído ou em outras palavras, o número de _timesteps_ necessários para o processo de ruídos Gausianos aleatórios dentro de uma amostra de dados.
- `beta_schedule`: o tipo de agendados de ruído para o uso de geração e treinamento.
- `beta_start` e `beta_end`: para começar e terminar os valores de ruído para o agendador de ruído.

Para predizer uma imagem com um pouco menos de ruído, passe o seguinte para o método do agendador [`~diffusers.DDPMScheduler.step`]: saída do modelo, `timestep`, e a atual `amostra`.

```py
>>> less_noisy_sample = scheduler.step(model_output=noisy_residual, timestep=2, sample=noisy_sample).prev_sample
>>> less_noisy_sample.shape
```

O `less_noisy_sample` pode ser passado para o próximo `timestep` onde ele ficará ainda com menos ruído! Vamos juntar tudo agora e visualizar o processo inteiro de retirada de ruído.

Comece, criando a função que faça o pós-processamento e mostre a imagem sem ruído como uma `PIL.Image`:

```py
>>> import PIL.Image
>>> import numpy as np


>>> def display_sample(sample, i):
...     image_processed = sample.cpu().permute(0, 2, 3, 1)
...     image_processed = (image_processed + 1.0) * 127.5
...     image_processed = image_processed.numpy().astype(np.uint8)

...     image_pil = PIL.Image.fromarray(image_processed[0])
...     display(f"Image at step {i}")
...     display(image_pil)
```

Para acelerar o processo de retirada de ruído, mova a entrada e o modelo para uma GPU:

```py
>>> model.to("cuda")
>>> noisy_sample = noisy_sample.to("cuda")
```

Agora, crie um loop de retirada de ruído que prediz o residual da amostra menos ruidosa, e computa a amostra menos ruidosa com o agendador:

```py
>>> import tqdm

>>> sample = noisy_sample

>>> for i, t in enumerate(tqdm.tqdm(scheduler.timesteps)):
...     # 1. predict noise residual
...     with torch.no_grad():
...         residual = model(sample, t).sample

...     # 2. compute less noisy image and set x_t -> x_t-1
...     sample = scheduler.step(residual, t, sample).prev_sample

...     # 3. optionally look at image
...     if (i + 1) % 50 == 0:
...         display_sample(sample, i + 1)
```

Sente-se e assista o gato ser gerado do nada além de ruído! 😻

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/diffusion-quicktour.png"/>
</div>

## Próximos passos

Esperamos que você tenha gerado algumas imagens legais com o 🧨 Diffusers neste tour rápido! Para suas próximas etapas, você pode

- Treine ou faça a configuração fina de um modelo para gerar suas próprias imagens no tutorial de [treinamento](./tutorials/basic_training).
- Veja exemplos oficiais e da comunidade de [scripts de treinamento ou configuração fina](https://github.com/huggingface/diffusers/tree/main/examples#-diffusers-examples) para os mais variados casos de uso.
- Aprenda sobre como carregar, acessar, mudar e comparar agendadores no guia [Usando diferentes agendadores](./using-diffusers/schedulers).
- Explore engenharia de prompt, otimizações de velocidade e memória, e dicas e truques para gerar imagens de maior qualidade com o guia [Stable Diffusion](./stable_diffusion).
- Se aprofunde em acelerar 🧨 Diffusers com guias sobre [PyTorch otimizado em uma GPU](./optimization/fp16), e guias de inferência para rodar [Stable Diffusion em Apple Silicon (M1/M2)](./optimization/mps).
