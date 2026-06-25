<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/huggingface/diffusers/77aadfee6a891ab9fcfb780f87c693f7a5beeb8e/docs/source/imgs/diffusers_library.jpg" width="400" style="border: none;"/>
    <br>
</p>

# Diffusers

Diffusers es una librería de modelos de difusión preentrenados de última generación para generar vídeos, imágenes y audio.

La librería gira en torno al [`DiffusionPipeline`], una API diseñada para:

- una inferencia sencilla con solo unas pocas líneas de código
- la flexibilidad de combinar componentes del pipeline (modelos, schedulers)
- cargar y usar adaptadores como LoRA

Diffusers también incluye optimizaciones, como el offloading y la cuantización, para garantizar que incluso los modelos más grandes sean accesibles en dispositivos con memoria limitada. Si la memoria no es un problema, Diffusers admite torch.compile para acelerar la inferencia.

¡Empieza ahora mismo con un modelo de Diffusers en el [Hub](https://huggingface.co/models?library=diffusers&sort=trending)!

## Aprende

Si eres principiante, te recomendamos empezar con el [curso de modelos de difusión de Hugging Face](https://huggingface.co/learn/diffusion-course/unit0/1). Aprenderás la teoría que hay detrás de los modelos de difusión, y aprenderás a usar la librería Diffusers para generar imágenes, ajustar tus propios modelos y mucho más.
