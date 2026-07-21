<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Instalação

🤗 Diffusers é testado no Python 3.8+ e PyTorch 1.7.0+. Siga as instruções de instalação abaixo para a biblioteca de deep learning que você está utilizando:

- [PyTorch](https://pytorch.org/get-started/locally/) instruções de instalação

## Instalação com pip

Recomenda-se instalar 🤗 Diffusers em um [ambiente virtual](https://docs.python.org/3/library/venv.html).
Se você não está familiarizado com ambiente virtuals, veja o [guia](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).
Um ambiente virtual facilita gerenciar diferentes projetos e evitar problemas de compatibilidade entre dependências.

Comece criando um ambiente virtual no diretório do projeto:

```bash
python -m venv .env
```

Ative o ambiente virtual:

```bash
source .env/bin/activate
```

Recomenda-se a instalação do 🤗 Transformers porque 🤗 Diffusers depende de seus modelos:

<frameworkcontent>
<pt>
```bash
pip install diffusers["torch"] transformers
```
</pt>
</frameworkcontent>

## Instalação a partir do código fonte

Antes da instalação do 🤗 Diffusers a partir do código fonte, certifique-se de ter o PyTorch e o 🤗 Accelerate instalados.

Para instalar o 🤗 Accelerate:

```bash
pip install accelerate
```

então instale o 🤗 Diffusers do código fonte:

```bash
pip install git+https://github.com/huggingface/diffusers
```

Esse comando instala a última versão em desenvolvimento `main` em vez da última versão estável `stable`.
A versão `main` é útil para se manter atualizado com os últimos desenvolvimentos.
Por exemplo, se um bug foi corrigido desde o último lançamento estável, mas um novo lançamento ainda não foi lançado.
No entanto, isso significa que a versão `main` pode não ser sempre estável.
Nós nos esforçamos para manter a versão `main` operacional, e a maioria dos problemas geralmente são resolvidos em algumas horas ou um dia.
Se você encontrar um problema, por favor abra uma [Issue](https://github.com/huggingface/diffusers/issues/new/choose), assim conseguimos arrumar o quanto antes!

## Instalação editável

Você precisará de uma instalação editável se você:

- Usar a versão `main` do código fonte.
- Contribuir para o 🤗 Diffusers e precisa testar mudanças no código.

Clone o repositório e instale o 🤗 Diffusers com os seguintes comandos:

```bash
git clone https://github.com/huggingface/diffusers.git
cd diffusers
```

<frameworkcontent>
<pt>
```bash
pip install -e ".[torch]"
```
</pt>
</frameworkcontent>

Esses comandos irão vincular a pasta que você clonou o repositório e os caminhos das suas bibliotecas Python.
Python então irá procurar dentro da pasta que você clonou além dos caminhos normais das bibliotecas.
Por exemplo, se o pacote python for tipicamente instalado no `~/anaconda3/envs/main/lib/python3.10/site-packages/`, o Python também irá procurar na pasta `~/diffusers/` que você clonou.

> [!WARNING]
> Você deve manter a pasta `diffusers` se quiser continuar usando a biblioteca.

Agora você pode facilmente atualizar seu clone para a última versão do 🤗 Diffusers com o seguinte comando:

```bash
cd ~/diffusers/
git pull
```

Seu ambiente Python vai encontrar a versão `main` do 🤗 Diffusers na próxima execução.

## Cache

Os pesos e os arquivos dos modelos são baixados do Hub para o cache que geralmente é o seu diretório home. Você pode mudar a localização do cache especificando as variáveis de ambiente `HF_HOME` ou `HUGGINFACE_HUB_CACHE` ou configurando o parâmetro `cache_dir` em métodos como [`~DiffusionPipeline.from_pretrained`].

Aquivos em cache permitem que você rode 🤗 Diffusers offline. Para prevenir que o 🤗 Diffusers se conecte à internet, defina a variável de ambiente `HF_HUB_OFFLINE` para `True` e o 🤗 Diffusers irá apenas carregar arquivos previamente baixados em cache.

```shell
export HF_HUB_OFFLINE=True
```

Para mais detalhes de como gerenciar e limpar o cache, olhe o guia de [caching](https://huggingface.co/docs/huggingface_hub/guides/manage-cache).

## Telemetria

Nossa biblioteca coleta informações de telemetria durante as requisições [`~DiffusionPipeline.from_pretrained`].
O dado coletado inclui a versão do 🤗 Diffusers e PyTorch, o modelo ou classe de pipeline requisitado,
e o caminho para um checkpoint pré-treinado se ele estiver hospedado no Hugging Face Hub.
Esse dado de uso nos ajuda a debugar problemas e priorizar novas funcionalidades.
Telemetria é enviada apenas quando é carregado modelos e pipelines do Hub,
e não é coletado se você estiver carregando arquivos locais.

Nos entendemos que nem todo mundo quer compartilhar informações adicionais, e nós respeitamos sua privacidade.
Você pode desabilitar a coleta de telemetria definindo a variável de ambiente `DISABLE_TELEMETRY` do seu terminal:

No Linux/MacOS:

```bash
export DISABLE_TELEMETRY=YES
```

No Windows:

```bash
set DISABLE_TELEMETRY=YES
```
