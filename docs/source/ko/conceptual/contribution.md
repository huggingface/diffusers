<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Diffusers에 기여하는 방법 🧨

오픈 소스 커뮤니티에서의 기여를 환영합니다! 누구나 참여할 수 있으며, 코드뿐만 아니라 질문에 답변하거나 문서를 개선하는 등 모든 유형의 참여가 가치 있고 감사히 여겨집니다. 질문에 답변하고 다른 사람들을 도와주며 소통하고 문서를 개선하는 것은 모두 커뮤니티에게 큰 도움이 됩니다. 따라서 관심이 있다면 두려워하지 말고 참여해보세요!

누구나 우리의 공개 Discord 채널에서 👋 인사하며 시작할 수 있도록 장려합니다. 우리는 diffusion 모델의 최신 동향을 논의하고 질문을 하며 개인 프로젝트를 자랑하고 기여에 대해 서로 도와주거나 그냥 어울리기 위해 모이는 곳입니다☕. <a href="https://Discord.gg/G7tWnz98XR"><img alt="Join us on Discord" src="https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white"></a>

어떤 방식으로든 기여하려는 경우, 우리는 개방적이고 환영하며 친근한 커뮤니티의 일부가 되기 위해 노력하고 있습니다. 우리의 [행동 강령](https://github.com/huggingface/diffusers/blob/main/CODE_OF_CONDUCT.md)을 읽고 상호 작용 중에 이를 존중하도록 주의해주시기 바랍니다. 또한 프로젝트를 안내하는 [윤리 지침](https://huggingface.co/docs/diffusers/conceptual/ethical_guidelines)에 익숙해지고 동일한 투명성과 책임성의 원칙을 준수해주시기를 부탁드립니다.

우리는 커뮤니티로부터의 피드백을 매우 중요하게 생각하므로, 라이브러리를 개선하는 데 도움이 될 가치 있는 피드백이 있다고 생각되면 망설이지 말고 의견을 제시해주세요 - 모든 메시지, 댓글, 이슈, 풀 리퀘스트(PR)는 읽히고 고려됩니다.

## 개요

이슈에 있는 질문에 답변하는 것에서부터 코어 라이브러리에 새로운 diffusion 모델을 추가하는 것까지 다양한 방법으로 기여를 할 수 있습니다.

이어지는 부분에서 우리는 다양한 방법의 기여에 대한 개요를 난이도에 따라 오름차순으로 정리하였습니다. 모든 기여는 커뮤니티에게 가치가 있습니다.

1. [Diffusers 토론 포럼](https://discuss.huggingface.co/c/discussion-related-to-httpsgithubcomhuggingfacediffusers)이나 [Discord](https://discord.gg/G7tWnz98XR)에서 질문에 대답하거나 질문을 할 수 있습니다.
2. [GitHub Issues 탭](https://github.com/huggingface/diffusers/issues/new/choose)에서 새로운 이슈를 열 수 있습니다.
3. [GitHub Issues 탭](https://github.com/huggingface/diffusers/issues)에서 이슈에 대답할 수 있습니다.
4. "Good first issue" 라벨이 지정된 간단한 이슈를 수정할 수 있습니다. [여기](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)를 참조하세요.
5. [문서](https://github.com/huggingface/diffusers/tree/main/docs/source)에 기여할 수 있습니다.
6. [Community Pipeline](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3Acommunity-examples)에 기여할 수 있습니다.
7. [예제](https://github.com/huggingface/diffusers/tree/main/examples)에 기여할 수 있습니다.
8. "Good second issue" 라벨이 지정된 어려운 이슈를 수정할 수 있습니다. [여기](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22Good+second+issue%22)를 참조하세요.
9. 새로운 파이프라인, 모델 또는 스케줄러를 추가할 수 있습니다. ["새로운 파이프라인/모델"](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+pipeline%2Fmodel%22) 및 ["새로운 스케줄러"](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+scheduler%22) 이슈를 참조하세요. 이 기여에 대해서는 [디자인 철학](https://github.com/huggingface/diffusers/blob/main/PHILOSOPHY.md)을 확인해주세요.

앞서 말한 대로, **모든 기여는 커뮤니티에게 가치가 있습니다**. 이어지는 부분에서 각 기여에 대해 조금 더 자세히 설명하겠습니다.

4부터 9까지의 모든 기여에는 PR을 열어야 합니다. [PR을 열기](#how-to-open-a-pr)에서 자세히 설명되어 있습니다.

### 1. Diffusers 토론 포럼이나 Diffusers Discord에서 질문하고 답변하기

Diffusers 라이브러리와 관련된 모든 질문이나 의견은 [토론 포럼](https://discuss.huggingface.co/c/discussion-related-to-httpsgithubcomhuggingfacediffusers/63)이나 [Discord](https://discord.gg/G7tWnz98XR)에서 할 수 있습니다. 이러한 질문과 의견에는 다음과 같은 내용이 포함됩니다(하지만 이에 국한되지는 않습니다):
- 지식을 공유하기 위해서 훈련 또는 추론 실험에 대한 결과 보고
- 개인 프로젝트 소개
- 비공식 훈련 예제에 대한 질문
- 프로젝트 제안
- 일반적인 피드백
- 논문 요약
- Diffusers 라이브러리를 기반으로 하는 개인 프로젝트에 대한 도움 요청
- 일반적인 질문
- Diffusion 모델에 대한 윤리적 질문
- ...

포럼이나 Discord에서 질문을 하면 커뮤니티가 지식을 공개적으로 공유하도록 장려되며, 미래에 동일한 질문을 가진 초보자에게도 도움이 될 수 있습니다. 따라서 궁금한 질문은 언제든지 하시기 바랍니다.
또한, 이러한 질문에 답변하는 것은 커뮤니티에게 매우 큰 도움이 됩니다. 왜냐하면 이렇게 하면 모두가 학습할 수 있는 공개적인 지식을 문서화하기 때문입니다.

**주의**하십시오. 질문이나 답변에 투자하는 노력이 많을수록 공개적으로 문서화된 지식의 품질이 높아집니다. 마찬가지로, 잘 정의되고 잘 답변된 질문은 모두에게 접근 가능한 고품질 지식 데이터베이스를 만들어줍니다. 반면에 잘못된 질문이나 답변은 공개 지식 데이터베이스의 전반적인 품질을 낮출 수 있습니다.
간단히 말해서, 고품질의 질문이나 답변은 *명확하고 간결하며 관련성이 있으며 이해하기 쉽고 접근 가능하며 잘 형식화되어 있어야* 합니다. 자세한 내용은 [좋은 이슈 작성 방법](#how-to-write-a-good-issue) 섹션을 참조하십시오.

**채널에 대한 참고사항**:
[*포럼*](https://discuss.huggingface.co/c/discussion-related-to-httpsgithubcomhuggingfacediffusers/63)은 구글과 같은 검색 엔진에서 더 잘 색인화됩니다. 게시물은 인기에 따라 순위가 매겨지며, 시간순으로 정렬되지 않습니다. 따라서 이전에 게시한 질문과 답변을 쉽게 찾을 수 있습니다.
또한, 포럼에 게시된 질문과 답변은 쉽게 링크할 수 있습니다.
반면 *Discord*는 채팅 형식으로 되어 있어 빠른 대화를 유도합니다.
질문에 대한 답변을 빠르게 받을 수는 있겠지만, 시간이 지나면 질문이 더 이상 보이지 않습니다. 또한, Discord에서 이전에 게시된 정보를 찾는 것은 훨씬 어렵습니다. 따라서 포럼을 사용하여 고품질의 질문과 답변을 하여 커뮤니티를 위한 오래 지속되는 지식을 만들기를 권장합니다. Discord에서의 토론이 매우 흥미로운 답변과 결론을 이끌어내는 경우, 해당 정보를 포럼에 게시하여 미래 독자들에게 더 쉽게 액세스할 수 있도록 권장합니다.

### 2. GitHub 이슈 탭에서 새로운 이슈 열기

🧨 Diffusers 라이브러리는 사용자들이 마주치는 문제를 알려주는 덕분에 견고하고 신뢰할 수 있습니다. 따라서 이슈를 보고해주셔서 감사합니다.

기억해주세요, GitHub 이슈는 Diffusers 라이브러리와 직접적으로 관련된 기술적인 질문, 버그 리포트, 기능 요청 또는 라이브러리 디자인에 대한 피드백에 사용됩니다.

간단히 말해서, Diffusers 라이브러리의 **코드와 관련되지 않은** 모든 것(문서 포함)은 GitHub가 아닌 [포럼](https://discuss.huggingface.co/c/discussion-related-to-httpsgithubcomhuggingfacediffusers/63)이나 [Discord](https://discord.gg/G7tWnz98XR)에서 질문해야 합니다.

**새로운 이슈를 열 때 다음 가이드라인을 고려해주세요**:
- 이미 같은 이슈가 있는지 검색했는지 확인해주세요(GitHub의 이슈 탭에서 검색 기능을 사용하세요).
- 다른(관련된) 이슈에 새로운 이슈를 보고하지 말아주세요. 다른 이슈와 관련이 높다면, 새로운 이슈를 열고 관련 이슈에 링크를 걸어주세요.
- 이슈를 영어로 작성해주세요. 영어에 익숙하지 않다면, [DeepL](https://www.deepl.com/translator)과 같은 뛰어난 무료 온라인 번역 서비스를 사용하여 모국어에서 영어로 번역해주세요.
- 이슈가 최신 Diffusers 버전으로 업데이트하면 해결될 수 있는지 확인해주세요. 이슈를 게시하기 전에 `python -c "import diffusers; print(diffusers.__version__)"` 명령을 실행하여 현재 사용 중인 Diffusers 버전이 최신 버전과 일치하거나 더 높은지 확인해주세요.
- 새로운 이슈를 열 때 투자하는 노력이 많을수록 답변의 품질이 높아지고 Diffusers 이슈 전체의 품질도 향상됩니다.

#### 2.1 재현가능하고 최소한인 버그 리포트

새로운 이슈는 일반적으로 다음과 같은 내용을 포함합니다.

버그 보고서는 항상 재현 가능한 코드 조각을 포함하고 가능한 한 최소한이어야 하며 간결해야 합니다.
자세히 말하면:
- 버그를 가능한 한 좁혀야 합니다. **전체 코드 파일을 그냥 던지지 마세요**.
- 코드의 서식을 지정해야 합니다.
- Diffusers가 의존하는 외부 라이브러리를 제외한 다른 외부 라이브러리는 포함하지 마십시오.
- **반드시** 환경에 대한 모든 필요한 정보를 제공해야 합니다. 이를 위해 쉘에서 `diffusers-cli env`를 실행하고 표시된 정보를 이슈에 복사하여 붙여넣을 수 있습니다.
- 이슈를 설명해야 합니다. 독자가 문제가 무엇이며 왜 문제인지 모르면 해결할 수 없습니다.
- **항상** 독자가 가능한 한 적은 노력으로 문제를 재현할 수 있도록 해야 합니다. 코드 조각이 라이브러리가 없거나 정의되지 않은 변수 때문에 실행되지 않는 경우 독자가 도움을 줄 수 없습니다. 재현 가능한 코드 조각이 가능한 한 최소화되고 간단한 Python 셸에 복사하여 붙여넣을 수 있도록 해야 합니다.
- 문제를 재현하기 위해 모델과/또는 데이터셋이 필요한 경우 독자가 해당 모델이나 데이터셋에 접근할 수 있도록 해야 합니다. 모델이나 데이터셋을 [Hub](https://huggingface.co)에 업로드하여 쉽게 다운로드할 수 있도록 할 수 있습니다. 문제 재현을 가능한 한 쉽게하기 위해 모델과 데이터셋을 가능한 한 작게 유지하려고 노력하세요.

자세한 내용은 [좋은 이슈 작성 방법](#how-to-write-a-good-issue) 섹션을 참조하세요.

버그 보고서를 열려면 [여기](https://github.com/huggingface/diffusers/issues/new?assignees=&labels=bug&projects=&template=bug-report.yml)를 클릭하세요.


#### 2.2. 기능 요청

세계적인 기능 요청은 다음 사항을 다룹니다:

1. 먼저 동기부여:
* 라이브러리와 관련된 문제/불만이 있는가요? 그렇다면 왜 그런지 설명해주세요. 문제를 보여주는 코드 조각을 제공하는 것이 가장 좋습니다.
* 프로젝트에 필요한 기능인가요? 우리는 그에 대해 듣고 싶습니다!
* 커뮤니티에 도움이 될 수 있는 것을 작업했고 그것에 대해 생각하고 있는가요? 멋지네요! 어떤 문제를 해결했는지 알려주세요.
2. 기능을 *상세히 설명하는* 문단을 작성해주세요;
3. 미래 사용을 보여주는 **코드 조각**을 제공해주세요;
4. 이것이 논문과 관련된 경우 링크를 첨부해주세요;
5. 도움이 될 수 있는 추가 정보(그림, 스크린샷 등)를 첨부해주세요.

기능 요청은 [여기](https://github.com/huggingface/diffusers/issues/new?assignees=&labels=&template=feature_request.md&title=)에서 열 수 있습니다.

#### 2.3 피드백

라이브러리 디자인과 그것이 왜 좋은지 또는 나쁜지에 대한 이유에 대한 피드백은 핵심 메인테이너가 사용자 친화적인 라이브러리를 만드는 데 엄청난 도움이 됩니다. 현재 디자인 철학을 이해하려면 [여기](https://huggingface.co/docs/diffusers/conceptual/philosophy)를 참조해 주세요. 특정 디자인 선택이 현재 디자인 철학과 맞지 않는다고 생각되면, 그 이유와 어떻게 변경되어야 하는지 설명해 주세요. 반대로 특정 디자인 선택이 디자인 철학을 너무 따르기 때문에 사용 사례를 제한한다고 생각되면, 그 이유와 어떻게 변경되어야 하는지 설명해 주세요. 특정 디자인 선택이 매우 유용하다고 생각되면, 미래의 디자인 결정에 큰 도움이 되므로 이에 대한 의견을 남겨 주세요.

피드백에 관한 이슈는 [여기](https://github.com/huggingface/diffusers/issues/new?assignees=&labels=&template=feedback.md&title=)에서 열 수 있습니다.

#### 2.4 기술적인 질문

기술적인 질문은 주로 라이브러리의 특정 코드가 왜 특정 방식으로 작성되었는지 또는 코드의 특정 부분이 무엇을 하는지에 대한 질문입니다. 질문하신 코드 부분에 대한 링크를 제공하고 해당 코드 부분이 이해하기 어려운 이유에 대한 자세한 설명을 해주시기 바랍니다.

기술적인 질문에 관한 이슈를 [여기](https://github.com/huggingface/diffusers/issues/new?assignees=&labels=bug&template=bug-report.yml)에서 열 수 있습니다.

#### 2.5 새로운 모델, 스케줄러 또는 파이프라인 추가 제안

만약 diffusion 모델 커뮤니티에서 Diffusers 라이브러리에 추가하고 싶은 새로운 모델, 파이프라인 또는 스케줄러가 있다면, 다음 정보를 제공해주세요:

* Diffusion 파이프라인, 모델 또는 스케줄러에 대한 간단한 설명과 논문 또는 공개된 버전의 링크
* 해당 모델의 오픈 소스 구현에 대한 링크
* 모델 가중치가 있는 경우, 가중치의 링크

모델에 직접 기여하고자 하는 경우, 최선의 안내를 위해 우리에게 알려주세요. 또한, 가능하다면 구성 요소(모델, 스케줄러, 파이프라인 등)의 원래 저자를 GitHub 핸들로 태그하는 것을 잊지 마세요.

모델/파이프라인/스케줄러에 대한 요청을 [여기](https://github.com/huggingface/diffusers/issues/new?assignees=&labels=New+model%2Fpipeline%2Fscheduler&template=new-model-addition.yml)에서 열 수 있습니다.

### 3. GitHub 이슈 탭에서 문제에 대한 답변하기

GitHub에서 이슈에 대한 답변을 하기 위해서는 Diffusers에 대한 기술적인 지식이 필요할 수 있지만, 정확한 답변이 아니더라도 모두가 시도해기를 권장합니다. 이슈에 대한 고품질 답변을 제공하기 위한 몇 가지 팁:
- 가능한 한 간결하고 최소한으로 유지합니다.
- 주제에 집중합니다. 이슈에 대한 답변은 해당 이슈에 관련된 내용에만 집중해야 합니다.
- 코드, 논문 또는 다른 소스를 제공하여 답변을 증명하거나 지지합니다.
- 코드로 답변합니다. 간단한 코드 조각이 이슈에 대한 답변이거나 이슈를 해결하는 방법을 보여준다면, 완전히 재현 가능한 코드 조각을 제공해주세요.

또한, 많은 이슈들은 단순히 주제와 무관하거나 다른 이슈의 중복이거나 관련이 없는 경우가 많습니다. 이러한 이슈들에 대한 답변을 제공하고, 이슈 작성자에게 더 정확한 정보를 제공하거나, 중복된 이슈에 대한 링크를 제공하거나, [포럼](https://discuss.huggingface.co/c/discussion-related-to-httpsgithubcomhuggingfacediffusers/63) 이나 [Discord](https://discord.gg/G7tWnz98XR)로 리디렉션하는 것은 메인테이너에게 큰 도움이 됩니다.

이슈가 올바른 버그 보고서이고 소스 코드에서 수정이 필요하다고 확인한 경우, 다음 섹션을 살펴보세요.

다음 모든 기여에 대해서는 PR을 열여야 합니다. [PR 열기](#how-to-open-a-pr) 섹션에서 자세히 설명되어 있습니다.

### 4. "Good first issue" 고치기

*Good first issues*는 [Good first issue](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22) 라벨로 표시됩니다. 일반적으로, 이슈는 이미 잠재적인 해결책이 어떻게 보이는지 설명하고 있어서 수정하기 쉽습니다.
만약 이슈가 아직 닫히지 않았고 이 문제를 해결해보고 싶다면, "이 이슈를 해결해보고 싶습니다."라는 메시지를 남기면 됩니다. 일반적으로 세 가지 시나리오가 있습니다:
- a.) 이슈 설명이 이미 해결책을 제안합니다. 이 경우, 해결책이 이해되고 합리적으로 보인다면, PR 또는 드래프트 PR을 열어서 수정할 수 있습니다.
- b.) 이슈 설명이 해결책을 제안하지 않습니다. 이 경우, 어떤 해결책이 가능할지 물어볼 수 있고, Diffusers 팀의 누군가가 곧 답변해줄 것입니다. 만약 어떻게 수정할지 좋은 아이디어가 있다면, 직접 PR을 열어도 됩니다.
- c.) 이미 이 문제를 해결하기 위해 열린 PR이 있지만, 이슈가 아직 닫히지 않았습니다. PR이 더 이상 진행되지 않았다면, 새로운 PR을 열고 이전 PR에 링크를 걸면 됩니다. PR은 종종 원래 기여자가 갑자기 시간을 내지 못해 더 이상 진행하지 못하는 경우에 더 이상 진행되지 않게 됩니다. 이는 오픈 소스에서 자주 발생하는 일이며 매우 정상적인 상황입니다. 이 경우, 커뮤니티는 새로 시도하고 기존 PR의 지식을 활용해주면 매우 기쁠 것입니다. 이미 PR이 있고 활성화되어 있다면, 제안을 해주거나 PR을 검토하거나 PR에 기여할 수 있는지 물어보는 등 작성자를 도와줄 수 있습니다.


### 5. 문서에 기여하기

좋은 라이브러리는 항상 좋은 문서를 갖고 있습니다! 공식 문서는 라이브러리를 처음 사용하는 사용자들에게 첫 번째 접점 중 하나이며, 따라서 문서에 기여하는 것은 매우 가치 있는 기여입니다.

라이브러리에 기여하는 방법은 다양합니다:

- 맞춤법이나 문법 오류를 수정합니다.
- 공식 문서가 이상하게 표시되거나 링크가 깨진 경우, 올바르게 수정하는 데 시간을 내주시면 매우 기쁠 것입니다.
- 문서의 입력 또는 출력 텐서의 모양이나 차원을 수정합니다.
- 이해하기 어렵거나 잘못된 문서를 명확하게 합니다.
- 오래된 코드 예제를 업데이트합니다.
- 문서를 다른 언어로 번역합니다.

[공식 Diffusers 문서 페이지](https://huggingface.co/docs/diffusers/index)에 표시된 모든 내용은 공식 문서의 일부이며, 해당 [문서 소스](https://github.com/huggingface/diffusers/tree/main/docs/source)에서 수정할 수 있습니다.

문서에 대한 변경 사항을 로컬에서 확인하는 방법은 [이 페이지](https://github.com/huggingface/diffusers/tree/main/docs)를 참조해주세요.


### 6. 커뮤니티 파이프라인에 기여하기

> [!TIP]
> 커뮤니티 파이프라인에 대해 자세히 알아보려면 [커뮤니티 파이프라인](../using-diffusers/custom_pipeline_overview#community-pipelines) 가이드를 읽어보세요. 커뮤니티 파이프라인이 왜 필요한지 궁금하다면 GitHub 이슈 [#841](https://github.com/huggingface/diffusers/issues/841)를 확인해보세요 (기본적으로, 우리는 diffusion 모델이 추론에 사용될 수 있는 모든 방법을 유지할 수 없지만 커뮤니티가 이를 구축하는 것을 방해하고 싶지 않습니다).

커뮤니티 파이프라인에 기여하는 것은 창의성과 작업을 커뮤니티와 공유하는 좋은 방법입니다. [`DiffusionPipeline`]을 기반으로 빌드하여 `custom_pipeline` 매개변수를 설정함으로써 누구나 로드하고 사용할 수 있도록 할 수 있습니다. 이 섹션에서는 UNet이 단일 순방향 패스만 수행하고 스케줄러를 한 번 호출하는 간단한 파이프라인 (단계별 파이프라인)을 만드는 방법을 안내합니다.

1. 커뮤니티 파이프라인을 위한 one_step_unet.py 파일을 생성하세요. 이 파일은 사용자에 의해 설치되는 패키지를 포함할 수 있지만, [`DiffusionPipeline`]에서 모델 가중치와 스케줄러 구성을 로드하기 위해 하나의 파이프라인 클래스만 있어야 합니다. `__init__` 함수에 UNet과 스케줄러를 추가하세요.

    또한 [`~DiffusionPipeline.save_pretrained`]를 사용하여 파이프라인과 그 구성 요소를 저장할 수 있도록 `register_modules` 함수를 추가해야 합니다.

```py
from diffusers import DiffusionPipeline
import torch

class UnetSchedulerOneForwardPipeline(DiffusionPipeline):
    def __init__(self, unet, scheduler):
        super().__init__()

        self.register_modules(unet=unet, scheduler=scheduler)
```

1. forward 패스에서 (`__call__`로 정의하는 것을 추천합니다), 원하는 어떤 기능이든 추가할 수 있습니다. "one-step" 파이프라인의 경우, 무작위 이미지를 생성하고 `timestep=1`로 설정하여 UNet과 스케줄러를 한 번 호출합니다.

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

이제 UNet과 스케줄러를 전달하여 파이프라인을 실행하거나, 파이프라인 구조가 동일한 경우 사전 학습된 가중치를 로드할 수 있습니다.

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

파이프라인을 GitHub 커뮤니티 파이프라인 또는 Hub 커뮤니티 파이프라인으로 공유할 수 있습니다.

<hfoptions id="pipeline type">
<hfoption id="GitHub pipeline">

GitHub 파이프라인을 공유하려면 Diffusers [저장소](https://github.com/huggingface/diffusers)에서 PR을 열고 one_step_unet.py 파일을 [examples/community](https://github.com/huggingface/diffusers/tree/main/examples/community) 하위 폴더에 추가하세요.

</hfoption>
<hfoption id="Hub pipeline">

Hub 파이프라인을 공유하려면, 허브에 모델 저장소를 생성하고 one_step_unet.py 파일을 업로드하세요.

</hfoption>
</hfoptions>

### 7. 훈련 예제에 기여하기

Diffusers 예제는 [examples](https://github.com/huggingface/diffusers/tree/main/examples) 폴더에 있는 훈련 스크립트의 모음입니다.

두 가지 유형의 훈련 예제를 지원합니다:

- 공식 훈련 예제
- 연구용 훈련 예제

연구용 훈련 예제는 [examples/research_projects](https://github.com/huggingface/diffusers/tree/main/examples/research_projects)에 위치하며, 공식 훈련 예제는 `research_projects` 및 `community` 폴더를 제외한 [examples](https://github.com/huggingface/diffusers/tree/main/examples)의 모든 폴더를 포함합니다.
공식 훈련 예제는 Diffusers의 핵심 메인테이너가 유지 관리하며, 연구용 훈련 예제는 커뮤니티가 유지 관리합니다.
이는 공식 파이프라인 vs 커뮤니티 파이프라인에 대한 [6. 커뮤니티 파이프라인 기여하기](#6-contribute-a-community-pipeline)에서 제시한 이유와 동일합니다: 핵심 메인테이너가 diffusion 모델의 모든 가능한 훈련 방법을 유지 관리하는 것은 현실적으로 불가능합니다.
Diffusers 핵심 메인테잉너와 커뮤니티가 특정 훈련 패러다임을 너무 실험적이거나 충분히 인기 없는 것으로 간주하는 경우, 해당 훈련 코드는 `research_projects` 폴더에 넣고 작성자가 유지 관리해야 합니다.

공식 훈련 및 연구 예제는 하나 이상의 훈련 스크립트, requirements.txt 파일 및 README.md 파일을 포함하는 디렉토리로 구성됩니다. 사용자가 훈련 예제를 사용하려면 리포지토리를 복제해야 합니다:

```bash
git clone https://github.com/huggingface/diffusers
```

그리고 훈련에 필요한 모든 추가적인 의존성도 설치해야 합니다:

```bash
pip install -r /examples/<your-example-folder>/requirements.txt
```

따라서 예제를 추가할 때, `requirements.txt` 파일은 훈련 예제에 필요한 모든 pip 종속성을 정의해야 합니다. 이렇게 설치된 모든 종속성을 사용하여 사용자가 예제의 훈련 스크립트를 실행할 수 있어야 합니다. 예를 들어, [DreamBooth `requirements.txt` 파일](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/requirements.txt)을 참조하세요.

Diffusers 라이브러리의 훈련 예제는 다음 철학을 따라야 합니다:
- 예제를 실행하는 데 필요한 모든 코드는 하나의 Python 파일에 있어야 합니다.
- 사용자는 명령 줄에서 `python <your-example>.py --args`와 같이 예제를 실행할 수 있어야 합니다.
- 예제는 간단하게 유지되어야 하며, Diffusers를 사용한 훈련 방법을 보여주는 **예시**로 사용되어야 합니다. 예제 스크립트의 목적은 최첨단 diffusion 모델을 만드는 것이 아니라, 너무 많은 사용자 정의 로직을 추가하지 않고 이미 알려진 훈련 방법을 재현하는 것입니다. 이 점의 부산물로서, 예제는 좋은 교육 자료로써의 역할을 하기 위해 노력합니다.

예제에 기여하기 위해서는, 이미 존재하는 예제인 [dreambooth](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py)와 같은 예제를 참고하여 어떻게 보여야 하는지에 대한 아이디어를 얻는 것이 매우 권장됩니다.
Diffusers와 긴밀하게 통합되어 있기 때문에, 기여자들이 [Accelerate 라이브러리](https://github.com/huggingface/accelerate)를 사용하는 것을 강력히 권장합니다.
예제 스크립트가 작동하는 경우, 반드시 예제를 정확하게 사용하는 방법을 설명하는 포괄적인 `README.md`를 추가해야 합니다. 이 README에는 다음이 포함되어야 합니다:
- [여기](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth#running-locally-with-pytorch)에 표시된 예제 스크립트를 실행하는 방법에 대한 예제 명령어.
- [여기](https://api.wandb.ai/report/patrickvonplaten/xm6cd5q5)에 표시된 훈련 결과 (로그, 모델 등)에 대한 링크로 사용자가 기대할 수 있는 내용을 보여줍니다.
- 비공식/연구용 훈련 예제를 추가하는 경우, **반드시** git 핸들을 포함하여 이 훈련 예제를 유지 관리할 것임을 명시하는 문장을 추가해야 합니다. [여기](https://github.com/huggingface/diffusers/tree/main/examples/research_projects/intel_opts#diffusers-examples-with-intel-optimizations)에 표시된 것과 같습니다.

만약 공식 훈련 예제에 기여하는 경우, [examples/test_examples.py](https://github.com/huggingface/diffusers/blob/main/examples/test_examples.py)에 테스트를 추가하는 것도 확인해주세요. 비공식 훈련 예제에는 이 작업이 필요하지 않습니다.

### 8. "Good second issue" 고치기

"Good second issue"는 [Good second issue](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22Good+second+issue%22) 라벨로 표시됩니다. Good second issue는 [Good first issues](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)보다 해결하기가 더 복잡합니다.
이슈 설명은 일반적으로 이슈를 해결하는 방법에 대해 덜 구체적이며, 관심 있는 기여자는 라이브러리에 대한 꽤 깊은 이해가 필요합니다.
Good second issue를 해결하고자 하는 경우, 해당 이슈를 해결하기 위해 PR을 열고 PR을 이슈에 링크하세요. 이미 해당 이슈에 대한 PR이 열려있지만 병합되지 않은 경우, 왜 병합되지 않았는지 이해하기 위해 살펴보고 개선된 PR을 열어보세요.
Good second issue는 일반적으로 Good first issue 이슈보다 병합하기가 더 어려우므로, 핵심 메인테이너에게 도움을 요청하는 것이 좋습니다. PR이 거의 완료된 경우, 핵심 메인테이너는 PR에 참여하여 커밋하고 병합을 진행할 수 있습니다.

### 9. 파이프라인, 모델, 스케줄러 추가하기

파이프라인, 모델, 스케줄러는 Diffusers 라이브러리에서 가장 중요한 부분입니다.
이들은 최첨단 diffusion 기술에 쉽게 접근하도록 하며, 따라서 커뮤니티가 강력한 생성형 AI 애플리케이션을 만들 수 있도록 합니다.

새로운 모델, 파이프라인 또는 스케줄러를 추가함으로써, 사용자 인터페이스에 새로운 강력한 사용 사례를 활성화할 수 있으며, 이는 전체 생성형 AI 생태계에 매우 중요한 가치를 제공할 수 있습니다.

Diffusers에는 세 가지 구성 요소에 대한 여러 개발 요청이 있습니다. 특정 구성 요소를 아직 정확히 어떤 것을 추가하고 싶은지 모르는 경우, 다음 링크를 참조하세요:
- [모델 또는 파이프라인](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+pipeline%2Fmodel%22)
- [스케줄러](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+scheduler%22)


세 가지 구성 요소를 추가하기 전에, [철학 가이드](philosophy)를 읽어보는 것을 강력히 권장합니다. 세 가지 구성 요소 중 어느 것을 추가하든, 디자인 철학과 관련된 API 일관성을 유지하기 위해 우리의 디자인 철학과 크게 다른 구성 요소는 병합할 수 없습니다. 디자인 선택에 근본적으로 동의하지 않는 경우, [피드백 이슈](https://github.com/huggingface/diffusers/issues/new?assignees=&labels=&template=feedback.md&title=)를 열어 해당 디자인 패턴/선택이 라이브러리 전체에서 변경되어야 하는지, 디자인 철학을 업데이트해야 하는지에 대해 논의할 수 있습니다. 라이브러리 전체의 일관성은 우리에게 매우 중요합니다.

PR에 원본 코드베이스/논문 링크를 추가하고, 가능하면 PR에서 원래 작성자에게 직접 알림을 보내어 진행 상황을 따라갈 수 있도록 해주세요.

PR에서 막힌 경우나 도움이 필요한 경우, 첫 번째 리뷰나 도움을 요청하는 메시지를 남기는 것을 주저하지 마세요.

#### Copied from mechanism

`# Copied from mechanism` 은 파이프라인, 모델 또는 스케줄러 코드를 추가할 때 이해해야 할 독특하고 중요한 기능입니다. Diffusers 코드베이스 전체에서 이를 자주 볼 수 있는데, 이를 사용하는 이유는 코드베이스를 이해하기 쉽고 유지 관리하기 쉽게 유지하기 위함입니다. `# Copied from mechanism` 으로 표시된 코드는 복사한 코드와 정확히 동일하도록 강제됩니다. 이를 통해 `make fix-copies`를 실행할 때 많은 파일에 걸쳐 변경 사항을 쉽게 업데이트하고 전파할 수 있습니다.

예를 들어, 아래 코드 예제에서 [`~diffusers.pipelines.stable_diffusion.StableDiffusionPipelineOutput`]은 원래 코드이며, `AltDiffusionPipelineOutput`은 `# Copied from mechanism`을 사용하여 복사합니다. 유일한 차이점은 클래스 접두사를 `Stable`에서 `Alt`로 변경한 것입니다.

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

더 자세히 알고 싶다면 [~Don't~ Repeat Yourself*](https://huggingface.co/blog/transformers-design-philosophy#4-machine-learning-models-are-static) 블로그 포스트의 이 섹션을 읽어보세요.

## 좋은 이슈 작성 방법

**이슈를 잘 작성할수록 빠르게 해결될 가능성이 높아집니다.**

1. 이슈에 적절한 템플릿을 사용했는지 확인하세요. [새 이슈를 열 때](https://github.com/huggingface/diffusers/issues/new/choose) 올바른 템플릿을 선택해야 합니다. *버그 보고서*, *기능 요청*, *API 디자인에 대한 피드백*, *새로운 모델/파이프라인/스케줄러 추가*, *포럼*, 또는 빈 이슈 중에서 선택하세요. 이슈를 열 때 올바른 템플릿을 선택하는 것이 중요합니다.
2. **명확성**: 이슈에 적합한 제목을 지정하세요. 이슈 설명을 가능한 간단하게 작성하세요. 이슈를 이해하고 해결하는 데 걸리는 시간을 줄이기 위해 가능한 한 명확하게 작성하세요. 하나의 이슈에 대해 여러 문제를 포함하지 않도록 주의하세요. 여러 문제를 발견한 경우, 각각의 이슈를 개별적으로 열어주세요. 버그인 경우, 어떤 버그인지 가능한 한 정확하게 설명해야 합니다. "diffusers에서 오류"와 같이 간단히 작성하지 마세요.
3. **재현 가능성**: 재현 가능한 코드 조각이 없으면 해결할 수 없습니다. 버그를 발견한 경우, 유지 관리자는 그 버그를 재현할 수 있어야 합니다. 이슈에 재현 가능한 코드 조각을 포함해야 합니다. 코드 조각은 Python 인터프리터에 복사하여 붙여넣을 수 있는 형태여야 합니다. 코드 조각이 작동해야 합니다. 즉, 누락된 import나 이미지에 대한 링크가 없어야 합니다. 이슈에는 오류 메시지와 정확히 동일한 오류 메시지를 재현하기 위해 수정하지 않고 복사하여 붙여넣을 수 있는 코드 조각이 포함되어야 합니다. 이슈에 사용자의 로컬 모델 가중치나 로컬 데이터를 사용하는 경우, 독자가 액세스할 수 없는 경우 이슈를 해결할 수 없습니다. 데이터나 모델을 공유할 수 없는 경우, 더미 모델이나 더미 데이터를 만들어 사용해보세요.
4. **간결성**: 가능한 한 간결하게 유지하여 독자가 문제를 빠르게 이해할 수 있도록 도와주세요. 문제와 관련이 없는 코드나 정보는 모두 제거해주세요. 버그를 발견한 경우, 문제를 설명하는 가장 간단한 코드 예제를 만들어보세요. 버그를 발견한 후에는 작업 흐름 전체를 문제에 던지는 것이 아니라, 에러가 발생하는 훈련 코드의 어느 부분이 문제인지 먼저 이해하고 몇 줄로 재현해보세요. 전체 데이터셋 대신 더미 데이터를 사용해보세요.
5. 링크 추가하기. 특정한 이름, 메서드, 또는 모델을 참조하는 경우, 독자가 더 잘 이해할 수 있도록 링크를 제공해주세요. 특정 PR이나 이슈를 참조하는 경우, 해당 이슈에 링크를 걸어주세요. 독자가 무엇을 말하는지 알고 있다고 가정하지 마세요. 이슈에 링크를 추가할수록 좋습니다.
6. 포맷팅. 파이썬 코드 구문으로 코드를 포맷팅하고, 일반 코드 구문으로 에러 메시지를 포맷팅해주세요. 자세한 내용은 [공식 GitHub 포맷팅 문서](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)를 참조하세요.
7. 이슈를 해결해야 하는 티켓이 아니라, 잘 작성된 백과사전 항목으로 생각해보세요. 추가된 이슈는 공개적으로 사용 가능한 지식에 기여하는 것입니다. 잘 작성된 이슈를 추가함으로써 메인테이너가 문제를 해결하는 데 도움을 주는 것뿐만 아니라, 전체 커뮤니티가 라이브러리의 특정 측면을 더 잘 이해할 수 있도록 도움을 주는 것입니다.

## 좋은 PR 작성 방법

1. 카멜레온이 되세요. 기존의 디자인 패턴과 구문을 이해하고, 코드 추가가 기존 코드베이스에 매끄럽게 흐르도록 해야 합니다. 기존 디자인 패턴이나 사용자 인터페이스와 크게 다른 PR은 병합되지 않습니다.
2. 초점을 맞추세요. 하나의 문제만 해결하는 PR을 작성해야 합니다. "추가하면서 다른 문제도 해결하기"에 빠지지 않도록 주의하세요. 여러 개의 관련 없는 문제를 해결하는 PR을 작성하는 것은 리뷰하기가 훨씬 어렵습니다.
3. 도움이 되는 경우, 추가한 내용이 어떻게 사용되는지 예제 코드 조각을 추가해보세요.
4. PR의 제목은 기여 내용을 요약해야 합니다.
5. PR이 이슈를 해결하는 경우, PR 설명에 이슈 번호를 언급하여 연결되도록 해주세요 (이슈를 참조하는 사람들이 작업 중임을 알 수 있도록).
6. 진행 중인 작업을 나타내려면 제목에 `[WIP]`를 접두사로 붙여주세요. 이는 중복 작업을 피하고, 병합 준비가 된 PR과 구분할 수 있도록 도움이 됩니다.
7. [좋은 이슈를 작성하는 방법](#how-to-write-a-good-issue)에 설명된 대로 텍스트를 구성하고 형식을 지정해보세요.
8. 기존 테스트가 통과하는지 확인하세요
9. 높은 커버리지를 가진 테스트를 추가하세요. 품질 테스트가 없으면 병합할 수 없습니다.
- 새로운 `@slow` 테스트를 추가하는 경우, 다음 명령을 사용하여 통과하는지 확인하세요.
`RUN_SLOW=1 python -m pytest tests/test_my_new_model.py`.
CircleCI는 느린 테스트를 실행하지 않지만, GitHub Actions는 매일 실행합니다!
10. 모든 공개 메서드는 마크다운과 잘 작동하는 정보성 docstring을 가져야 합니다. 예시로 [`pipeline_latent_diffusion.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/latent_diffusion/pipeline_latent_diffusion.py)를 참조하세요.
11. 레포지토리가 빠르게 성장하고 있기 때문에, 레포지토리에 큰 부담을 주는 파일이 추가되지 않도록 주의해야 합니다. 이미지, 비디오 및 기타 텍스트가 아닌 파일을 포함합니다. 이러한 파일을 배치하기 위해 hf.co 호스팅 `dataset`인 [`hf-internal-testing`](https://huggingface.co/hf-internal-testing) 또는 [huggingface/documentation-images](https://huggingface.co/datasets/huggingface/documentation-images)를 활용하는 것이 우선입니다.
외부 기여인 경우, 이미지를 PR에 추가하고 Hugging Face 구성원에게 이미지를 이 데이터셋으로 이동하도록 요청하세요.

## PR을 열기 위한 방법

코드를 작성하기 전에, 이미 누군가가 같은 작업을 하고 있는지 확인하기 위해 기존의 PR이나 이슈를 검색하는 것이 좋습니다. 확실하지 않은 경우, 피드백을 받기 위해 이슈를 열어보는 것이 항상 좋은 아이디어입니다.

🧨 Diffusers에 기여하기 위해서는 기본적인 `git` 사용법을 알아야 합니다. `git`은 가장 쉬운 도구는 아니지만, 가장 훌륭한 매뉴얼을 가지고 있습니다. 셸에서 `git --help`을 입력하고 즐기세요. 책을 선호하는 경우, [Pro Git](https://git-scm.com/book/en/v2)은 매우 좋은 참고 자료입니다.

다음 단계를 따라 기여를 시작하세요 ([지원되는 Python 버전](https://github.com/huggingface/diffusers/blob/main/setup.py#L244)):

1. 저장소 페이지에서 'Fork' 버튼을 클릭하여 [저장소](https://github.com/huggingface/diffusers)를 포크합니다. 이렇게 하면 코드의 사본이 GitHub 사용자 계정에 생성됩니다.

2. 포크한 저장소를 로컬 디스크에 클론하고, 기본 저장소를 원격으로 추가하세요:

 ```bash
 $ git clone git@github.com:<your GitHub handle>/diffusers.git
 $ cd diffusers
 $ git remote add upstream https://github.com/huggingface/diffusers.git
 ```

3. 개발 변경 사항을 보관할 새로운 브랜치를 생성하세요:

 ```bash
 $ git checkout -b a-descriptive-name-for-my-changes
 ```

`main` 브랜치 위에서 **절대** 작업하지 마세요.

1. 가상 환경에서 다음 명령을 실행하여 개발 환경을 설정하세요:

 ```bash
 $ pip install -e ".[dev]"
 ```

만약 저장소를 이미 클론한 경우, 가장 최신 변경 사항을 가져오기 위해 `git pull`을 실행해야 할 수도 있습니다.

5. 기능을 브랜치에서 개발하세요.

기능을 작업하는 동안 테스트 스위트가 통과되는지 확인해야 합니다. 다음과 같이 변경 사항에 영향을 받는 테스트를 실행해야 합니다:

 ```bash
 $ pytest tests/<TEST_TO_RUN>.py
 ```

테스트를 실행하기 전에 테스트를 위해 필요한 의존성들을 설치하였는지 확인하세요. 다음의 커맨드를 통해서 확인할 수 있습니다:

 ```bash
 $ pip install -e ".[test]"
 ```

다음 명령어로 전체 테스트 묶음 실행할 수도 있지만, Diffusers가 많이 성장하였기 때문에 결과를 적당한 시간 내에 생성하기 위해서는 강력한 컴퓨터가 필요합니다. 다음은 해당 명령어입니다:

 ```bash
 $ make test
 ```

🧨 Diffusers는 소스 코드를 일관되게 포맷팅하기 위해 `black`과 `isort`를 사용합니다. 변경 사항을 적용한 후에는 다음과 같이 자동 스타일 수정 및 코드 검증을 적용할 수 있습니다:


 ```bash
 $ make style
 ```

🧨 Diffusers `ruff`와 몇개의 커스텀 스크립트를 이용하여 코딩 실수를 확인합니다. 품질 제어는 CI에서 작동하지만, 동일한 검사를 다음을 통해서도 할 수 있습니다:

 ```bash
 $ make quality
 ```

변경사항에 대해 만족한다면 `git add`를 사용하여 변경된 파일을 추가하고 `git commit`을 사용하여 변경사항에 대해 로컬상으로 저장한다:

 ```bash
 $ git add modified_file.py
 $ git commit -m "A descriptive message about your changes."
 ```

코드를 정기적으로 원본 저장소와 동기화하는 것은 좋은 아이디어입니다. 이렇게 하면 변경 사항을 빠르게 반영할 수 있습니다:

 ```bash
 $ git pull upstream main
 ```

변경 사항을 계정에 푸시하려면 다음을 사용하세요:

 ```bash
 $ git push -u origin a-descriptive-name-for-my-changes
 ```

6. 만족하셨다면, GitHub에서 포크한 웹페이지로 이동하여 'Pull request'를 클릭하여 변경사항을 프로젝트 메인테이너에게 검토를 요청합니다.

7. 메인테이너가 변경 사항을 요청하는 것은 괜찮습니다. 핵심 기여자들에게도 일어나는 일입니다! 따라서 변경 사항을 Pull request에서 볼 수 있도록 로컬 브랜치에서 작업하고 변경 사항을 포크에 푸시하면 자동으로 Pull request에 나타납니다.

### 테스트

라이브러리 동작과 여러 예제를 테스트하기 위해 포괄적인 테스트 묶음이 포함되어 있습니다. 라이브러리 테스트는 [tests 폴더](https://github.com/huggingface/diffusers/tree/main/tests)에서 찾을 수 있습니다.

`pytest`와 `pytest-xdist`를 선호하는 이유는 더 빠르기 때문입니다. 루트 디렉토리에서 라이브러리를 위해 `pytest`로 테스트를 실행하는 방법은 다음과 같습니다:

```bash
$ python -m pytest -n auto --dist=loadfile -s -v ./tests/
```

사실, `make test`는 이렇게 구현되어 있습니다!

작업 중인 기능만 테스트하기 위해 더 작은 테스트 세트를 지정할 수 있습니다.

기본적으로 느린 테스트는 건너뜁니다. `RUN_SLOW` 환경 변수를 `yes`로 설정하여 실행할 수 있습니다. 이는 많은 기가바이트의 모델을 다운로드합니다. 충분한 디스크 공간과 좋은 인터넷 연결 또는 많은 인내심이 필요합니다!

```bash
$ RUN_SLOW=yes python -m pytest -n auto --dist=loadfile -s -v ./tests/
```

`unittest`는 완전히 지원됩니다. 다음은 `unittest`를 사용하여 테스트를 실행하는 방법입니다:

```bash
$ python -m unittest discover -s tests -t . -v
$ python -m unittest discover -s examples -t examples -v
```

### upstream(main)과 forked main 동기화하기

upstream 저장소에 불필요한 참조 노트를 추가하고 관련 개발자에게 알림을 보내는 것을 피하기 위해,
forked 저장소의 main 브랜치를 동기화할 때 다음 단계를 따르세요:
1. 가능한 경우, forked 저장소에서 브랜치와 PR을 사용하여 upstream과 동기화하는 것을 피하세요. 대신 forked main으로 직접 병합하세요.
2. PR이 절대적으로 필요한 경우, 브랜치를 체크아웃한 후 다음 단계를 사용하세요:
```bash
$ git checkout -b your-branch-for-syncing
$ git pull --squash --no-commit upstream main
$ git commit -m '<your message without GitHub references>'
$ git push --set-upstream origin your-branch-for-syncing
```

### 스타일 가이드

Documentation string에 대해서는, 🧨 Diffusers는 [Google 스타일](https://google.github.io/styleguide/pyguide.html)을 따릅니다.
