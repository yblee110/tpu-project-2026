# GPT는 토큰을 하나씩 씁니다. 이 모델은 한 블록을 한 번에 씁니다.

## Flax NNX와 TPU로 Diffusion Language Model 만들기: 셜록 홈즈 편

GPT는 수년간 텍스트 생성의 절대적인 왕이었습니다. 하지만 텍스트를 생성하는 근본적으로 더 나은 방법이 있다면 어떨까요? 왼쪽에서 오른쪽으로 한 토큰씩이 아니라, **전체 블록을 병렬로** 생성하는 방법 — 마치 노이즈가 결정화되어 완성된 문장이 되는 것을 지켜보는 것처럼요.

바로 이것이 **Diffusion Language Model(DLLM)**이 하는 일입니다. Stable Diffusion 같은 이미지 디퓨전 모델에서 영감을 받아, DLLM은 마스킹("노이즈")된 토큰 시퀀스에서 시작해 반복적으로 디노이징하여 깔끔한 텍스트를 만들어냅니다. 그리고 놀라운 점은? GPT를 Diffusion 모델로 바꾸는 데 **단 5가지 작은 변경**만 필요하다는 것입니다.

이 글에서는 **셜록 홈즈 전집**으로 학습한 character-level DLLM을 **Flax NNX** — Flax의 차세대 뉴럴 네트워크 API — 로 구현하고, **Google Cloud TPU**에서 실행한 과정을 정리합니다. 전체 구현은 하나의 Jupyter Notebook에 들어가며, 표준 GPT를 Diffusion 모델로 바꾸는 데 얼마나 적은 코드 변경이 필요한지 보여줍니다.

모든 코드는 [yblee/tpu-project-2026](https://github.com/yblee/tpu-project-2026) 저장소에서 확인할 수 있으며, 원본 [tiny-diffusion](https://github.com/nathan-barry/tiny-diffusion)에 크게 영감받아 JAX/Flax NNX로 확장한 프로젝트입니다.

---

## Diffusion Language Model이란?

Stable Diffusion이나 Midjourney를 사용해보셨다면 핵심 아이디어를 이미 아시는 겁니다. 이미지 디퓨전 모델은 순수 노이즈에서 시작해 반복적으로 디노이징하여 완성된 이미지를 만들어냅니다. **DLLM도 동일한 원리를 이산 텍스트 시퀀스에 적용합니다.**

이전 토큰들을 보고 다음 토큰을 예측하는 대신, DLLM은 일부 토큰이 특수 `MASK` 토큰으로 교체된 시퀀스를 입력받아 원래 토큰을 복원하는 방법을 학습합니다.

### DLLM vs. GPT 한눈에 비교

| 구분 | GPT (Autoregressive) | DLLM (Diffusion) |
| :--- | :--- | :--- |
| **생성 방식** | 순차적 (한 토큰씩) | 병렬 (블록 단위) |
| **어텐션** | Causal (이전 토큰만 참조) | Bidirectional (모든 토큰 참조) |
| **학습 목표** | 다음 토큰 예측 | 마스킹된 토큰 복원 (디노이징) |
| **컨텍스트** | 단방향 | 전방향 / 양방향 |

핵심 인사이트는 양방향 어텐션입니다. 생성 과정에서 모델이 전체 시퀀스를 볼 수 있어 — 이미 디코딩된 토큰과 여전히 마스킹된 위치 모두를 — 각 위치에 대해 더 나은 예측을 동시에 할 수 있습니다.

---

## 학습 데이터: 셜록 홈즈 전집

접근성과 재미를 살리기 위해 Arthur Conan Doyle의 **셜록 홈즈 전집**으로 모델을 학습시켰습니다. 데이터는 Kaggle의 [Sherlock Books](https://www.kaggle.com/datasets/talesgomes27/sherleck-books) 데이터셋을 사용했으며, *주홍색 연구*부터 *셜록 홈즈의 사건집*까지 소설 4편과 단편 56편이 모두 포함되어 있습니다.

왜 하필 셜록 홈즈일까요?

- **풍부하고 일관된 문체** — Doyle의 빅토리아 시대 영어는 독특한 리듬과 어휘를 가지고 있어, 생성 품질을 직관적으로 평가하기 좋습니다
- **실험에 적합한 크기** — 전체 코퍼스가 유의미한 패턴을 학습할 만큼 크면서도, 빠르게 반복 실헡할 수 있을 만큼 작습니다 (테라바이트 단위의 저장소나 멀티 GPU 환경 불필요)
- **Character-level 모델링** — 문자 수준에서 모델은 빅토리아 시대 단어의 스펠링, 대화 포맷팅(`"Holmes said, "`), 그리고 `"elementary"`나 `"the game is afoot"` 같은 반복 구문까지 학습합니다
- **퍼블릭 도메인** — Project Gutenberg을 통해 모든 작품이 자유롭게 이용 가능합니다

데이터 파이프라인은 Project Gutenberg의 헤더/푸터를 제거하고, 모든 문자를 정수 매핑으로 인코딩합니다. Diffusion 학습을 위한 마스크 토큰으로 특수 `_` 토큰이 어휘에 추가됩니다.

---

## GPT를 Diffusion 모델로 바꾸는 5가지 변경점

이 프로젝트에서 가장 흥미로운 점 중 하나는 **얼마나 적은 것**이 바뀌어야 하는가입니다. `gpt.py`와 `diffusion.py` 사이의 코드는 약 80%가 동일합니다. 다음은 5가지 정밀한 수정 사항입니다.

**1. 어휘에 마스크 토큰 추가**

```python
chars = sorted(list(set(text)))
chars = ["_"] + chars  # 마스크 토큰 추가
mask_token_id = stoi["_"]
```

밑줄 `_`이 "노이즈" 토큰 역할을 하며, 예측해야 할 위치를 나타냅니다.

**2. Causal 어텐션을 양방향 어텐션으로 전환**

```python
# GPT:  y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
# DLLM: y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
```

Causal 마스크를 제거하면 모든 위치가 다른 모든 위치에 attend할 수 있어, 마스킹된 토큰을 공동으로 예측하는 데 필수적입니다.

**3. 학습 목표를 다음 토큰 예측에서 마스킹 복원으로 변경**

```python
# GPT:  y = x를 1만큼 시프트 (다음 토큰 예측)
# DLLM: y = 원래 x (마스킹된 입력에서 복원)

x_orig = jnp.stack([dataset[i : i + block_size] for i in ix])
mask = jax.random.uniform(rng, ...) < mask_probs
x = jnp.where(mask, mask_token_id, x_orig)  # 일부 토큰을 마스크로 교체
```

학습 시 각 샘플은 무작위 마스킹 비율을 가집니다. 모델은 손상된 `x`를 보고 `x_orig`를 복원하려고 시도합니다.

**4. 마스킹된 토큰만 Loss에 기여**

```python
loss_all = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
loss = jnp.sum(loss_all * mask) / (jnp.sum(mask) + 1e-6)
```

예측을 요청받지 않은 위치에 대해서는 모델을 페널티하지 않습니다. 오직 마스킹된 위치만 Loss에 반영됩니다.

**5. 순차적 디코딩을 Confidence 기반 병렬 디코딩으로 교체**

이것이 가장 중요한 변경이며, 별도의 섹션에서 다룹니다.

---

## 병렬 디코딩: DLLM이 텍스트를 생성하는 방식

GPT의 단순한 "다음 토큰 예측 → 추가 → 반복" 루프와 달리, DLLM은 반복적인 **confidence 기반 디코딩** 전략을 사용합니다.

```python
while jnp.any(masked):
    logits, _ = model(x)
    probs = jax.nn.softmax(logits / temp, axis=-1)
    top_k_probs, top_k_indices = jax.lax.top_k(probs, k=top_k)
    confidences = jnp.sum(top_k_probs, axis=-1)

    # 모델이 충분히 확신하는 위치만 디코딩
    decode_mask = (confidences >= confidence_threshold) & masked
    if not jnp.any(decode_mask):
        # 매 스텝 최소 한 위치는 강제 디코딩
        flat_idx = jnp.argmax(jnp.where(masked, confidences, -1.0))
        decode_mask = decode_mask.at[jnp.unravel_index(flat_idx, decode_mask.shape)].set(True)

    x = jnp.where(decode_mask, sampled_tokens, x)
    masked = masked & ~decode_mask
```

동작 과정은 다음과 같습니다:

1. 마스킹된 토큰 블록으로 시작
2. 전체 시퀀스를 모델에 통과
3. 각 위치에서 confidence(top-k 확률 합) 계산
4. Confidence가 임계값을 넘는 토큰을 **확정**
5. 모든 위치가 디코딩될 때까지 반복
6. 다음 블록으로 이동

결과적으로? GPT처럼 240개 토큰을 240번의 순차적 스텝으로 생성하는 대신, 모델은 매 스텝 여러 토큰을 디코딩할 수 있어 — 때로는 전체 블록을 단 10-20회 반복으로 해결합니다.

---

## 왜 Flax NNX인가?

여기서부터가 핵심입니다. 원본 `tiny-diffusion`은 PyTorch로 구현되어 있지만, 저는 전체 학습 파이프라인을 **Flax NNX** — Flax의 새로운 객체지향 API — 로 재작성했습니다.

### 기존 Flax과 NNX 비교

기존 Flax(`linen`)은 모듈이 불변 데이터클래스였고, 별도의 파라미터 딕셔너리를 관리해야 하는 함수형 프로그래밍 모델을 사용했습니다. NNX는 PyTorch와 매우 유사한 **stateful, 객체지향 모듈**을 도입하여 이를 바꿉니다.

```python
class MultiHeadAttention(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.c_q = nnx.Linear(n_embd, n_embd, use_bias=False, rngs=rngs)
        self.c_k = nnx.Linear(n_embd, n_embd, use_bias=False, rngs=rngs)
        self.c_v = nnx.Linear(n_embd, n_embd, use_bias=False, rngs=rngs)
        self.c_proj = nnx.Linear(n_embd, n_embd, use_bias=False, rngs=rngs)

    def __call__(self, x, cos_sin):
        B, T, C = x.shape
        q = self.c_q(x).reshape(B, T, n_head, head_dim)
        k = self.c_k(x).reshape(B, T, n_head, head_dim)
        v = self.c_v(x).reshape(B, T, n_head, head_dim)
        # ... 어텐션 연산
        return self.c_proj(y)
```

파라미터가 모듈에 직접 존재한다는 점(`self.c_q`, `self.c_k` 등)을 주목하세요. 별도의 `params` 딕셔너리가 필요 없습니다. 코드가 훨씬 읽기 쉽고 디버깅하기 편해집니다.

### NNX로 학습 루프 구성

학습 루프도 마찬가지로 간결합니다.

```python
rngs = nnx.Rngs(1337)
model = DiffusionModel(rngs)
optimizer = nnx.Optimizer(model, optax.adamw(learning_rate))

@nnx.jit
def train_step(model, optimizer, x, y, mask):
    def loss_fn(model):
        _, loss = model(x, y, mask)
        return loss
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)
    return loss
```

핵심 포인트:
- **`nnx.Optimizer`** 는 모델과 옵티마이저 상태를 모두 래핑하여, `optimizer.update(grads)`가 모델 파라미터를 in-place로 업데이트합니다
- **`nnx.jit`** 은 기존 `jax.jit`을 대체하며 NNX 그래프 트레이싱을 자동으로 처리합니다
- **`nnx.value_and_grad`** 는 `jax.value_and_grad`와 동일하게 작동하지만 NNX 모듈을 이해합니다
- `train_state` 보일러플레이트나 `apply_fn` 간접 참조가 필요 없습니다

### `nnx.Variable`로 파라미터가 아닌 상태 관리

한 가지 미묘하지만 중요한 디테일: 미리 계산된 Rotary Embedding은 일반 파라미터가 아닌 `nnx.Variable`을 사용해 저장합니다.

```python
class DiffusionModel(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        # ...
        cos, sin = self._precompute_rotary_embeddings(block_size * 2)
        self.cos = nnx.Variable(cos)  # 학습 가능한 파라미터가 아님
        self.sin = nnx.Variable(sin)
```

`nnx.Variable`은 텐서를 모듈 상태의 일부로 등록(직렬화와 JIT 트레이싱에 포함)하지만, 그래디언트 계산에서는 제외합니다. PyTorch의 `register_buffer`에 해당하는 NNX 기능입니다.

---

## 스케일업: JAX로 TPU 데이터 병렬화 구현

JAX/Flax을 사용한 주요 동기 중 하나는 **Google Cloud TPU**를 활용한 분산 학습이었습니다. JAX는 이를 거의 놀라울 정도로 쉽게 만듭니다.

### Device Mesh와 Sharding

```python
devices = jax.devices()
mesh = Mesh(devices, ('batch',))
data_sharding = NamedSharding(mesh, P('batch'))
replicated_sharding = NamedSharding(mesh, P())
```

단 세 줄로 배치 차원을 사용 가능한 모든 TPU 코어에 분할하는 디바이스 메시를 정의합니다. v3-8 Pod에서는 8개 코어가 각각 배치의 1/8을 처리합니다.

### 데이터 Sharding

```python
def get_batch(split, rng):
    # ... x, x_orig, mask 생성 ...
    x = jax.device_put(x, data_sharding)       # TPU에 분산 배치
    x_orig = jax.device_put(x_orig, data_sharding)
    mask = jax.device_put(mask, data_sharding)
    return x, x_orig, mask
```

`jax.device_put`과 `NamedSharding`을 함께 사용하면 배치 데이터가 자동으로 분산됩니다. 컴파일러가 나머지를 처리하므로 수동 통신 프리미티브가 필요 없습니다.

### 확장된 아키텍처

TPU 버전은 PyTorch 기준 대비 모델을 크게 확장했습니다.

| 파라미터 | PyTorch (GPU) | JAX/Flax (TPU) |
| :--- | :--- | :--- |
| `n_embd` | 384 | 768 |
| `n_head` | 6 | 12 |
| `n_layer` | 6 | 12 |
| `block_size` | 256 | 1024 |
| `batch_size` | 64 | 256 |
| 파라미터 수 | ~10.7M | ~85M |

이를 **GPT-2 Small/Medium 규모**에 컨텍스트 윈도우가 4배 더 큰 모델로 끌어올렸습니다. 단일 GPU에서는 엄청나게 느렸을 것이지만, TPU Pod에서는 효율적으로 학습됩니다.

---

## 모델 아키텍처 심층 분석

아키텍처 자체는 여러 현대적 설계 선택이 적용된 깔끔한 Transformer입니다.

**LayerNorm 대신 RMSNorm** — 학습 가능한 파라미터가 없는 간소화된 정규화:
```python
def rms_norm(x):
    return x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + 1e-6)
```

**Rotary Positional Embeddings (RoPE)** — Q와 K에 적용되는 상대적 위치 인코딩:
```python
def apply_rotary_emb(x, cos, sin):
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return jnp.concatenate([y1, y2], axis=-1)
```

**QK-Norm** — 어텐션 전에 Query와 Key를 정규화 (학습 안정화):
```python
q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
q, k = rms_norm(q), rms_norm(k)  # QK norm
```

**ReLU-Squared 활성화 함수** — GeGLU의 간단하면서도 효과적인 대안:
```python
x = jax.nn.relu(x) ** 2
```

**모든 Linear 레이어에 bias 없음** — 대규모 학습의 현대적 베스트 프랙티스를 따름.

전체 블록은 표준 Pre-Norm Transformer 패턴을 따릅니다.

```python
class Block(nnx.Module):
    def __call__(self, x, cos_sin):
        x = x + self.attn(rms_norm(x), cos_sin)
        x = x + self.mlp(rms_norm(x))
        return x
```

---

## 디코딩 과정 시각화

DLLM의 가장 매력적인 측면 중 하나는 텍스트가 생성되는 과정을 지켜보는 것입니다. 노트북에는 디코딩 과정을 "암호 해독"처럼 렌더링하는 실시간 시각화가 포함되어 있습니다. 마스킹된 위치는 무작위 암호 문자가 빠르게 순환하다가, 모델이 확신을 얻으면 진홍색 하이라이트와 함께 해당 문자가 확정됩니다.

시각화는 각 반복에서 `decode_mask`를 추적하여 작동합니다.

```python
while jnp.any(masked):
    # ... 모델 순전파 ...
    # 마스킹된 위치: 빠르게 회전하는 암호 문자 (파란색 네온)
    # 디코딩된 위치: 높은 confidence로 확정 (진홍색 하이라이트)
    display_crypto_state(temp_tokens, current_mask, block_start)
```

보는 재미가 있습니다. 문자들이 노이즈에서 "결정화"되듯 나타나며, 흔한 단어와 패턴을 먼저 해결하고 애매한 위치를 나중에 채우는 경향이 있습니다.

---

## 배운 점

**DLLM은 생각보다 접근하기 쉽습니다.** GPT를 Diffusion 모델로 바꾸는 데 필요한 5가지 수정은 각각 작고 개별적으로 잘 이해된 것들입니다. 결과물은 우아하며 근본적으로 다른 방식으로 텍스트를 생성합니다.

**Flax NNX는 JAX의 진입 장벽을 크게 낮춥니다.** 함수형 프로그래밍 오버헤드 때문에 JAX를 피해왔다면, NNX가 정답입니다. API가 PyTorch와 충분히 비슷하여 코드 포팅이 수월하면서도, JAX의 컴파일과 분산 컴퓨팅 이점을 모두 누릴 수 있습니다.

**JAX로 TPU 스케일링은 (거의) 너무 쉽습니다.** `Mesh`, `NamedSharding`, `device_put`으로 TPU 코어에 걸쳐 학습을 분산하는 데 단 몇 줄면 충분합니다. 컴파일러가 통신과 동기화를 자동으로 처리합니다.

**병렬 디코딩은 실제로 빠릅니다.** 한 번에 1개가 아닌 20-30개 토큰을 디코딩하는 것을 지켜보는 것은 비자기회귀 생성(non-autoregressive generation)이 왜 활발한 연구 분야인지를 보여주는 인상적인 시연입니다.

---

## 시작하기

전체 구현은 다음 파일에서 탐색할 수 있습니다.

- `diffusion_nnx.ipynb` — 인터랙티브 TPU 노트북 (Google Colab 호환)
- `diffusion_nnx.py` — 독립 실행형 TPU 학습 스크립트
- `diffusion.py` / `gpt.py` — 비교용 PyTorch 구현

**로컬에서 실행:**
```bash
git clone https://github.com/yblee/tpu-project-2026.git
cd tpu-project-2026/tiny-diffusion
uv sync
uv run diffusion.py  # 사전 학습된 가중치로 텍스트 생성
```

**TPU에서 실행:**
TPU가 활성화된 Google Colab에서 `diffusion_nnx.ipynb`를 열고 모든 셀을 실행하세요. 노트북이 자동으로 JAX TPU 의존성을 설치하고, Kaggle에서 셜록 홈즈 데이터셋을 다운로드하여 학습을 시작합니다.

---

## 참고 및 크레딧

이 프로젝트는 [yblee/tpu-project-2026](https://github.com/yblee/tpu-project-2026)에서 호스팅됩니다. JAX/Flax NNX 구현과 TPU 적응은 Nathan Barry의 원본 [tiny-diffusion](https://github.com/nathan-barry/tiny-diffusion)에 크게 **영감을 받았습니다** — GPT를 Diffusion 모델로 바꾸는 데 얼마나 적은 변경이 필요한지를 보여주는 훌륭한 교육용 프로젝트입니다. PyTorch 기준 코드는 Andrej Karpathy의 [nanochat](https://github.com/karpathy/nanochat)과 ["Let's build GPT"](https://github.com/karpathy/ng-video-lecture) 구현을 기반으로 합니다.
