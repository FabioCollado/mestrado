# README — Explicação do código de fine-tuning (LoRA/Unsloth + TRL)

Este documento explica os módulos, hiperparâmetros e o fluxo de treinamento/inferência usados no seu projeto de geração de **votos** a partir de **relatórios** com **Llama-3.1-8B** quantizado em **4 bits**, adaptado com **LoRA (PEFT)** e treinado com **TRL/SFTTrainer**.

---

## Estrutura geral

* **`llm_utils.py`**

  * Utilitários para **carregar/salvar** modelo/tokenizer, **medir memória**, e **carregar datasets** no formato *instruction → input → output* (padrão Alpaca).
* **Notebook de treinamento**

  * Define **modelos base 4-bit** (Unsloth), carrega/adiciona **LoRA**, prepara dataset, cria **SFTTrainer** (TRL) e treina/avalia/salva adapters.

---

## 1) `llm_utils.py`

### 1.1 Carregar / salvar

```python
def salvar_modelo(my_model_name, model, tokenizer):
    model.save_pretrained(my_model_name)  # salva APENAS os adapters LoRA
    tokenizer.save_pretrained(my_model_name)
```

* Salva **somente** os *adapters* (PEFT). O diretório conterá os pesos/arquivos da LoRA e metadados para reanexá-la no base model.

```python
def carregar_modelo(model_name, max_seq_length, dtype, load_in_4bit, criar_camada_LORA=False, inference=False):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name, max_seq_length=max_seq_length, dtype=dtype, load_in_4bit=load_in_4bit
    )
    if criar_camada_LORA:
        model = FastLanguageModel.get_peft_model(
            model,
            r=16, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
            lora_alpha=16, lora_dropout=0, bias="none",
            use_gradient_checkpointing="unsloth", random_state=3407, use_rslora=False, loftq_config=None
        )
    if inference: FastLanguageModel.for_inference(model)  # ativa inferência 2x mais rápida (Unsloth)
    return model, tokenizer
```

* **`criar_camada_LORA=True`**: injeta adapters num **modelo base**.
* **`criar_camada_LORA=False`**: para **continuar** de um diretório que **já** contém adapters LoRA (se o diretório for dos adapters salvos).
* **`for_inference`**: ativa otimizações de decodificação.

> **Atenção**: ao continuar de `./fine_tuned_models/...`, esse diretório precisa conter os **arquivos da LoRA** com referência ao **modelo base**. Se você apontar para um modelo base puro, defina `criar_camada_LORA=True` para reanexar a LoRA.

### 1.2 Medição de memória

```python
def print_memory_usage():
    gpu = torch.cuda.get_device_properties(0)
    print(f"GPU = {gpu.name}. Max memory = {gpu.total_memory/1e9:.3f} GB.")
    print(f"{torch.cuda.max_memory_reserved()/1e9:.3f} GB of memory reserved.")
```

```python
def print_memory_used_in_training(trainer_stats):
    # ⚠️ Observação: do jeito atual, "used_memory_for_lora" tende a 0,
    # pois 'start_gpu_memory' é lido depois do treino e usa a mesma métrica de pico.
```

**Correção sugerida** para medir uso durante o treino:

1. Antes de `trainer.train()`: `torch.cuda.reset_peak_memory_stats()`
2. Depois do treino: `peak = torch.cuda.max_memory_reserved()`
3. Compare com um **baseline** salvo **antes** do treinamento.

### 1.3 Datasets (formatação Alpaca)

* *Templates* (PT/EN) — usa-se `alpaca_prompt_pt`:

  ```
  ### Instrução:
  {instruction}

  ### Entrada:
  {input}

  ### Resposta:
  {output}
  ```

* **`create_mapping_function(eos_token, alpaca_prompt)`**: gera `text` concatenando *instruction/input/output + EOS*.

* **`carregar_dataset(data_files, eos_token, test_size)`**:

  * Lê JSON com chaves `instruction`, `input`, `output` (e opcionalmente `desembargador`).
  * Mapeia para campo `text`.
  * Faz `train_test_split(test_size=...)`.

* **`pick_from_dataset(dataset, batch_start, batch_end)`**: subamostra por *offset* (útil p/ experimentos).

---

## 2) Notebook de Treinamento

### 2.1 Seleção do modelo (Unsloth 4-bit)

* Exemplos: `"unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"`.
* **`load_in_4bit=True`**: carrega o base quantizado (BNB 4-bit) → memória baixa e *throughput* alto.

### 2.2 Continuidade de treino

```python
CONTINUAR_TREINAMENTO_SALVO = True  # usa diretório com adapters já salvos
RESUME_FROM_CHECKPOINT = False       # retoma de checkpoint parcial do Trainer
```

* **Continuar treinamento**:

  * Aponte `model_name = MY_MODEL_NAME` (pasta com LoRA).
  * **`criar_camada_LORA=False`** (já existe).
* **Novo treinamento**:

  * `model_name = MODEL_NAME` (base 4-bit).
  * **`criar_camada_LORA=True`** para injetar adapters.

### 2.3 Hiperparâmetros LoRA

* `r=16`, `lora_alpha=16`, `lora_dropout=0`, `bias="none"`.
* `target_modules`: atenção + MLP (`q/k/v/o/gate/up/down_proj`) — ampla capacidade de adaptação.
* `use_gradient_checkpointing="unsloth"`: reduz VRAM com recomputação (trade-off tempo × memória).

### 2.4 Dados

* `max_seq_length = 10000`: contexto longo (atenção ao **VRAM**).
* `packing=False`: mantém exemplos **um por sequência** (mais simples; **packing=True** acelera em exemplos curtos).

### 2.5 Trainer (TRL/SFT)

```python
trainer = SFTTrainer(
  model=model, tokenizer=tokenizer,
  train_dataset=train_dataset, eval_dataset=eval_dataset,
  dataset_text_field="text", max_seq_length=max_seq_length, dataset_num_proc=2,
  packing=False,
  args=TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,    # efetivo = 16
    warmup_steps=5,
    num_train_epochs=1,               # ou use max_steps
    learning_rate=2e-4,
    fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(),
    logging_steps=10,
    do_eval=False,                    # evitar estouro ao avaliar
    save_steps=200,
    load_best_model_at_end=False,
    optim="adamw_8bit",               # otimização em 8-bit (economia de VRAM)
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="outputs",
    report_to="wandb",
  ),
)
```

* **Batch efetivo**: `2 × 8 = 16`.
* **AdamW 8-bit**: menor memória para otimizador.
* **bf16/ fp16**: selecionados conforme GPU (no seu log, *bf16* disponível).

> **Se quiser avaliação durante treino sem OOM**: reduza `max_seq_length` em `eval`, use `eval_accumulation_steps`, `predict_with_generate=False` ou um *subset* de validação.

### 2.6 Treino, salvamento e re-carregamento

* `trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT)`
* `salvar_modelo(...)`: grava **adapters** + tokenizer em `MY_MODEL_NAME`.
* Recarregar para **inferência**:

  ```python
  model, tok = carregar_modelo(MY_MODEL_NAME, ..., criar_camada_LORA=False, inference=True)
  ```

---

## 3) Inferência (exemplo rápido)

```python
inputs = tokenizer([alpaca_prompt.format("Continue a sequência de Fibonacci.",
                                         "1, 1, 2, 3, 5, 8", "")],
                   return_tensors="pt").to("cuda")
from transformers import TextStreamer
_ = model.generate(**inputs, streamer=TextStreamer(tokenizer), max_new_tokens=128, use_cache=True)
```

---

## 4) Boas práticas & armadilhas

* **Memória**: `max_seq_length=10000` com `packing=False` pode exigir picos altos; ajuste *seq len*, *packing* e *batch*.
* **Avaliação**: se preciso, faça **offline** num script separado com `predict_with_generate=False`.
* **Reprodutibilidade**: fixe `seed` (3407), registre *hash* do dataset e *commit* do código.
* **Diretório de adapters**: garanta que contém `adapter_config.json` apontando para o **base model** correto.
* **Métrica de memória**: aplique a **correção** indicada para medir VRAM de treino real.
* **Risco de *catastrophic forgetting***: prefira *LR* baixa (você usa `2e-4`), *weight decay* e monitoramento por validação (quando viável).

---

## 5) Parâmetros principais — resumo

| Componente | Valor                                                              |
| ---------- | ------------------------------------------------------------------ |
| Base       | Meta-Llama-3.1-8B Instruct (Unsloth 4-bit)                         |
| PEFT       | LoRA `r=16`, `alpha=16`, `dropout=0`, `bias=none` em Q/K/V/O + MLP |
| Otimizador | `adamw_8bit`, `lr=2e-4`, `weight_decay=0.01`, `warmup=5`, `linear` |
| Treino     | batch=2, grad-accum=8 (efetivo=16), epochs=1 (ou `max_steps`)      |
| Precisão   | bf16 (se suportado) ou fp16                                        |
| Seq len    | `max_seq_length=10000`                                             |
| Dataset    | Formato Alpaca (instr/input/output + EOS), split 9720/1200         |
| Log        | `wandb` (logs a cada 10 steps), checkpoints a cada 200 steps       |

Pronto — este **README.md** cobre como o código carrega modelos 4-bit, injeta LoRA, prepara dados, treina com TRL e faz inferência, com observações de desempenho e estabilidade.
