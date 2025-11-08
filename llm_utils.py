import torch
from unsloth import FastLanguageModel
from datasets import load_dataset

def salvar_modelo(my_model_name, model, tokenizer):
    #This ONLY saves the LoRA adapters, and not the full model.
    model.save_pretrained(my_model_name) # Local saving
    tokenizer.save_pretrained(my_model_name)
    print('Modelo', my_model_name, 'salvo.')
    # model.push_to_hub("your_name/lora_model", token = "...") # Online saving
    # tokenizer.push_to_hub("your_name/lora_model", token = "...") # Online saving

def carregar_modelo(model_name, 
                    max_seq_length, 
                    dtype, 
                    load_in_4bit,
                    criar_camada_LORA:bool = False,
                    inference:bool = False):
    model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )
    if criar_camada_LORA:
        # Criar LORA 
        model = FastLanguageModel.get_peft_model(
            model,
            r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj",],
            lora_alpha = 16,
            lora_dropout = 0, # Supports any, but = 0 is optimized
            bias = "none",    # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
            random_state = 3407,
            use_rslora = False,  # We support rank stabilized LoRA
            loftq_config = None, # And LoftQ
        )
    if inference: FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    print('Modelo', model_name, 'carregado.')
    return model, tokenizer

# ==================================================== PRINT MEMORY USAGE ==============================================================

def print_memory_usage():
    #@title Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

#usar após o treino
def print_memory_used_in_training(trainer_stats):
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory         /max_memory*100, 3)
    lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

#================================================================         DATASETS          ================================================================

alpaca_prompt_eng = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

alpaca_prompt_pt = """Abaixo está uma instrução que descreve uma tarefa, juntamente com uma entrada que fornece mais contexto. Escreva uma resposta que complete adequadamente o pedido.

### Instrução:
{}

### Entrada:
{}

### Resposta:
{}"""

alpaca_prompt = alpaca_prompt_pt
#def generate_and_tokenize_prompt(data_point):
#    full_prompt = alpaca_prompt.format(data_point["instruction"], data_point["input"], data_point['output']) + EOS_TOKEN
#    return tokenized_full_prompt
def create_mapping_function(eos_token, alpaca_prompt):
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs       = examples["input"]
        outputs      = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = alpaca_prompt.format(instruction, input, output) + eos_token
            texts.append(text)
        return { "text" : texts, }
    return formatting_prompts_func
def carregar_dataset(data_files, eos_token):
    formatting_prompts_func = create_mapping_function(eos_token = eos_token, 
                                                      alpaca_prompt = alpaca_prompt)
    dataset = load_dataset("json", data_files = data_files, split = "train")
    dataset = dataset.map(formatting_prompts_func, batched = True,)
    split_dataset = dataset.train_test_split(test_size = 128, 
                                                   shuffle = True, 
                                                   seed = 42)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    
    print('train_dataset:', train_dataset)
    print('eval_dataset:', eval_dataset)
    return train_dataset, eval_dataset

# offset
def pick_from_dataset(data_set, batch_start:int, batch_end:int):
    new_dataset = data_set.select(range(batch_start, batch_end))
    print('New train_dataset:', new_dataset)
    print('New train_dataset picked from', str(batch_start), 'to', batch_end)
    return new_dataset