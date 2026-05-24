"""
Template: Fine-Tuning de LLMs (LoRA/QLoRA con Hugging Face)
Este script contiene el código estándar para ajustar (fine-tune) modelos de lenguaje
usando Parameter-Efficient Fine-Tuning (PEFT) con LoRA y SFTTrainer de TRL.
"""

import os
import torch
from typing import Dict, Any

# ==========================================
# REQUISITOS DE LIBRERÍAS (pip install):
# pip install transformers peft trl datasets accelerate
# Opcional para cuantización (sólo Linux/Windows con CUDA): pip install bitsandbytes
# ==========================================

try:
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        BitsAndBytesConfig
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer
    HAS_HF_SUITE = True
except ImportError:
    HAS_HF_SUITE = False


def build_finetuning_pipeline(
    model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    output_dir: str = "./results"
):
    if not HAS_HF_SUITE:
        raise ImportError(
            "Debes instalar transformers, peft, trl, datasets y accelerate para ejecutar esta plantilla."
        )

    print(f"[*] Cargando Tokenizador para: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Requerido para muchos modelos generativos (causal decoder)

    # Configuración de Cuantización de 4 bits (QLoRA)
    # NOTA: bitsandbytes no está soportado oficialmente de forma nativa en macOS (Apple Silicon).
    # Si se entrena en macOS, se debe usar bfloat16/float32 sin BitsAndBytesConfig.
    # En un servidor Linux con GPU Nvidia, activa `use_qlora = True`.
    use_qlora = False
    
    bnb_config = None
    device_map = "auto"
    
    if use_qlora:
        print("[*] Configurando cuantización de 4 bits (QLoRA)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        print("[*] Cargando modelo en precisión float16 estándar (sin cuantización bitsandbytes)...")

    # Determinar el tipo de tensor según el hardware
    if torch.backends.mps.is_available():
        device_map = {"": "mps"}
        torch_dtype = torch.float32  # MPS tiene mejor soporte con float32/float16 sin bnb
    elif torch.cuda.is_available():
        device_map = "auto"
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        device_map = {"": "cpu"}
        torch_dtype = torch.float32

    print(f"[*] Cargando Modelo base: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch_dtype
    )
    
    # Configurar LoRA (Low-Rank Adaptation)
    peft_config = LoraConfig(
        r=8,                       # Rango de la matriz adaptadora
        lora_alpha=16,             # Factor de escalado
        target_modules=["q_proj", "v_proj"], # Capas a adaptar (Varía según la arquitectura, ej. Llama tiene q_proj, v_proj, k_proj, o_proj)
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # ==========================================
    # DATASET DE EJEMPLO (Formato Atención al Cliente)
    # ==========================================
    # Para SFT (Supervised Fine-Tuning) se recomienda estructurar en formato de Chat/Instrucción
    # Ej: "### Usuario: {pregunta} \n### Asistente: {respuesta}"
    
    sample_dataset = [
        {
            "prompt": "### Usuario: ¿Cómo puedo restablecer mi contraseña?\n### Asistente: Puedes restablecer tu contraseña haciendo clic en '¿Olvidaste tu contraseña?' en la página de inicio de sesión y siguiendo las instrucciones enviadas a tu correo."
        },
        {
            "prompt": "### Usuario: ¿Cuáles son las horas de soporte técnico?\n### Asistente: Nuestro equipo de soporte técnico está disponible de lunes a viernes, de 9:00 AM a 6:00 PM, hora de Canarias."
        },
        {
            "prompt": "### Usuario: ¿Hacen envíos internacionales?\n### Asistente: Sí, realizamos envíos a todo el mundo. Los tiempos y costos varían dependiendo de la ubicación del cliente."
        }
    ]
    
    # Convertir a Dataset de HuggingFace
    dataset = Dataset.from_list(sample_dataset)

    # Configuración de Entrenamiento
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit" if use_qlora else "adamw_torch",
        save_steps=10,
        logging_steps=2,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="none" # Cambiar a "wandb" o "tensorboard" en producción para loguear
    )

    # Inicializar SFTTrainer de TRL
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="prompt",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
    )

    print("[*] Iniciando entrenamiento...")
    # trainer.train()  # Descomentar para ejecutar de verdad
    print("[+] Simulación de entrenamiento finalizada. Para entrenar realmente, ejecuta `trainer.train()`.")
    
    # Guardar modelo adaptado
    # trainer.model.save_pretrained(os.path.join(output_dir, "lora_adapter"))
    # print(f"[+] Adaptador LoRA guardado en: {os.path.join(output_dir, 'lora_adapter')}")


if __name__ == "__main__":
    if HAS_HF_SUITE:
        # Nota: Descargar un modelo de 1.1B puede tomar unos minutos dependiendo de tu conexión.
        # Solo se ejecuta la inicialización de la configuración para comprobar validez.
        try:
            build_finetuning_pipeline()
        except Exception as e:
            print(f"[!] Error ejecutando inicialización de pipeline: {e}")
            print("[i] Esto es normal si no se dispone de suficiente memoria RAM o GPU local.")
    else:
        print("[!] Por favor instala las librerías de Hugging Face primero.")
