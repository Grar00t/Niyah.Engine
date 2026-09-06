#!/usr/bin/env python3
"""LoRA domain adaptation over provenance-preserving JSONL documents."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from model_profiles import detect_model_profile


def format_record(row: dict) -> str:
    return (
        f"SOURCE_URL: {row['source_url']}\n"
        f"LICENSE: {row['license']}\n"
        f"DOMAIN: {row.get('domain', 'unknown')}\n\n"
        f"{row['text']}"
    )


def load_trainable_base(model_ref: str, *, native_quantization: str | None):
    if native_quantization == "mxfp4":
        model = AutoModelForCausalLM.from_pretrained(
            model_ref,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=False,
        )
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        return model, "native_mxfp4"

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        quantization_config=quantization,
        device_map="auto",
        trust_remote_code=False,
    )
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )
    return model, "bitsandbytes_nf4"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LoRA causal-language-model domain adaptation on a provenance corpus"
    )
    parser.add_argument("--model", required=True, help="Hugging Face model id or local model path")
    parser.add_argument("--data", required=True, help="JSONL corpus produced by clean_corpus.py")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    args = parser.parse_args()

    if args.epochs <= 0:
        raise SystemExit("epochs must be > 0")
    if args.max_seq_length <= 0:
        raise SystemExit("max-seq-length must be > 0")
    if args.learning_rate <= 0:
        raise SystemExit("learning-rate must be > 0")
    if args.lora_r <= 0 or args.lora_alpha <= 0:
        raise SystemExit("lora-r and lora-alpha must be > 0")
    if not torch.cuda.is_available():
        raise SystemExit("local LoRA training requires a CUDA device")

    profile = detect_model_profile(args.model)
    dataset = load_dataset("json", data_files=args.data, split="train")
    required = {"text", "source_url", "license", "content_sha256"}
    missing = required.difference(dataset.column_names)
    if missing:
        raise ValueError(f"Corpus lacks required provenance fields: {sorted(missing)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(row: dict) -> dict:
        return tokenizer(
            format_record(row),
            truncation=True,
            max_length=args.max_seq_length,
        )

    tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)
    model, quantization_method = load_trainable_base(
        args.model,
        native_quantization=profile.native_quantization,
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(profile.lora_target_modules),
        ),
    )

    use_bf16 = torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()

    args.output.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    (args.output / "training_manifest.json").write_text(
        json.dumps(
            {
                "base_model": args.model,
                "model_profile": profile.name,
                "data": args.data,
                "epochs": args.epochs,
                "max_seq_length": args.max_seq_length,
                "learning_rate": args.learning_rate,
                "objective": "causal_lm_domain_adaptation",
                "method": "lora",
                "quantization": quantization_method,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_target_modules": list(profile.lora_target_modules),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
