#!/usr/bin/env python3
"""QLoRA domain adaptation over provenance-preserving JSONL documents."""
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


def format_record(row: dict) -> str:
    return (
        f"SOURCE_URL: {row['source_url']}\n"
        f"LICENSE: {row['license']}\n"
        f"DOMAIN: {row.get('domain', 'unknown')}\n\n"
        f"{row['text']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="QLoRA causal-language-model adaptation on a local provenance corpus"
    )
    parser.add_argument('--model', required=True, help='model-hub model id or local model path')
    parser.add_argument('--data', required=True, help='JSONL corpus produced by clean_corpus.py')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--epochs', type=float, default=1.0)
    parser.add_argument('--max-seq-length', type=int, default=2048)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    args = parser.parse_args()

    if args.epochs <= 0:
        raise SystemExit('epochs must be > 0')
    if args.max_seq_length <= 0:
        raise SystemExit('max-seq-length must be > 0')
    if args.learning_rate <= 0:
        raise SystemExit('learning-rate must be > 0')
    if not torch.cuda.is_available():
        raise SystemExit('QLoRA 4-bit training requires a CUDA device')

    dataset = load_dataset('json', data_files=args.data, split='train')
    required = {'text', 'source_url', 'license', 'content_sha256'}
    missing = required.difference(dataset.column_names)
    if missing:
        raise ValueError(f'Corpus lacks required provenance fields: {sorted(missing)}')

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

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization,
        device_map='auto',
        trust_remote_code=False,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias='none',
            task_type='CAUSAL_LM',
            target_modules=[
                'q_proj',
                'k_proj',
                'v_proj',
                'o_proj',
                'gate_proj',
                'up_proj',
                'down_proj',
            ],
        ),
    )

    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        fp16=True,
        report_to='none',
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
    (args.output / 'training_manifest.json').write_text(
        json.dumps(
            {
                'base_model': args.model,
                'data': args.data,
                'epochs': args.epochs,
                'max_seq_length': args.max_seq_length,
                'learning_rate': args.learning_rate,
                'objective': 'causal_lm_domain_adaptation',
                'method': 'qlora_nf4',
            },
            indent=2,
        )
        + '\n',
        encoding='utf-8',
    )


if __name__ == '__main__':
    main()
