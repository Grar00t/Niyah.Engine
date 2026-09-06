#!/usr/bin/env python3
"""Direct local inference for a base model or saved PEFT adapter."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def load_model(model_ref: str):
    model_path = Path(model_ref)
    if model_path.is_dir() and (model_path / 'adapter_config.json').exists():
        return AutoPeftModelForCausalLM.from_pretrained(model_ref, device_map='auto')
    return AutoModelForCausalLM.from_pretrained(
        model_ref,
        device_map='auto',
        trust_remote_code=False,
    )


def build_prompt(tokenizer, prompt: str) -> str:
    if getattr(tokenizer, 'chat_template', None):
        return tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def append_audit(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate text with a local model or PEFT adapter')
    parser.add_argument('--model', required=True)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top-p', type=float, default=0.95)
    parser.add_argument('--audit-log', type=Path)
    args = parser.parse_args()

    if args.max_new_tokens <= 0:
        raise SystemExit('max-new-tokens must be > 0')
    if args.temperature < 0:
        raise SystemExit('temperature must be >= 0')
    if not 0.0 < args.top_p <= 1.0:
        raise SystemExit('top-p must be in (0, 1]')

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model(args.model)

    rendered_prompt = build_prompt(tokenizer, args.prompt)
    inputs = tokenizer(rendered_prompt, return_tensors='pt').to(model.device)
    generation = {
        'max_new_tokens': args.max_new_tokens,
        'pad_token_id': tokenizer.eos_token_id,
        'do_sample': args.temperature > 0,
    }
    if args.temperature > 0:
        generation['temperature'] = args.temperature
        generation['top_p'] = args.top_p

    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation)

    prompt_tokens = inputs['input_ids'].shape[-1]
    completion_ids = output_ids[0, prompt_tokens:]
    answer = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

    output = {
        'model': args.model,
        'answer': answer,
        'input_sha256': sha256_text(args.prompt),
        'output_sha256': sha256_text(answer),
        'generated_tokens': int(completion_ids.shape[-1]),
    }

    if args.audit_log:
        append_audit(
            args.audit_log,
            {
                'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                **output,
            },
        )

    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
