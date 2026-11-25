"""Portable single-GPU recipe for the two multi-object training stages."""
import argparse
import gzip
import hashlib
import json
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent


def records(stage):
    with gzip.open(HERE / f"{stage}.jsonl.gz", "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def supervised_target(row):
    evidence = sorted(row["gold"]["evidence"], key=lambda item: -Decimal(item["value"]))
    resolution = Decimal(row["resolution"])
    readings = []
    for item in evidence:
        value = (Decimal(item["value"]) / resolution).to_integral_value() * resolution
        readings.append([item["entity"], int(value) if value == value.to_integral_value() else float(value)])
    return json.dumps({"readings": readings, "ranking": row["gold"]["ordering"]}, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["sft", "grpo"])
    ap.add_argument("--images", type=Path, required=True, help="Parent of plotqa/train/png/")
    init = ap.add_mutually_exclusive_group(required=True)
    init.add_argument("--adapter", type=Path, help="Initial PEFT adapter directory")
    init.add_argument("--fresh-lora", action="store_true", help="Start a new SFT experiment from the base model")
    ap.add_argument("--output", type=Path, required=True, help="New output directory")
    ap.add_argument("--seed", type=int, choices=[20261125, 20261126], default=20261125,
                    help="GRPO training seed (SFT uses its frozen seed)")
    args = ap.parse_args()
    from datasets import Dataset, Image, List
    from peft import LoraConfig, PeftModel, get_peft_model
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, set_seed
    from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer
    from training.reward import score

    protocol = json.loads((HERE / "protocol.json").read_bytes())
    supervised = args.stage == "sft"
    adapter_sha = protocol["sft"]["initial"] if supervised else protocol["grpo"]["init_sha"]
    if args.fresh_lora and not supervised:
        ap.error("--fresh-lora applies to SFT only")
    if args.adapter:
        observed = hashlib.sha256((args.adapter / "adapter_model.safetensors").read_bytes()).hexdigest()
        if observed != adapter_sha:
            marker = args.adapter / "training-origin.json"
            origin = json.loads(marker.read_bytes()) if marker.is_file() else {}
            if supervised or origin != {"stage": "sft", "sha256": observed, "base_revision": protocol["revision"]}:
                ap.error("Use the archived initial adapter or an SFT adapter produced by this recipe")
    if args.output.exists() and any(args.output.iterdir()):
        ap.error("Choose an empty output directory")
    rows = records(args.stage)
    examples = []
    gold = {}
    for row in rows:
        image = args.images / row["image"]
        if hashlib.sha256(image.read_bytes()).hexdigest() != row["image_sha256"]:
            raise ValueError(f"Image mismatch: {row['qid']}")
        prompt = row["question"] + "\n\n" + protocol["instruction"]
        example = {"qid": row["qid"], "prompt": [{"role": "user", "content": prompt}]}
        if supervised:
            example["images"] = [str(image)]
            example["completion"] = [{"role": "assistant", "content": supervised_target(row)}]
        else:
            example["image"] = str(image)
        examples.append(example)
        gold[row["qid"]] = row["gold"]
    dataset = Dataset.from_list(examples)
    dataset = dataset.cast_column("images", List(Image())) if supervised else dataset.cast_column("image", Image())
    seed = protocol["sft"]["seed"] if supervised else args.seed
    set_seed(seed)
    torch.set_num_threads(4)
    processor = AutoProcessor.from_pretrained(protocol["base"], revision=protocol["revision"], padding_side="left")
    processor.image_processor.size["longest_edge"] = 768 * 32 * 32
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        protocol["base"], revision=protocol["revision"], dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda")
    if args.fresh_lora:
        config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"])
        model = get_peft_model(base, config)
    else:
        model = PeftModel.from_pretrained(base, args.adapter, is_trainable=True)
    common = dict(output_dir=str(args.output), seed=seed, data_seed=seed, bf16=True, report_to="none",
                  remove_unused_columns=False, dataloader_num_workers=0, dataloader_pin_memory=False,
                  gradient_accumulation_steps=8, optim="adamw_torch", weight_decay=0.0,
                  logging_steps=1, save_strategy="steps", save_total_limit=2)
    if supervised:
        cfg = SFTConfig(**common, per_device_train_batch_size=1, num_train_epochs=1,
                        learning_rate=3e-5, lr_scheduler_type="cosine", warmup_steps=50, max_length=None,
                        completion_only_loss=True, gradient_checkpointing=True, save_steps=50,
                        gradient_checkpointing_kwargs={"use_reentrant": False})
        trainer = SFTTrainer(model=model, args=cfg, train_dataset=dataset, processing_class=processor)
    else:
        cfg = GRPOConfig(**common, per_device_train_batch_size=2, max_steps=64,
                         steps_per_generation=4, num_generations=4, shuffle_dataset=False,
                         learning_rate=1e-5, lr_scheduler_type="constant", beta=0.0, save_steps=16,
                         temperature=0.8, top_p=0.95, top_k=0, max_prompt_length=None,
                         max_completion_length=512, loss_type="grpo", scale_rewards="group",
                         gradient_checkpointing=False, num_iterations=1, use_vllm=False)

        def reading_reward(completions, qid, **kwargs):
            return [score(answer[-1]["content"], gold[question])["score"] / 100
                    for question, answer in zip(qid, completions, strict=True)]

        trainer = GRPOTrainer(model=model, args=cfg, train_dataset=dataset,
                              processing_class=processor, reward_funcs=reading_reward)
    trainer.train()
    output_adapter = args.output / "adapter"
    trainer.save_model(str(output_adapter))
    digest = hashlib.sha256((output_adapter / "adapter_model.safetensors").read_bytes()).hexdigest()
    marker = {"stage": args.stage, "sha256": digest, "base_revision": protocol["revision"]}
    (output_adapter / "training-origin.json").write_text(json.dumps(marker), encoding="utf-8")


if __name__ == "__main__":
    main()
