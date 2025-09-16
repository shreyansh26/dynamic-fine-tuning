import random
from utils.training_utils import (
    SupervisedDataset,
    DataCollatorForSupervisedDataset,
)
from utils.loss_fn import ForCausalLMLoss, DFTForCausalLMLoss
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
from tqdm import tqdm
from datetime import datetime
from torch.distributed.fsdp import fully_shard, FSDPModule, MixedPrecisionPolicy
from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2ForCausalLM, Qwen2Attention, Qwen2MLP
from dotenv import load_dotenv
import functools
import torch.distributed as dist
import wandb
import uuid
import torch
import transformers
import os
import math
import numpy as np
import argparse
from datetime import datetime
try:
    # Python 3.9+
    from zoneinfo import ZoneInfo
    IST = ZoneInfo("Asia/Kolkata")
except ImportError:
    # Fallback to pytz for older Pythons
    import pytz
    IST = pytz.timezone("Asia/Kolkata")

load_dotenv()

hf_token = os.getenv("HF_TOKEN")

def current_timestamp_ist() -> str:
    """
    Returns the current time in IST (Asia/Kolkata) as an ISO-like string:
      YYYY-MM-DDTHH:MM:SS
    e.g. "2025-07-10T14:30:05"
    """
    now = datetime.now(tz=IST)
    return now.strftime("%Y-%m-%dT%H:%M:%S")

def disable_model_dropout(model: torch.nn.Module):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def setup_model(model_name, max_length, loss_type):
    config = transformers.AutoConfig.from_pretrained(
        model_name,
        token=hf_token,
    )
    config.use_cache = False

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    if loss_type == "sft":
        model.loss_function = ForCausalLMLoss
    elif loss_type == "dft":
        model.loss_function = DFTForCausalLMLoss
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=max_length,
        padding_side="right",
        use_fast=False,
        token=hf_token,
        trust_remote_code=True,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return model, tokenizer

def compile_model(model, backend="inductor", fullgraph=False):
    for m in reversed(list(model.modules())):
        if isinstance(m, Qwen2Attention):
            m.compile(backend=backend, fullgraph=fullgraph)
        if isinstance(m, Qwen2MLP):
            m.compile(backend=backend, fullgraph=fullgraph)

def compile_loss(loss_function, backend="inductor"):
    return torch.compile(loss_function, backend=backend)

def evaluation(
    model,
    eval_dataloader,
    wandb,
    local_rank,
):
    if local_rank == 0:
        print("RUNNING EVAL")

    model.eval()
    losses = 0
    for step, batch in enumerate(eval_dataloader):
        inputs = {
                "input_ids": batch["input_ids"].to(model.device),
                "labels": batch["labels"].to(model.device),
            }
        if "attention_mask" in batch:
            inputs["attention_mask"] = batch["attention_mask"].to(model.device)
        if "position_ids" in batch:
            inputs["position_ids"] = batch["position_ids"].to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs.loss
        losses += loss.float()

    losses = losses / (step + 1)
    val_loss = get_all_reduce_mean(losses.clone()).item()

    if local_rank == 0:
        print(f"Validation Loss {val_loss:.5f}")

    if local_rank == 0:
        wandb.log(
            {
                "val_loss": val_loss,
            }
        )

    return val_loss


def get_dataloader(
    max_length,
    dataset,
    world_size,
    local_rank,
    shuffle,
    seed,
    collator,
    batch_size,
):
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=shuffle,
        seed=seed,
    )

    loader = DataLoader(
        dataset,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        batch_size=batch_size,
        collate_fn=collator,
        sampler=sampler,
    )

    return sampler, loader


def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]

    result += list(model._parameters.keys())
    return result


def get_optimizer(model, lr, weight_decay):
    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    return torch.optim.AdamW(
        params=optimizer_grouped_parameters,
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=weight_decay,
    )


def should_run_eval(total_steps, times_to_run, current_step):
    return current_step % (total_steps // times_to_run) == 0


def log_stats(pbar, wandb, epoch, loss_tensor, grad_norm, scheduler, step_size):
    last_lr = scheduler.get_last_lr()[0]

    wandb.log(
        {
            "current_loss": loss_tensor,
            "current_epoch": epoch,
            "learning_rate": last_lr,
            "grad_norm": grad_norm,
        },
    )

    current_loss = f"{loss_tensor:.5f}"
    current_lr = f"{last_lr:.10f}"

    pbar.set_description(f"Epoch {epoch:.2f}, Loss: {current_loss}, LR: {current_lr}")
    pbar.update(step_size)


def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor


def get_warmup_steps(num_training_steps, warmup_ratio=0.1):
    return math.ceil(num_training_steps * warmup_ratio)

def get_decay_steps(num_training_steps, decay_ratio=0.25):
    return math.ceil(num_training_steps * decay_ratio)

def clip_model_gradients(model, max_grad_norm):
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm).item()


def get_scheduler(local_rank, scheduler_type, optimizer, max_steps):
    warmup_steps = get_warmup_steps(max_steps)
    decay_steps = get_decay_steps(max_steps)

    if local_rank == 0 and scheduler_type != "warmup_stable_decay":
        print(f"[WARMUP STEPS]: {warmup_steps}")
        print(f"[MAX STEPS]: {max_steps}")
        print(f"[SCHEDULER]: {scheduler_type}")

    if local_rank == 0 and scheduler_type == "warmup_stable_decay":
        print(f"[WARMUP STEPS]: {warmup_steps}")
        print(f"[DECAY STEPS]: {decay_steps}")
        print(f"[STABLE STEPS]: {max_steps - decay_steps - warmup_steps}")
        print(f"[MAX STEPS]: {max_steps}")
        print(f"[SCHEDULER]: {scheduler_type}")

    if scheduler_type != "warmup_stable_decay":
        return transformers.get_scheduler(
            name=scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )
    else:
        return transformers.get_wsd_schedule(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_decay_steps=decay_steps,
            num_training_steps=max_steps,
            warmup_type="linear",
            decay_type="cosine"
        )


def save_model(local_rank, model, tokenizer, outpath, current_epoch, current_step, use_dcp_api):
    if use_dcp_api:
        cpu_state_dict = get_model_state_dict(
                model=model,
                options=StateDictOptions(
                    full_state_dict=True,
                    cpu_offload=True,
                ),
            )
        if local_rank == 0:
            print(f"SAVING MODEL")
            cfg = model.config
            cfg.architectures = ["Qwen2ForCausalLM"]
            base = Qwen2ForCausalLM(cfg)
            base.load_state_dict(cpu_state_dict, strict=True)
            outpath += f"/epoch_{current_epoch}/step_{current_step}"
            base.save_pretrained(outpath)
            tokenizer.save_pretrained(outpath)
    else:
        sharded_sd = model.state_dict()
        cpu_state_dict = {}
        for param_name, sharded_param in sharded_sd.items():
            full_param = sharded_param.full_tensor()
            if local_rank == 0:
                cpu_state_dict[param_name] = full_param.cpu()
            else:
                del full_param

        if local_rank == 0:
            print(f"SAVING MODEL")
            outpath += f"/epoch_{current_epoch}/step_{current_step}"
            model.save_pretrained(outpath, state_dict=cpu_state_dict)
            tokenizer.save_pretrained(outpath)
    
    dist.barrier()  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dcp-api", action="store_true", default=False)
    parser.add_argument("--loss-type", choices=["sft", "dft"], default="sft")

    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", rank=local_rank, world_size=world_size)

    model_name = "Qwen/Qwen2.5-Math-1.5B"
    scheduler_type = "cosine"
    # scheduler_type = "warmup_stable_decay"
    seed = 877645  # set your seed
    transformers.set_seed(seed)

    date_of_run = current_timestamp_ist()
    notes = f"qwen2.5_1.5b_flash_attn_fsdp2_torch_compile_dcp_numina_50k_{args.loss_type}"
    run_id = "exp_" + date_of_run + "_" + notes
    output_dir = f"/mnt/ssd2/shreyansh/models/qwen25/{run_id}"
    max_length = 8192  # adjust as needed
    disable_dropout = False
    gradient_checkpointing = True
    clip_gradients = True
    shuffle = True  # multipack sampler already does random sampling
    train_batch_size = 16 # adjust as needed
    validation_batch_size = 16  # adjust as needed
    epochs = 1  # adjust as needed
    gradient_accumulation_steps = 4
    lr = 5e-05 # 5e-06  # adjust as needed
    weight_decay = 0.01  # adjust as needed
    gradient_clipping = 1.0  # adjust as needed
    train_on_inputs = False  # whether to train on instruction tokens
    packing = None # None, "ffd"
    compile = True

    if local_rank == 0:
        print("Using DCP API: ", args.dcp_api)
        print(f"OUTPUT DIR: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    model, tokenizer = setup_model(model_name, max_length, loss_type=args.loss_type)
    num_params = sum([p.numel() for p in model.parameters()])

    if compile:
        torch._dynamo.config.capture_scalar_outputs = True
        backend = "inductor"
        compile_model(model, backend=backend, fullgraph=False)
        model.loss_function = compile_loss(model.loss_function, backend=backend)

    fsdp_kwargs = {
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
        ),
        "reshard_after_forward": True
    }

    for layer in model.model.layers:
        fully_shard(layer, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)

    assert isinstance(model, Qwen2ForCausalLM)
    assert isinstance(model, FSDPModule)

    optimizer = get_optimizer(model, lr, weight_decay)
    
    train_ds = ["data/numina_math_train.jsonl"]
    val_ds = ["data/numina_math_test.jsonl"]

    train_dataset = SupervisedDataset(train_on_inputs, tokenizer, train_ds, packing=packing)
    val_dataset = SupervisedDataset(train_on_inputs, tokenizer, val_ds, packing=packing)
    if packing == "ffd":
        assert model.config._attn_implementation == "flash_attention_2"
        collator = DataCollatorForLanguageModeling(
            pad_token_id=tokenizer.pad_token_id,
            completion_only_loss=True,
            padding_free=True,
            return_position_ids=True,
            pad_to_multiple_of=None,
        )
    else:
        collator = DataCollatorForSupervisedDataset(tokenizer)

    train_sampler, train_loader = get_dataloader(
        max_length,
        train_dataset,
        world_size,
        local_rank,
        shuffle,
        seed,
        collator,
        train_batch_size,
    )
    val_sampler, val_loader = get_dataloader(
        max_length,
        val_dataset,
        world_size,
        local_rank,
        shuffle,
        seed,
        collator,
        validation_batch_size,
    )

    total_steps_per_epoch = len(train_loader)

    max_steps = total_steps_per_epoch * epochs / gradient_accumulation_steps
    scheduler = get_scheduler(local_rank, scheduler_type, optimizer, max_steps)

    if local_rank == 0:
        run = wandb.init(
            project="combined_sft_qwen2.5_1.5b_numina",
            name=run_id,
            config={
                "model_name": model_name,
                "run_id": run_id,
                "date": date_of_run,
                "dataset_size": len(train_dataset),
                "dataset": ",".join(train_ds),
                "validation": ",".join(val_ds),
                "weight_decay": weight_decay,
                "clip_gradients": clip_gradients,
                "learning_rate": lr,
                "shuffle": shuffle,
                "seed": seed,
                "disable_dropout": disable_dropout,
                "train_on_inputs": train_on_inputs,
                "epochs": epochs,
                "acc_steps": gradient_accumulation_steps,
                "batch_size": train_batch_size,
                "total_batch_size": train_batch_size * world_size,
                "scheduler_type": scheduler_type,
            },
        )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if disable_dropout:
        disable_model_dropout(model)

    model.train()
    dist.barrier()
    train_iterator = iter(train_loader)
    for epoch in range(0, epochs):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        current_epoch = epoch + 1

        pbar = tqdm(
            total=total_steps_per_epoch,
            colour="blue",
            desc=f"Epoch {epoch}.00",
            disable=(local_rank != 0),
        )
        current_step = 0
        while True:
            acc_loss = torch.tensor(0.).to(model.device)
            actual_accumulation_steps = 0
            for acc_step in range(gradient_accumulation_steps):
                model.require_backward_grad_sync = (acc_step == gradient_accumulation_steps - 1) or \
                                                    (current_step == total_steps_per_epoch - 1)
                try:
                    batch = next(train_iterator)
                except StopIteration:
                    train_iterator = iter(train_loader)
                    batch = next(train_iterator)

                inputs = {
                        "input_ids": batch["input_ids"].to(model.device),
                        "labels": batch["labels"].to(model.device),
                    }
                if "attention_mask" in batch:
                    inputs["attention_mask"] = batch["attention_mask"].to(model.device)
                if "position_ids" in batch:
                    inputs["position_ids"] = batch["position_ids"].to(model.device)
                # forward
                outputs = model(**inputs)
                loss = outputs.loss
                acc_loss += loss.item()
                loss /= gradient_accumulation_steps

                # backward
                loss.backward()
                
                current_step += 1
                actual_accumulation_steps += 1

                if current_step == total_steps_per_epoch:
                    break

            acc_loss /= actual_accumulation_steps
            
            # clipping
            if clip_gradients:
                grad_norm = clip_model_gradients(model, gradient_clipping)

            # weight update
            optimizer.step()
            scheduler.step()

            # zero gradients after weight update
            optimizer.zero_grad(set_to_none=True)

            # avg loss over all processes
            acc_loss = get_all_reduce_mean(acc_loss).item()

            if local_rank == 0:
                log_stats(
                    pbar,
                    wandb,
                    round((current_step / total_steps_per_epoch), 2) + epoch,
                    acc_loss,
                    grad_norm,
                    scheduler,
                    actual_accumulation_steps
                )

            if current_step == total_steps_per_epoch:
                validation_loss = evaluation(
                    model,
                    val_loader,
                    wandb,
                    local_rank,
                )

                dist.barrier()  

                save_model(
                    local_rank,
                    model,
                    tokenizer,
                    output_dir,
                    current_epoch,
                    current_step,
                    args.dcp_api,
                )

                dist.barrier()

                model.train()
                break

    # save final model
    save_model(local_rank, model, tokenizer, output_dir, epochs, "final", args.dcp_api)