# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json

import torch.distributed as dist

from contextlib import contextmanager

from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA

SEQ_LEN = 2048

def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def force_parallel(checkpoint: dict, n_gpus: int = 2):
    assert n_gpus == 2, "More parallelizm to be done."
    ckpt1 = dict()
    ckpt2 = dict()
    for k, v in checkpoint.items():
        size = v.size()
        # ColumnParallel
        if k.split(".")[-2] in ["wq", "wk", "wv", "w1", "w3", "output"]:
            ckpt1[k] = v[:size[0] // 2, :]
            ckpt2[k] = v[size[0] // 2:, :]
        # RowParallel
        elif k.split(".")[-2] in ["wo", "w2", "tok_embeddings"]:
            ckpt1[k] = v[:, :size[1] // 2]
            ckpt2[k] = v[:, size[1] // 2:]
        # NonParallel?
        else:
            ckpt1[k] = v
            ckpt2[k] = v
    return ckpt1, ckpt2


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    # assert world_size == len(
    #     checkpoints
    # ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    if world_size == len(checkpoints):  # default mode
        ckpt_path = checkpoints[local_rank]
        checkpoint = torch.load(ckpt_path, map_location="cpu")
    elif world_size == 2 and len(checkpoints) == 1:  # 7B to two GPUs
        if local_rank == 0:
            ckpt_path = checkpoints[local_rank]
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            checkpoint, _ = force_parallel(checkpoint)
        if local_rank == 1:
            ckpt_path = checkpoints[0]
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            _, checkpoint = force_parallel(checkpoint)
        dist.barrier()
    else:
        raise NotImplementedError("Further parallelization to be done.")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def make_buffers(max_len: int, pad_token: int, gpu: int):
    fin = torch.tensor([-13], dtype=torch.long).cuda()
    prompt = torch.full((1, max_len), pad_token).cuda().long()
    return fin, prompt

def create_settings_file(server_input, filename="settings.json"):

    #write the 4 if statements above more concisely
    prob_mode = server_input.get("prob_mode", False)
    temperature = server_input.get("temperature", 0.0)
    top_p = server_input.get("top_p", 0.95)
    prob_prev_pos = server_input.get("prob_prev_pos", 0)
    stop_str = server_input.get("stop_str", "[PLAN END]")
    prob_top_k = server_input.get("prob_top_k", 15)

    #create settings file
    settings = {
        "prob_mode": prob_mode,
        "temperature": temperature,
        "top_p": top_p,
        "stop_str": stop_str,
        "prob_prev_pos": prob_prev_pos,
        "prob_top_k": prob_top_k
    }
    
    with open(filename, "w") as f:
        json.dump(settings, f)

    

def tokenize_or_wait(prompt: torch.Tensor, fin: torch.Tensor, gen: LLaMA, gpu: int, parallel: bool):
    if gpu == 0:
        # input_prompt = input("Person: ")
        while True:
            # print("waiting...")
            try:
                with open("prompt", "r") as f_prompt:
                    # wait for "`prompt" file to be created, then read and destroy "prompt" file
                    server_input = json.load(f_prompt)

                    assert "prompt" in server_input, "prompt not found in server input"
                    input_prompt = server_input["prompt"]

                    #create settings file
                    create_settings_file(server_input, filename="settings.json")

                    f_prompt.close()

                    os.remove("prompt")
                    break
            except FileNotFoundError:
                time.sleep(0.01)
                continue

        print("[LLaMa] Input prompt: ", input_prompt[:10])  # debug
                
        if len(input_prompt) == 0:
            prompt[0, 0] = fin
        else:
            tokens = gen.tokenize(input_prompt).cuda()
            prompt[0, :len(tokens)] = tokens
    if parallel:
        with tmp_process_group():
            mlink(prompt)
    return 0


def write_or_close(prompt: torch.Tensor, fin: torch.Tensor, gen: LLaMA, t: float, p: float, stop_str: str, prob_mode: bool, prob_prev_pos: int, prob_top_k: int = 0):
    if prompt[0, 0] == fin:
        return None
    else:
        result, info = gen.generate(prompt, max_gen_len=SEQ_LEN, temperature=t, top_p=p, stop_str=stop_str, prob_mode=prob_mode, prob_prev_pos=prob_prev_pos, prob_top_k=prob_top_k)
        prompt = prompt.fill_(gen.tokenizer.pad_id)
        return result, info


@contextmanager
def tmp_process_group(backend="nccl"):
    # Source: https://github.com/pytorch/elastic/blob/master/examples/imagenet/main.py
    new_pg = dist.new_group(backend=backend)
    try:
        yield new_pg
    finally:
        dist.destroy_process_group(new_pg)


def mlink(prompt: torch.Tensor):
    # Source: https://h-huang.github.io/tutorials/intermediate/dist_tuto.html
    rank = dist.get_rank()
    size = dist.get_world_size()
    send_buff = prompt.clone()
    recv_buff = prompt.clone()

    if rank == 0:
        send_reqs = []
        for i in range(1, size):
            send_reqs.append(dist.isend(send_buff, i))
        for req in send_reqs:
            req.wait()
    else:
        dist.recv(recv_buff, 0)
    prompt[:] = recv_buff[:]


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = SEQ_LEN,
    max_batch_size: int = 1,
):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator = load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )
    fin, prompt = make_buffers(max_seq_len, generator.tokenizer.pad_id, local_rank)
    dist.barrier()
    while True:
        tokenize_or_wait(prompt=prompt, fin=fin, gen=generator, gpu=local_rank, parallel=world_size > 1)
        settings = json.load(open("settings.json", "r"))
        result, info = write_or_close(prompt=prompt, fin=fin, gen=generator, \
                                      p=settings["top_p"], 
                                      t=settings["temperature"], \
                                      stop_str=settings["stop_str"], 
                                      prob_mode=settings["prob_mode"], 
                                      prob_prev_pos=settings["prob_prev_pos"],
                                      prob_top_k=settings["prob_top_k"])
        dist.barrier()
        if result is None:
            break
        else:
            if local_rank == 0:
                res = result[0]
                result_json = json.dumps({"result": res, "info": info})
                print("[LLaMa] Output: ", res[:20])  # debug
                #create a file named result
                assert not os.path.exists("result"), "[LLaMa] result file already exists"
                with open("result", "w") as f_result:
                    f_result.write(result_json)
                    f_result.close()
                print("[LLaMa] result file created with content: ", res[:20])
        dist.barrier()
    return


if __name__ == "__main__":
    if os.path.exists("result"):
        os.remove("result")
    fire.Fire(main)
