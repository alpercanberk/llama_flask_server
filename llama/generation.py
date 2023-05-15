# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import List

import torch
import time
import numpy as np

from llama.tokenizer import Tokenizer
from llama.model import Transformer
from llama.model import TransformerBlock


class LLaMA:
    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def tokenize(self, prompt: str):
        # print(prompt)  # For testing purpose only
        return torch.tensor(self.tokenizer.encode(prompt, bos=True, eos=False), dtype=torch.long)

    def generate(
        self,
        prompt: torch.Tensor,
        max_gen_len: int,
        stop_str :str = None,
        temperature: float = 0.8,
        top_p: float = 0.95,
        prob_mode: bool = False,
        prob_prev_pos: int = 0,
        prob_top_k: int = 15,
    ) -> List[str]:
        
        if prob_mode:
            print("[ProbMode] Prob prev pos: ", prob_prev_pos)
        
        prompt_len = prompt.size(1) - (prompt == self.tokenizer.pad_id).view(-1).sum().item()
        input_text_mask = prompt != self.tokenizer.pad_id
        start_pos = prompt_len
        prev_pos = 0
        len_stop_str = len(self.tokenize(stop_str))+1

        chosen_log_probs = []
        chosen_decode_tokens = []

        top_k_log_probs = []
        top_k_decode_tokens = []

        entropy = []
        top_k_entropy = []

        layer_activations = []

        start_time = time.time()

        for cur_pos in range(start_pos, max_gen_len):
            
            if prob_mode:

                if prob_prev_pos == 0:
                    logits = self.model.forward_with_probs(prompt[:, prob_prev_pos:cur_pos], prob_prev_pos)
                else:
                    logits = self.model.forward_with_probs(prompt[:, prob_prev_pos:cur_pos], prob_prev_pos)

                #logits has shape (1, #tokens, vocab_size)
               
                for i in range(logits.shape[1]):
                    # Get the probability of the correct token
                    prob = torch.softmax(logits[0, i], dim=-1)[prompt[0, prob_prev_pos + i].item()].item()
                    # Add the log probability to the sum

                    chosen_log_probs.append(torch.log(torch.tensor(prob)).item())
                    chosen_decode_tokens.append(self.tokenizer.decode(prompt[:, prob_prev_pos + i].tolist()))

                    all_probs = torch.softmax(logits[0, i], dim=-1)
                    #get the top k tokens
                    top_k_tokens = torch.topk(all_probs, prob_top_k)
                    #get the corresponding decoding of the top k tokens
                    decoded_top_k_tokens = [self.tokenizer.decode(torch.tensor([token]).tolist()) for token in top_k_tokens.indices.tolist()]
                    #get log probabilities of the top k tokens
                    # top_k_log_probs = torch.log(top_k_tokens.values)

                    top_k_log_probs.append(torch.log(top_k_tokens.values).tolist())
                    top_k_decode_tokens.append(decoded_top_k_tokens)


                    #entropy of the whole distribution
                    entropy.append(-torch.sum(all_probs * torch.log(all_probs)).tolist())
                    #entropy of the top k tokens
                    top_k_entropy.append(-torch.sum(top_k_tokens.values * torch.log(top_k_tokens.values)).tolist())

                    # print(top_k_tokens)
                    # print(decoded_top_k_tokens)
                    # print(top_k_log_probs)
                    # print(entropy)
                    # print(top_k_entropy)

                # for i, layer in enumerate(self.model.layers):
                #     if isinstance(layer, TransformerBlock) and i == 39:
                #         # print("hi", i)
                #         # print("cache_k:", layer.attention.cache_k.shape, len(entropy))

                #         cache_k = layer.attention.cache_k[:, :len(entropy)]
                #         cache_k = torch.round(cache_k * 1000) / 1000

                #         cache_v = layer.attention.cache_v[:, :len(entropy)]
                #         cache_v = torch.round(cache_v * 1000) / 1000

                #         #make cache_k floats have much much lower precision

                #         # cache_v = layer.attention.cache_v[:, :len(entropy)].half().tolist()

                #         # print("cache_v:", layer.attention.cache_v.shape, layer.attention.cache_v)
                #         layer_activations.append({'cache_k': cache_k.tolist(), 'cache_v': cache_v.tolist()})

                    #pair the top k tokens with their log probabilities
                    # sequence_of_top_k_tokens = list(zip(sequence_of_top_k_tokens, top_k_log_probs.tolist()))
                    # for i in range(len(sequence_of_top_k_tokens)):
                    #     print(f"Top {i+1} tokens: {sequence_of_top_k_tokens[i][0]} with log probability {sequence_of_top_k_tokens[i][1]}")
                break
                

            logits = self.model.forward(prompt[:, prev_pos:cur_pos], prev_pos)

     
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], prompt[:, cur_pos], next_token
            )

            prompt[:, cur_pos] = next_token
            prev_pos = cur_pos

            if stop_str is not None:
                assert len(prompt) == 1, "Batch generation is not supported with this setting."
                potential_stop_str = prompt[0, max(start_pos, cur_pos-len_stop_str):cur_pos]
                # potential_stop_str = prompt[0, max(0, cur_pos-len_stop_str):cur_pos]
                if stop_str in self.tokenizer.decode(potential_stop_str.tolist()):
                    break

        end_time = time.time()
        print(f"[Generate] Generation time : {(end_time - start_time)}")
        
        info = {
            'chosen_log_probs': chosen_log_probs,
            'chosen_decode_tokens': chosen_decode_tokens,
            'top_k_log_probs': top_k_log_probs,
            'top_k_decode_tokens': top_k_decode_tokens,
            'entropy': entropy,
            'top_k_entropy': top_k_entropy,
            'layer_activations': layer_activations
        }
        # Decoding
        decoded = []
        t = prompt[0, :start_pos + max_gen_len].tolist()
        # cut to eos tok if any
        try:
            t = t[:t.index(self.tokenizer.eos_id)]
        except ValueError:
            pass
        # Don't know what is -1 and how it's generated
        t = [s for s in t if s != -1]
        try:
            decoded.append(self.tokenizer.decode(t))
            if not prob_mode:
                info['tokens'] = [self.tokenizer.decode(s) for s in t]
            # print(f"Tokens after decoding: {info['tokens']}")
            # print(f"Decoded: {decoded}")
            return decoded, info
        except Exception as e:
            print(f"Tokenization error: {e}")
            print("Trying per symbol tokenization.")
        # print(f"Tokens before decoding: {t}")
        for st in t:
            try:
                rt = self.tokenizer.decode(st)
                decoded.append(rt)
            except IndexError:
                print(f"Tokenizer error on token {st}")

        return decoded, info


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

