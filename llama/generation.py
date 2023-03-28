# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import List

import torch

from llama.tokenizer import Tokenizer
from llama.model import Transformer


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
    ) -> List[str]:
        
        prompt_len = prompt.size(1) - (prompt == self.tokenizer.pad_id).view(-1).sum().item()
        input_text_mask = prompt != self.tokenizer.pad_id
        start_pos = prompt_len
        prev_pos = 0
        len_stop_str = len(self.tokenize(stop_str))+1
        sequence_log_prob = 0
        sequence_log_perplexity = 0

        for cur_pos in range(start_pos, max_gen_len):
            
            if prob_mode:
                #create a new prompt by appending an EOS token as the last character of the prompt before the padding tokens
                sequence_log_prob = 0
                logits = self.model.forward_with_probs(prompt[:, prev_pos:cur_pos], prev_pos)
                #logits has shape (1, #tokens, vocab_size)
               
                for i in range(logits.shape[1]):
                    # Get the probability of the correct token
                    prob = torch.softmax(logits[0, i], dim=-1)[prompt[0, prev_pos + i].item()].item()
                    # Add the log probability to the sum
                    sequence_log_prob += torch.log(torch.tensor(prob))

                #compute perplexity
                sequence_log_perplexity = ((1 / logits.shape[1]) * sequence_log_prob).item()
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
        
        info = {
            "log_perplexity": sequence_log_perplexity,
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

