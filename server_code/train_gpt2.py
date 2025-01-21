import os
import sys
with open(sys.argv[0]) as f:
    code = f.read()
import uuid
import math
import glob
import struct
import inspect
from contextlib import nullcontext
from dataclasses import dataclass
from functools import wraps

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.distributed as dist
from hellaswag import iterate_examples, render_example
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

# using a global to toggle flash-attention
# wouldn't golbals be hard to change if this is imported as a module?
FLASH = 0

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        if FLASH:
            # flashattention
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # manual implementation of attention
            # this materializes the large (T,T) matrix for all the queries and keys
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.LLMC_SKIP_INIT = 1 # don't init this one, we will tie weights
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights, use a torch rng object to be very careful
        self.init_rng = torch.Generator()
        self.init_rng.manual_seed(42)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # apply special scaled init to the residual projections, per GPT-2 paper
            std = 0.02 if not hasattr(module, 'LLMC_RESIDUAL_SCALE_FLAG') else 0.02/math.sqrt(2 * self.config.n_layer)
            # we want to skip initializing lm_head, which shares parameters with wte
            # and wte was already initialized down below during the Embedding init
            if not hasattr(module, 'LLMC_SKIP_INIT'):
                torch.nn.init.normal_(module.weight, mean=0.0, std=std, generator=self.init_rng)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02, generator=self.init_rng)

    def forward(self, idx, targets=None, return_logits=True, full_logits=False):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        elif full_logits:
            logits = self.lm_head(x)
            loss = None
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss


    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, zero_stage):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            logger.info(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            logger.info(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        if master_process:
            logger.info(f"using fused AdamW: {use_fused}")
        if zero_stage == 1:
            if master_process:
                logger.info("using ZeroRedundancyOptimizer")
            optimizer = ZeroRedundancyOptimizer(**optim_groups[0], optimizer_class=torch.optim.AdamW,
                                                lr=learning_rate, betas=betas, fused=use_fused)
            optimizer.add_param_group(optim_groups[1])
        else:
            if master_process:
                logger.info("using regular AdamW")
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused)
        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


    @torch.no_grad()
    def get_most_likely_row(self, tokens, mask):
        """
        torch.compile throws an error when model() is called with tensors of
        dynamic dimension.
        I think its picking up on the fact that most of the time model() has
        fixed dimensions, but that is not the case when choosing the most likely
        completion. The method model.generate does allow for dynamic dimensions,
        so we do the same here.

        or is it that model() in particular need fixed size tensors?

        is it woth it to compute the logits for multiple questions at the same
        time?
        """
        torch._dynamo.mark_dynamic(tokens, 1)
        torch._dynamo.mark_dynamic(mask, 1)
        logits, _ = self(tokens, full_logits=True)

        shift_logits = (logits[..., :-1, :]).contiguous()
        # logits B T-1 C
        shift_tokens = (tokens[..., 1:]).contiguous()
        # tokens B T-1
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        # BC T-1
        flat_shift_tokens = shift_tokens.view(-1)
        #print(flat_shift_logits.shape, flat_shift_tokens.shape)
        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
        shift_losses = shift_losses.view(tokens.size(0), -1)
        # now get the average loss just for the completion region (where mask == 1), in each row
        shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
        masked_shift_losses = shift_losses * shift_mask
        # sum and divide by the number of 1s in the mask
        sum_loss = masked_shift_losses.sum(dim=1)
        avg_loss = sum_loss / shift_mask.sum(dim=1)
        # now we have a loss for each of the 4 completions
        # the one with the lowest loss should be the most likely
        pred_norm = avg_loss.argmin().item()
        return pred_norm

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240520:
        if master_process:
            logger.info("ERROR: magic number mismatch in the data .bin file!")
            logger.info("---> HINT: Are you passing in a correct file with --input_bin?")
            logger.info("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
            logger.info("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2] # number of tokens (claimed)
    return ntok # for now just return the number of tokens

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += shard_ntok
        self.ntok_total = ntok_total
        if master_process:
            logger.info(f"DataLoader: total number of tokens: {ntok_total:,} across {len(self.files)} files")

        # kick things off
        self.current_shard = None
        self.reset()

    def reset(self):
        # we're being a bit clever here: if we already had shard 0 loaded,
        # then don't do the work to reload it, just reset the pointer
        if self.current_shard != 0:
            self.current_shard = 0
            self.tokens = _load_data_shard(self.files[self.current_shard])
        self.current_position = self.process_rank * self.B * self.T

    def advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the start pointer in current shard
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds advance the shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x, y
    def state_dict(self, for_next=False):
        # The processes have coniguous data shards, the master process has the
        # first one.
        # If you are saving after iterating over a minibatch, then advance
        # current position accordingly
        current_shard = self.current_shard
        current_position = self.current_position
        if for_next and (self.current_position + (B * T * self.num_processes + 1) > len(self.tokens)):
            # We update shard the entire page num_processes * B * T tokens
            # doesn't fit in a single shard
            # When we load, there is no need to check if our batch is
            # going to fit inside a shard, as this is guaranteed
            # IF B * T *num_processes <= toc per shard
            current_shard = (self.current_shard + 1) % len(self.files)
            current_position = self.process_rank * self.B * self.T
        d = OrderedDict(
            current_position=current_position,
            current_shard=current_shard,
            files = self.files,
        )
        return d
    def load_state_dict(self, state_dict):
        self.current_position = state_dict['current_position'] + self.B * self.T * self.process_rank
        self.current_shard = state_dict['current_shard']
        self.files = state_dict['files']
        self.tokens = _load_data_shard(self.files[self.current_shard])

@torch.no_grad()
def pad_vocab(tensor, multiple=128, value=0):
    """
    The dimension of the vocab size in GPT-2 is 50,257
    which is unfortunately a very unfriendly number for a lot of
    matrix operations on the GPU. So we pad it to the nearest
    friendlier multiple, e.g. 50,304 if multiple=128 when we
    export the weights into C land. This is a NOOP algorithmically
    and is only done to make the tensor operations more efficient.
    """
    assert tensor.ndim == 2
    V, C = tensor.shape
    assert V == 50257, "just being defensive here"
    # calculate padded vocab size by rounding up to nearest multiple
    Vp = ((V + multiple - 1) // multiple) * multiple
    # pad the tensor
    pad_rows = Vp - V
    padded = tensor if pad_rows == 0 else F.pad(tensor, (0, 0, 0, pad_rows), value=value)
    assert padded.shape == (Vp, C)
    return padded

# -----------------------------------------------------------------------------
# int main

if __name__ == "__main__":
    import time
    import argparse
    import tiktoken
    import logging
    from pythonjsonlogger import jsonlogger
    import csv
    import json
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo


    # default settings will overfit a tiny batch of data
    # and save model weights and debug state to disk on the first iteration
    parser = argparse.ArgumentParser()
    # file system input / output
    parser.add_argument("--input_bin", type=str, default="dev/data/tinyshakespeare/tiny_shakespeare_val.bin", help="input .bin to train on")
    parser.add_argument("--input_val_bin", type=str, default="", help="input .bin to eval validation loss on")
    # TODO: rename output dir ?
    parser.add_argument("--output_dir", type=str, default="", help="output directory to which to write logs and checkpoints")
    parser.add_argument("--model", type=str, default="gpt2", help="gpt2|gpt2-medium|gpt2-large|gpt2-xl|d12|d24|d36|d48")
    # token layout for each step of the optimization
    parser.add_argument("--batch_size", type=int, default=4, help="batch size, in units of #batch dimensions")
    parser.add_argument("--sequence_length", type=int, default=64, help="sequence length")
    parser.add_argument("--total_batch_size", type=int, default=256, help="total desired batch size, in units of #tokens")
    # workload (number of steps)
    parser.add_argument("--num_iterations", type=int, default=10, help="number of iterations to run")
    parser.add_argument("--inference_only", type=int, default=0, help="only run inference")
    parser.add_argument("--checkpoint", type=str, help="TODO")
    # optimization
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate warmup iterations")
    parser.add_argument("--warmup_iters", type=int, default=0, help="learning rate warmup iterations")
    parser.add_argument("--learning_rate_decay_frac", type=float, default=1.0, help="learning rate warmup iterations")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="maximum gradient magnitude")
    # evaluation
    parser.add_argument("--val_max_steps", type=int, default=20, help="how many batches of val to average?")
    parser.add_argument("--val_loss_every", type=int, default=256, help="how often to get the validation loss?")
    parser.add_argument("--sample_every", type=int, default=256, help="how often to sample from the model?")
    parser.add_argument("--hellaswag_every", type=int, default=256, help="how often to eval on HellaSwag?")
    # debugging
    parser.add_argument("--overfit_single_batch", type=int, default=1, help="overfit just one batch of data")
    # numerics
    parser.add_argument("--tensorcores", type=int, default=0, help="use tensorcores")
    # memory management
    parser.add_argument("--device", type=str, default="", help="by default we autodetect, or set it here")
    parser.add_argument("--compile", type=int, default=0, help="torch.compile the model")
    parser.add_argument("--flash", type=int, default=0, help="use flash attention")
    parser.add_argument("--dtype", type=str, default="float32", help="float32|float16|bfloat16")
    parser.add_argument("--zero_stage", type=int, default=0, help="zero redundancy optimizer stage (0/1/2/3)")
    parser.add_argument("--save_every", type=int, default=256, help="how often to checkpoint")
    args = parser.parse_args()

    # args error checking and convenience variables
    B, T = args.batch_size, args.sequence_length
    assert 1 <= T <= 1024
    assert args.dtype in {"float32", "float16", "bfloat16"}
    assert args.model in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "d12", "d24", "d36", "d48"}

    # set up DDP (distributed data parallel). torchrun sets this env variable
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = 0 # each process gets the exact same seed
        zero_stage = args.zero_stage
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        zero_stage = 0
        ddp_world_size = 1
        master_process = True
        seed_offset = 0
        # select the device
        if args.device:
            # provided explicitly by the user
            device = args.device
        else:
            # attempt to autodetect the device
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
    device_type = 'cuda' if 'cuda' in device else 'cpu'

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
    # begin logging
    if master_process:
        logfile = None
        if args.checkpoint:
            run_id = checkpoint['run_id']
        else:
            run_id = str(uuid.uuid4())
        logdir = f'logs/{run_id}/'
        os.makedirs(logdir, exist_ok=True)
        # logeamos
        mexico_tz = ZoneInfo('America/Mexico_City')

        logger = logging.getLogger("gpt_training")
        logger.setLevel(logging.INFO)

        stat = logging.getLogger("estadisticas")
        stat.setLevel(logging.INFO)

        # configuracion del manejo en la consola

        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s' )
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # configuracion para json
        json_log_file = f"{logdir}training_log.jsonl"
        file_handler = logging.FileHandler(json_log_file)
        logging.Formatter.formatTime = (
            lambda self, record, datefmt=None:
                datetime
                .fromtimestamp(record.created, timezone.utc)
                .astimezone(mexico_tz)
                .isoformat(sep="T", timespec="milliseconds")
        )
        file_handler.setLevel(logging.DEBUG)
        json_formatter = jsonlogger.JsonFormatter(
            fmt='%(asctime)s %(levelname)s %(message)s',
        )
        file_handler.setFormatter(json_formatter)
        logger.addHandler(file_handler)

        writer = SummaryWriter(f'runs/{run_id}')

        logger.info(f"using device: {device}")
        logger.info(f"Running pytorch {torch.version.__version__}")
        if args.checkpoint:
            logger.info(f"Resuming training from checkpoint {args.checkpoing}")
        else:
            logger.info("Training from scratch")

    # calculate gradient accumulation from the desired total batch size and the current run configuration
    tokens_per_fwdbwd = B * T * ddp_world_size
    assert args.total_batch_size % tokens_per_fwdbwd == 0
    grad_accum_steps = args.total_batch_size // tokens_per_fwdbwd
    if master_process:
        logger.info(f"total desired batch size: {args.total_batch_size}")
        logger.info(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    # set up a context manager following the desired dtype and device
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

    # rng / reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # set the torch precision mode to use TensorFloat32 (TF32) for matmuls
    # docs https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    if args.tensorcores:
        torch.set_float32_matmul_precision('high')

    # turn on/off flash attention
    assert args.flash in {0, 1}
    FLASH = args.flash

    # init (and write) the tokenizer
    enc = tiktoken.get_encoding("gpt2")
    # init the model
    if args.checkpoint:
        model_config = checkpoint['config']
    elif args.model[0] == "d":
        model_config = {
            "d12": GPTConfig(block_size=1024, vocab_size=50257, n_layer=12, n_head=12, n_embd=768),
            "d24": GPTConfig(block_size=1024, vocab_size=50257, n_layer=24, n_head=16, n_embd=1024),
            "d36": GPTConfig(block_size=1024, vocab_size=50257, n_layer=36, n_head=20, n_embd=1280),
            "d48": GPTConfig(block_size=1024, vocab_size=50257, n_layer=48, n_head=25, n_embd=1600),
        }[args.model]
    else:
        raise NotImplementedError()
    model = GPT(model_config)
    if args.checkpoint:
        model.load_state_dict(checkpoint['model'])
    model.train()
    model.to(device)
    if args.compile:
        if hasattr(config, "coordinate_descent_tuning"):
            config.coordinate_descent_tuning = True # suggested by @Chillee
        if master_process:
            logger.info("compiling the model...")
        model = torch.compile(model)

    # -------------------------------------------------------------------------
    # Our own version of a simple DistributedDataLoader

    # load tokens
    train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
    # TODO:
    if args.checkpoint:
        train_loader.load_state_dict(checkpoint['train_loader'])

    val_loader = None
    if args.input_val_bin:
        val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)

    # -------------------------------------------------------------------------
    # main training loop

    # here we wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

    # init the optimizer
    optimizer = raw_model.configure_optimizers(weight_decay=args.weight_decay,
                                               learning_rate=args.learning_rate, betas=(0.9, 0.95),
                                               device_type=device, zero_stage=zero_stage)
    if args.checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        min_lr = args.learning_rate * args.learning_rate_decay_frac
        # 1) linear warmup for warmup_iters steps
        if it < args.warmup_iters:
            return args.learning_rate * (it+1) / args.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > args.num_iterations:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - args.warmup_iters) / (args.num_iterations - args.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        return min_lr + coeff * (args.learning_rate - min_lr)


    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    norm = -1.0   # dummy value to print in inference-only mode
    start = 0
    if args.checkpoint:
        start = checkpoint['step'] + 1


    for step in range(start, args.num_iterations + 1):
        last_step = (step == args.num_iterations)

        # once in a while evaluate the validation dataset
        if (args.val_loss_every > 0 \
            and (step % args.val_loss_every == 0 or last_step)) \
            and (val_loader is not None):
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss = 0.0
                for _ in range(args.val_max_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with ctx:
                        _, loss = model(x, y, return_logits=False)
                    val_loss += loss.item()
                val_loss /= args.val_max_steps
            # log to console and to file
            if master_process:
                logger.info(f"val_loss {val_loss}")
                writer.add_scalar('val_loss', val_loss, step)

                with open(f"{logdir}val_loss.csv", mode='a', newline='') as f:
                    csv_writer = csv.writer(f)

                    # If empty
                    f.seek(0, 2)  # Move to the end
                    if f.tell() == 0:
                        csv_writer.writerow(['iso8601time', 'unixtime', 'step', 'val_loss'])
                    timestamp_iso = datetime.now(mexico_tz).isoformat()
                    timestamp_unix = time.time()
                    csv_writer.writerow([timestamp_iso, timestamp_unix, step, val_loss])

        # once in a while perform model inference on the master process
        if ((args.sample_every > 0
            and (step % args.sample_every == 0 or last_step))
            and master_process):
            model.eval()
            # before we end, let's also do one round of inference
            # we'll kick off the generation with "<|endoftext|>", which designates the start of a new sequence
            start_ids = [enc.eot_token]
            xg = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
            max_new_tokens = 32
            temperature = 1.0
            top_k = 40
            yg = raw_model.generate(xg, max_new_tokens, temperature=temperature, top_k=top_k)
            logger.info('---------------')
            # https://github.com/karpathy/build-nanogpt/pull/80/files
            # TODO: log the ORIGINAL generation, the pached one should only be used for decoding
            yg_for_decoding = yg.clone()
            yg_for_decoding[yg_for_decoding > enc.max_token_value] = enc.eot_token
            generated_ids = yg[0].tolist()
            generated_text = enc.decode(yg_for_decoding[0].tolist())
            logger.info(generated_text)
            logger.info('---------------')

            with open(f'{logdir}text_generations.jsonl', 'a', encoding='utf-8') as f:
                json_line = json.dumps(dict(
                    iso8601time = datetime.now(mexico_tz).isoformat(),
                    unixtime = time.time(),
                    step = step,
                    start_ids = start_ids,
                    generated_ids = generated_ids,
                    start_text = enc.decode(start_ids),
                    generated_text = generated_text,
                ))
                f.write(json_line + '\n')


        # once in a while evaluate hellaswag
        if ((args.hellaswag_every > 0
            and (step % args.hellaswag_every == 0 or last_step))
            and master_process):
            model.eval()
            num_correct_norm = 0
            num_total = 0
            for i, example in enumerate(iterate_examples("val")):
                if i % ddp_world_size != ddp_rank:
                    continue
                _, tokens, mask, label = render_example(example)
                tokens = tokens.to(device)
                mask = mask.to(device)
                pred_norm = raw_model.get_most_likely_row(tokens, mask)
                num_total += 1
                num_correct_norm += int(pred_norm == label)
            if ddp:
                num_total = torch.tensor(num_total, dtype=torch.long, device=device)
                num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
                dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            acc_norm = num_correct_norm / num_total
            if master_process:
                logger.info(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
                writer.add_scalar('hellaswag_acc', acc_norm, step)
                with open(f"{logdir}hellaswag_acc.csv", mode='a', newline='') as f:
                    csv_writer = csv.writer(f)

                    # If empty
                    f.seek(0, 2)  # Move to the end
                    if f.tell() == 0:
                        csv_writer.writerow(['iso8601time', 'unixtime', 'step', 'hellaswag_acc'])
                    timestamp_iso = datetime.now(mexico_tz).isoformat()
                    timestamp_unix = time.time()
                    csv_writer.writerow([timestamp_iso, timestamp_unix, step, acc_norm.item()])


        # checkpointing
        if (last_step or (args.save_every > 0 and step %  args.save_every == 0)):
            if ddp:
                optimizer.consolidate_state_dict(to=0)
            if master_process:
                model_sd = raw_model.state_dict()
                if args.compile:
                    renamed_model_sd = OrderedDict()
                    for k, v in model_sd.items():
                        if k.startswith('_orig_mod.'):
                            k = k.lstrip('_orig_mod.')
                            renamed_model_sd[k] = v
                        else:
                            logger.error('A param of the compiled model does not start with _orig_mod.')
                    model_sd = renamed_model_sd
                # we need to consolidate the state of the dataloader somehow(?)
                log = dict(step=step, model=model_sd, optimizer=optimizer.state_dict(), run_id=run_id, train_loader=train_loader.state_dict(for_next=True))
                torch.save(log, f'{logdir}/state_step_{step}.pt')



        # bit confusing: we want to make sure to eval and sample on 0th iteration
        # but also after the very last iteration. so we loop for step <= num_iterations
        # instead of just < num_iterations (one extra due to <=), only to do
        # the validation/sampling one last time, and then we break right here as we're done.
        if last_step:
            break

        # --------------- TRAINING SECTION BEGIN -----------------
        model.train()

        # this is here so benchmarking and evaluation doesn't affect the timing
        # wait on the CPU for all device work to end so we get accurate per-iteration timings below
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()

        optimizer.zero_grad(set_to_none=True)
        # if we are trying to overfit a single batch, we reset the loader here
        if args.overfit_single_batch:
            train_loader.reset()
        # micro-batch loop where we do gradient accumulation to reach desired total batch size
        lossf = 0.0 # for getting the mean loss (as simple float) over the accumulation steps
        for micro_step in range(grad_accum_steps):
            # fetch a batch
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            if ddp:
                # we want only the last micro-step to sync grads in a DDP model
                # the official way to do this is with model.no_sync(), but that is a
                # context manager that bloats the code, so we just toggle this variable
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            # forward pass
            with ctx:
                _, loss = model(x, y, return_logits=False)
                # we have to scale the loss to account for gradient accumulation,
                # because the gradients just add on each successive backward().
                # addition of gradients corresponds to a SUM in the objective, but
                # instead of a SUM we want MEAN, so we scale the loss here
                loss = loss / grad_accum_steps
                lossf += loss.detach() # keep track of the mean loss
            # backward pass
            if not args.inference_only:
                loss.backward()
        if ddp:
            dist.all_reduce(lossf, op=dist.ReduceOp.AVG)
        lossf = lossf.item()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        # determine and set the learning rate for this iteration
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # step the optimizer
        optimizer.step()
        # --------------- TRAINING SECTION END -------------------
        # everything that follows now is just diagnostics, prints, logging, etc.

        # wait on the CPU for all device work to end so we get accurate per-iteration timings below
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        # time and print
        t1 = time.time()
        tokens_per_second = grad_accum_steps * ddp_world_size * B * T / (t1-t0)

        # step logging
        if master_process:
            logger.info(f"step {step+1:4d}/{args.num_iterations} | train loss {lossf:.6f} | norm {norm:.4f} | lr {lr:.2e} | ({(t1-t0)*1000:.2f} ms | {tokens_per_second:.0f} tok/s)")
            step_stats = dict(
                train_loss = lossf,
                norm = norm,
                lr = lr,
                time_micro_seconds = t1 - t0,
                tokens_per_second = tokens_per_second
            )
            for k, v in step_stats.items():
                writer.add_scalar(k, v, step)

            # How long does this take?
            if step % args.save_every == 0:
                for name, param in raw_model.named_parameters():
                    # torch compile adds the previx _orig_mod.
                    if args.compile:
                        if name.startswith('_orig_mod.'):
                            name = name.lstrip('_orig_mod.')
                        else:
                            logger.error('A named param of the compiled model does not start with _orig_mod.')
                    writer.add_histogram(name, param, step)

            with open(f"{logdir}step_stats.csv", mode='a', newline='') as f:
                csv_writer = csv.writer(f)

                # If empty
                f.seek(0, 2)  # Move to the end
                if f.tell() == 0:
                    csv_writer.writerow(['iso8601time', 'unixtime', 'step', 'train_loss', 'norm', 'lr', 'time_micro_seconds', 'tokens_per_second'])
                timestamp_iso = datetime.now(mexico_tz).isoformat()
                timestamp_unix = time.time()
                csv_writer.writerow(
                    [timestamp_iso, timestamp_unix, step, float(lossf), float(norm), lr, t1-t0, tokens_per_second])

                writer.flush()

    if master_process:
        logger.info(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    # -------------------------------------------------------------------------
    # clean up nice
    if master_process:
        writer.close()
    if ddp:
        destroy_process_group()
