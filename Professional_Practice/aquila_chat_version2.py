# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import os
import torch
from torch.utils.data import Dataset
import gc
gc.collect()
torch.cuda.empty_cache()
from flagai.auto_model.auto_loader import AutoLoader
from flagai.data.tokenizer import Tokenizer
from flagai.env_args import EnvArgs
from flagai.env_trainer_v1 import EnvTrainer
import jsonlines
import numpy as np
import cyg_conversation as conversation_lib
from flagai.model.tools.lora.prepare_lora import lora_transfer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# You can input all parameters by the command line.
# For example: python train_env_trainer.py --epochs=300 --batch_size=4 --env_type=pytorch
env_args = EnvArgs(
    env_type="bmtrain",
    batch_size=1,
    gradient_accumulation_steps=1,
    lr=2e-4,
    weight_decay=1e-3,
    epochs=100,
    log_interval=10,
    eval_interval=5000,
    num_gpus=1,
    load_dir=None,
    pytorch_device=device,
    save_dir="checkpoints_aquila",
    checkpoint_activations=False,
    save_interval=5000,
    fp16=True,
    training_script=__file__,
)
env_args = env_args.parse_args()
#env_args.wandb = False

# overwrite
if env_args.yaml_config:
    import yaml
    file_data = open(env_args.yaml_config, 'r', encoding="utf-8").read()
    data = yaml.load_all(file_data,Loader=yaml.FullLoader)
    delattr(env_args, 'yaml_config')
    arg_dict = env_args.__dict__
    for subdata in data:
        for key, value in subdata.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
trainer = EnvTrainer(env_args)

# Trainer as Trigger
if not env_args.not_call_launch:
    import sys
    sys.exit(0)

print(f"Trainer effective env_args={env_args} local_rank={trainer.local_rank}", flush=True)

checkpoints = env_args.pre_load_dir

model_name = env_args.model_name

print('*'*20, "model_name", model_name, flush=True)


cache_dir = os.path.join(checkpoints, model_name)
print('*'*20, "cache_dir", cache_dir)
tokenizer = Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)
print('*'*20, "tokenizer", tokenizer)

# avoid sync loading models in case of Mem OOM
if env_args.bmt_async_load:
    import time
    time.sleep(10*60*(trainer.local_rank%4))


config_file = os.path.join(cache_dir, 'config.json')
from flagai.model.aquila_model import AQUILAModel
model = AQUILAModel.init_from_json(config_file=config_file, device=device, fp16=True) 
# print('*'*20, "model", model)

#lora
if env_args.lora:
    model = lora_transfer(model,env_args)
    model.print_trainable_parameters()

## bmt_pre_load
checkpoint_path = os.path.join(cache_dir, "pytorch_model.bin")
if env_args.bmt_pre_load:
    model.load_weights(checkpoint_path)

trainer.pre_train(model)

# print('*'*20, "model", model, flush=True)

assert env_args.enable_sft_dataset_dir is not None and \
        env_args.enable_sft_dataset_file is not None

cur_dir = env_args.enable_sft_dataset_dir
jsonl_data = os.path.join(cur_dir, env_args.enable_sft_dataset_file)
jsonl_data_val = None
if env_args.enable_sft_dataset_val_file is not None:
    jsonl_data_val = os.path.join(cur_dir, env_args.enable_sft_dataset_file)
max_seq_len = 2048

def read_file(jsonl_file):
    conversations = []
    with jsonlines.open(jsonl_file) as reader:
        for line in reader:
            conversations.append(line)
    return conversations
def formatsource(source):
    results=[]
    results.append(source['question'])
    textlist=source['answer'].split("**")
    for subtext in textlist:
        sublist=subtext.split("\n")
        for textinfo in sublist:
            results.append(textinfo)
    return results
def _add_signal4gsm(header, source, get_conversation=True):

    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    unknown_role = "unknown"  # use default unknown role
    roles = {
        "human": conversation_lib.default_conversation.roles[0],  # human role->human
        "gpt": conversation_lib.default_conversation.roles[1],  # gpt role->assistant
    }

    for sentence in source["conversations"]:
        sentence_from = sentence["from"].lower()
        sentence["value"] = (
            BEGIN_SIGNAL
            + roles.get(sentence_from, unknown_role)
            + ": "
            + sentence["value"]
            + END_SIGNAL
        )
        if get_conversation:
            conversation += sentence["value"]
    return conversation

class GSM_socratic(Dataset):
    def __init__(self, conversations, tokenizer,loss_on_prefix=True,maxlen=512):
        super(GSM_socratic, self).__init__()
        self.examples=conversations
        self.qns = [ex["question"] for ex in self.examples]
        self.ans = [ex["answer"] for ex in self.examples]
        self.encoded_qns = []
        self.encoded_ans = []
        for q in self.qns:
                encoded = tokenizer.encode_plus(q)
                self.encoded_qns.append(encoded)
        for q in self.ans:
                encoded = tokenizer.encode_plus(q)
                self.encoded_ans.append(encoded)
        self.loss_on_prefix = loss_on_prefix
        self.maxlen = maxlen

    def __getitem__(self, idx):
        qn_tokens = self.encoded_qns[idx]["input_ids"]
        ans_tokens = self.encoded_ans[idx]["input_ids"]
        pad_tokens = [0] * (self.maxlen - len(qn_tokens) - len(ans_tokens))
        tokens = qn_tokens + ans_tokens + pad_tokens
        mask = (
            ([int(self.loss_on_prefix)] * len(qn_tokens))
            + ([1] * len(ans_tokens))
            + ([0] * len(pad_tokens))
        )        
        tokens = torch.tensor(tokens)
        mask = torch.tensor(mask)
        output = {
            "input_ids": tokens,
            "labels": mask,
        }
        return output

    def __len__(self):
        return len(self.examples)
    
conversations = read_file(jsonl_data)
data_len = len(conversations)
#train_size = int(data_len * 0.95)
train_size = data_len
train_conversations = conversations[:train_size]

train_dataset = GSM_socratic(train_conversations,
                                        tokenizer=tokenizer,
                                        maxlen=max_seq_len)


valid_dataset = None
if jsonl_data_val is not None:
    conversations_val = read_file(jsonl_data_val)
    valid_dataset = GSM_socratic(conversations_val,
                                            tokenizer=tokenizer,
                                            maxlen=max_seq_len)

trainer.do_train(
    train_dataset=train_dataset,
    valid_dataset=valid_dataset,
    optimizer=None,
    rank_split=False)
