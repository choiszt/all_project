import json

import numpy as np


def read_local_dataset(path):
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            yield json.loads(line.strip())


def convert_example(example, tokenizer, data_args, is_test=True):
    """
    :param example:一个样本，可以是一个字典，包含"content"或"src"字段和"summary"或"tgt"字段，用于表示对话的问句、答句和历史对话；
    :paramtokenizer：一个tokenizer对象，用于将文本转换为模型输入格式；
    :param data_args：一个包含预定义参数的命名元组，包括:param src_length、tgt_length等参数；
    :param is_test：一个标志位，用于表示是否为测试集。
    """
    if "content" in example:
        query = example["content"]
        response = example["summary"]
    elif "src" in example:
        query = example["src"][0] if isinstance(example["src"], list) else example["src"]
        response = example["tgt"][0] if isinstance(example["tgt"], list) else example["tgt"]
    history = example.get("history", None)

    if history is None or len(history) == 0:
        prompt = query
    else:
        prompt = ""
        for i, (old_query, old_response) in enumerate(history):
            prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, old_response)
        prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)

    # dataset for evaluation
    if is_test:
        inputs = {
            **tokenizer(prompt, max_length=data_args.src_length, truncation=True, truncation_side="left"),
            "labels": tokenizer(response, max_length=data_args.tgt_length, truncation=True, truncation_side="right")[
                "input_ids"
            ],
        }
    # dataset for training
    else:
        src_ids = tokenizer(
            prompt,
            add_special_tokens=False,
            max_length=data_args.src_length - 1,
            truncation=True,
            truncation_side="left",
        )["input_ids"]
        tgt_ids = tokenizer(
            response,
            add_special_tokens=False,
            max_length=data_args.tgt_length - 2,
            truncation=True,
            truncation_side="right",
        )["input_ids"]

        input_ids = tokenizer.build_inputs_with_special_tokens(src_ids, tgt_ids)

        context_length = input_ids.index(tokenizer.bos_token_id)
        mask_position = context_length - 1

        attention_mask = np.tri(len(input_ids), len(input_ids))
        attention_mask[:, :context_length] = 1
        attention_mask = attention_mask[None, :, :]
        attention_mask = (attention_mask < 0.5).astype("int64")

        labels = [-100] * context_length + input_ids[mask_position + 1 :]

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    return inputs


def custom_instruction_convert_example(example, tokenizer, data_args, is_test=True):
    """
    :param example:一个样本，可以是一个字典，包含"content"或"src"字段和"summary"或"tgt"字段，用于表示对话的问句、答句和历史对话；
    :paramtokenizer：一个tokenizer对象，用于将文本转换为模型输入格式；
    :param data_args：一个包含预定义参数的命名元组，包括:param src_length、tgt_length等参数；
    :param is_test：一个标志位，用于表示是否为测试集。
    """
    instruction = ""
    input = ""
    response = ""
    if "instruction" in example and "output" in example:
        instruction = example["instruction"]
        response = example["output"]
    else:
        assert False, "instruction and output are not in the input dictionary."
    if "input" in example["input"]:
        input = example["input"]

    prompt = instruction + input

    # dataset for evaluation
    if is_test:
        inputs = {
            **tokenizer(prompt, max_length=data_args.src_length, truncation=True, truncation_side="left"),
            "labels": tokenizer(response, max_length=data_args.tgt_length, truncation=True, truncation_side="right")[
                "input_ids"
            ],
        }
    # dataset for training
    else:
        src_ids = tokenizer(
            prompt,
            add_special_tokens=False,
            max_length=data_args.src_length - 1,
            truncation=True,
            truncation_side="left",
        )["input_ids"]
        tgt_ids = tokenizer(
            response,
            add_special_tokens=False,
            max_length=data_args.tgt_length - 2,
            truncation=True,
            truncation_side="right",
        )["input_ids"]

        input_ids = tokenizer.build_inputs_with_special_tokens(src_ids, tgt_ids)

        context_length = input_ids.index(tokenizer.bos_token_id)
        mask_position = context_length - 1

        attention_mask = np.tri(len(input_ids), len(input_ids))
        attention_mask[:, :context_length] = 1
        attention_mask = attention_mask[None, :, :]
        attention_mask = (attention_mask < 0.5).astype("int64")

        labels = [-100] * context_length + input_ids[mask_position + 1 :]

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    return inputs
