import argparse
import os
import time
import random
from functools import partial

import paddle
import numpy as np
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import UIE, AutoTokenizer
from paddlenlp.utils.log import logger
import paddle.nn.functional as F

import sys;sys.path.append("/mnt/ve_share/liushuai/PaddleNLP/applications/sentiment_analysis/unified_sentiment_extraction")
from evaluate import evaluate
from applications.sentiment_analysis.unified_sentiment_extraction.nlp课设.utils import convert_example, create_data_loader, reader, set_seed
from paddlenlp.metrics import SpanEvaluator

model = UIE.from_pretrained(args.model)
tokenizer = AutoTokenizer.from_pretrained(args.model)

def eval_distill():
    # 使用test进行训练
    logger.info("start to test...")
    test_ds = load_dataset(reader, data_path=args.test_path, max_seq_len=args.max_seq_len, lazy=False)
    test_data_loader = create_data_loader(test_ds, mode="test", batch_size=args.batch_size, trans_fn=trans_fn)

    if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        logger.info("load model from path: {}".format(args.init_from_ckpt))
        distill_logger.info("load model from path: {}".format(args.init_from_ckpt))
        state_dict = paddle.load(args.init_from_ckpt)
        model.set_dict(state_dict)

    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    # 设置优化器
    optimizer = paddle.optimizer.AdamW(
        learning_rate=args.learning_rate, parameters=student_model.parameters())
    metric = SpanEvaluator()
    criterion = paddle.nn.BCELoss()

    test_time = time.time()
    precision, recall, f1 = evaluate(student_model, metric, test_data_loader)
    logger.info("Test model, cost time: %d s" % (time.time() - test_time))
    logger.info("Evaluation precision: %.5f, recall: %.5f, F1: %.5f" % (precision, recall, f1))

if __name__ == "__main__":
    # yapf: disable
    parser = argparse.ArgumentParser()


    