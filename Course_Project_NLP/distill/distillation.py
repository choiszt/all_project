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
from visualdl import LogWriter
import logging

# 用于visualDL的logger
# 创建 LogWriter 对象，指定 logdir 参数，如果指定路径不存在将会创建一个文件夹
log_writer = LogWriter(logdir='../log/distill_log')

distill_logger = logging.getLogger("Distill_logger")
distill_logger.setLevel(logging.DEBUG)
# 创建文件处理器
file_handler = logging.FileHandler('./log/distill.log')
file_handler.setLevel(logging.DEBUG)
# 创建格式化器
formatter = logging.Formatter('[%(asctime)s] %(name)s [%(levelname)s]: %(message)s')
file_handler.setFormatter(formatter)
# 将文件处理器添加到logger
distill_logger.addHandler(file_handler)


def train_student(student_model, teacher_model, train_loader, optimizer, epoch, criterion, soft_weight, global_step, rank):
    """

    :param student_model, teacher_model: student & teacher model
    :param train_loader: training data loader
    :param optimizer: optimizer
    :epoch: training epoch
    :cirterion: used to calculate loss
    :soft_weight: loss = soft_weight * soft_loss + (1 - soft_weight) * hard_loss 
    :global step: step of train
    :rank: paddle.distributed.get_rank()
    :return: global step
    """
    student_model.train()
    teacher_model.eval()
    hard_list = []
    soft_list = []
    total_losses = []
    tic_train = time.time()
    for batch in train_loader:
        input_ids, token_type_ids, att_mask, pos_ids, start_ids, end_ids = batch
        # 硬标签
        start_ids_stu = paddle.cast(start_ids, "float32")
        end_ids_stu = paddle.cast(end_ids, "float32")
        with paddle.no_grad():
            # 教师模型的预测，用于生成软标签
            start_prob_teach, end_prob_teach = teacher_model(input_ids, token_type_ids, att_mask, pos_ids)
            # 进行温度缩放，获得软标签
            start_probs = F.softmax(start_prob_teach / args.temperature, axis=-1)
            end_probs = F.softmax(end_prob_teach / args.temperature, axis=-1)
            # 将概率值转换为 0 到 1 的范围
            start_probs = paddle.clip(start_probs, min=1e-6, max=1-1e-6)
            end_probs = paddle.clip(end_probs, min=1e-6, max=1-1e-6)
            # logits转换
            start_probs = paddle.log(start_probs / (1 - start_probs))
            end_probs = paddle.log(end_probs / (1 - end_probs))
            # 生成硬标签
            start_ids_teach = paddle.cast(start_prob_teach, "float32")
            end_ids_teach = paddle.cast(end_prob_teach, "float32")

        # 学生模型的预测
        start_prob_stu, end_prob_stu = student_model(input_ids, token_type_ids, att_mask, pos_ids)

        soft_loss = (criterion(start_prob_stu, start_ids_teach)
            + criterion(end_prob_stu, end_ids_teach)) / 2.0
        hard_loss = (criterion(start_prob_stu, start_ids_stu)
            + criterion(end_prob_stu, end_ids_stu)) / 2.0
        
        loss = soft_weight * soft_loss + (1 - soft_weight) * hard_loss 
        # 记录训练过程中的损失值、学习率、迭代次数等信息
        log_writer.add_scalar("train/loss", loss, global_step)
        log_writer.add_scalar("train/lr", optimizer.get_lr(), global_step)
        log_writer.add_scalar("train/iteration", global_step, global_step)
        # hard_list.append(hard_loss)
        # soft_list.append(soft_loss)
        # 计算硬损失
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        total_losses.append(float(loss))
        global_step += 1
        if global_step % args.logging_steps == 0 and rank == 0:
            time_diff = time.time() - tic_train
            loss_avg = sum(total_losses) / len(total_losses)
            logger.info(
                "global step %d, epoch: %d, loss: %.5f, speed: %.2f step/s"
                % (global_step, epoch, loss_avg, args.logging_steps / time_diff)
            )
            distill_logger.info(
                "global step %d, epoch: %d, loss: %.5f, speed: %.2f step/s"
                % (global_step, epoch, loss_avg, args.logging_steps / time_diff)
            )
            # 更新计时器
            tic_train = time.time()
            
    return student_model, global_step

def distill_knowledge():
    paddle.set_device(args.device)
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args.seed)

    # 加载教师模型和分词器
    teacher_model = UIE.from_pretrained(args.teacher_model)
    # 加载学生模型
    student_model = UIE.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # 加载训练集和验证集
    train_ds = load_dataset(reader, data_path=args.train_path, max_seq_len=args.max_seq_len, lazy=False)
    dev_ds = load_dataset(reader, data_path=args.dev_path, max_seq_len=args.max_seq_len, lazy=False)

    # 将数据集转换为可以输入模型的格式
    trans_fn = partial(convert_example, tokenizer=tokenizer, max_seq_len=args.max_seq_len)

    # 创建训练集和验证集的数据加载器
    train_data_loader = create_data_loader(train_ds, mode="train", batch_size=args.batch_size, trans_fn=trans_fn)
    dev_data_loader = create_data_loader(dev_ds, mode="dev", batch_size=args.batch_size, trans_fn=trans_fn)

    if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        logger.info("load model from path: {}".format(args.init_from_ckpt))
        distill_logger.info("load model from path: {}".format(args.init_from_ckpt))
        state_dict = paddle.load(args.init_from_ckpt)
        student_model.set_dict(state_dict)

    if paddle.distributed.get_world_size() > 1:
        student_model = paddle.DataParallel(student_model)

    # 设置优化器
    optimizer = paddle.optimizer.AdamW(
        learning_rate=args.learning_rate, parameters=student_model.parameters())

    metric = SpanEvaluator()
    criterion = paddle.nn.BCELoss()

    global_step = 0
    best_f1 = 0

    # 开始知识蒸馏训练
    for epoch in range(args.num_epochs):
        # 用于记录一轮训练的时长
        epoch_time = time.time()
        student_model, global_step = train_student(student_model, teacher_model, train_data_loader, optimizer, epoch, criterion, 0.5, global_step, rank)
        
        save_dir = os.path.join(args.save_dir, "student_model_%d" % epoch)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_to_save = student_model._layers if isinstance(student_model, paddle.DataParallel) else student_model
        model_to_save.save_pretrained(save_dir)
        logger.disable()
        tokenizer.save_pretrained(save_dir)
        logger.enable()

        if (epoch + 1) % args.eval_steps == 0:
            # 计算训练时长
            epoch_time_gap = time.time() - epoch_time
            # print presision, recall, f1
            precision, recall, f1 = evaluate(student_model, metric, dev_data_loader)
            logger.info("Epoch: %d, cost time: %d s" % (epoch, epoch_time_gap))
            logger.info("Evaluation precision: %.5f, recall: %.5f, F1: %.5f" % (precision, recall, f1))
            distill_logger.info("Epoch: %d, cost time: %d s" % (epoch, epoch_time_gap))
            distill_logger.info("Evaluation precision: %.5f, recall: %.5f, F1: %.5f" % (precision, recall, f1))
            # 记录测试数据
            log_writer.add_scalar("dev/precision", precision, global_step)
            log_writer.add_scalar("dev/recall", recall, global_step)
            log_writer.add_scalar("dev/f1", f1, global_step)
            # 将 epoch 的训练时长添加到 TensorBoard 中
            log_writer.add_scalar("epoch/train_time", epoch_time_gap, epoch)
            if f1 > best_f1:
                logger.info(f"best F1 performence has been updated: {best_f1:.5f} --> {f1:.5f}")
                distill_logger.info(f"best F1 performence has been updated: {best_f1:.5f} --> {f1:.5f}")
                best_f1 = f1
                best_dir = os.path.join(args.save_dir, "student_best")
                if not os.path.exists(best_dir):
                    os.makedirs(best_dir)
                model_to_save = student_model._layers if isinstance(student_model, paddle.DataParallel) else student_model
                model_to_save.save_pretrained(best_dir)
                logger.disable()
                tokenizer.save_pretrained(best_dir)
                logger.enable()
        
    logger.info("training finished")
    # 关闭计时器
    log_writer.close()


if __name__ == "__main__":
    # yapf: disable
    parser = argparse.ArgumentParser()

    # 知识蒸馏的参数
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--train_path", default=None, type=str, help="The path of train set.")
    parser.add_argument("--dev_path", default=None, type=str, help="The path of dev set.")
    parser.add_argument("--save_dir", default='./checkpoint', type=str, help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--teacher_model", type=str, help="path to teacher model.")
    parser.add_argument("--temperature", default=3, type=int, help="temperature of distillation")
    parser.add_argument("--max_seq_len", default=512, type=int, help="The maximum input sequence length. Sequences longer than this will be split automatically.")
    parser.add_argument("--num_epochs", default=10, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--seed", default=1000, type=int, help="Random seed for initialization")
    parser.add_argument("--logging_steps", default=16, type=int, help="The interval steps to logging.")
    parser.add_argument("--valid_steps", default=1000, type=int, help="The interval steps to evaluate model performance.")
    parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
    parser.add_argument("--model", choices=["uie-senta-micro", "uie-senta-medium", "uie-senta-mini", "uie-senta-micro", "uie-senta-nano"], default="uie-senta-micro", type=str, help="Select the pretrained model for few-shot learning.")
    parser.add_argument("--init_from_ckpt", default=None, type=str, help="The path of model parameters for initialization.")
    parser.add_argument("--eval_steps",default=1,type=int,help="step of evaluation.")
    # yapf: disable
    args = parser.parse_args()
    data_path = "/mnt/ve_share/liushuai/PaddleNLP/applications/sentiment_analysis/unified_sentiment_extraction/Data/json"
    args.train_path= os.path.join(data_path, "train_sample.json")
    args.dev_path=os.path.join(data_path, "dev_sample.json")
    args.test_path=os.path.join(data_path, "test.json")
    # args.save_dir="/mnt/ve_share/liushuai/PaddleNLP-develop/applications/sentiment_analysis/unified_sentiment_extraction/zx_workplace/distillation/student_model"

    args.teacher_model="/mnt/ve_share/liushuai/PaddleNLP/applications/sentiment_analysis/unified_sentiment_extraction/liushuai_workdir/results/base/model_best"
    args.init_from_ckpt="/mnt/ve_share/liushuai/PaddleNLP-develop/applications/sentiment_analysis/unified_sentiment_extraction/zx_workplace/checkpoint/student_best/model_state.pdparams"
    distill_knowledge()
    
