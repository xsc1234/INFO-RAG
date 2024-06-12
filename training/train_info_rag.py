#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
import argparse
import os
import math
import sys,json
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    AutoModelForCausalLM,
    LlamaForCausalLM,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    GenerationConfig,
)

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from torch.utils.data import Dataset
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from utils.ds_utils import get_train_ds_config
from utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters, make_model_gradient_checkpointing_compatible
from utils.model.model_utils import create_llama_model
import transformers
import subprocess
import copy
import regex,string
from collections import Counter
IGNORE_INDEX = -100
from dataclasses import dataclass

def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--uns_data_path',
                        type=str,
                        default='/tmp/data_files/')
    parser.add_argument(
        '--sft_only_data_path',
        nargs='*',
        default=[],
        help='Path to the dataset for only using in SFT phase.')
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files/',
        help=
        'Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--setting_max_length",
                        type=int,
                        default=1,
                        help="Max input length.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--gradient_checkpointing',
                        action='store_true',
                        help='Enable HF gradient checkpointing for model.')
    parser.add_argument('--disable_dropout',
                        action='store_true',
                        help='Disable the dropout of the model.')
    # deepspeed features
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    ## LoRA for efficient training setting
    parser.add_argument("--lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')
    parser.add_argument(
        "--lora_learning_rate",
        type=float,
        default=5e-4,
        help=
        "Initial LoRA learning rate (after the potential warmup period) to use."
    )
    ## Tensorboard logging
    parser.add_argument('--enable_tensorboard',
                        action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--tensorboard_path',
                        type=str,
                        default="step1_tensorboard")
    ## Print loss
    parser.add_argument('--print_loss',
                        action='store_true',
                        help='Prints loss at each step.')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args

def tokenize(
        prompt,
        completion,
        tokenizer: transformers.PreTrainedTokenizer,
):
    """Preprocess the data by tokenizing."""
    source_output = tokenizer.encode(prompt)
    input_seq = prompt + ' ' + completion
    passage_list = prompt
    tokenize_output = tokenizer(input_seq, padding=False, return_tensors=None,max_length=512,truncation=False)
    passage_list_tokenize_output = tokenizer(passage_list, padding=False, return_tensors=None, max_length=512, truncation=False)
    IGNORE_INDEX = -100
    source_len = len(source_output) - 1

    tokenize_output["labels"] = copy.deepcopy(tokenize_output["input_ids"])
    tokenize_output["labels"] = [IGNORE_INDEX] * source_len + tokenize_output["labels"][source_len:]
    return passage_list_tokenize_output,tokenize_output

special_token_list = [1,32000,32001]
import random
class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_type,
                 data_list):
        super(SupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.data_list = data_list

        if data_type == 'train':
            self.data_list = self.data_list[:int(1.0*len(self.data_list))]
        else:
            self.data_list = self.data_list[int(0.2*len(self.data_list))+1:]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        if i % (1000) == 0 and int(os.environ.get("LOCAL_RANK")) == 0:
            sp = subprocess.Popen(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out_str = sp.communicate()
            for out_element in out_str:
                for line in str(out_element).split('\\n'):
                    print(line, file=sys.stderr)
        return self.data_list[i]
@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        return instances

import joblib

def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    norm_predict = normalize_answer(prediction)
    norm_answer = normalize_answer(ground_truth)
    return float(norm_answer in norm_predict)

def eval_ans(prediction: str, reference: str):
    norm_pred, norm_ref = normalize_answer(prediction), normalize_answer(reference)
    em = exact_match_score(prediction,reference)

    zeros = (0., 0., 0.)

    pred_tokens, ref_tokens = norm_pred.split(), norm_ref.split()
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return (em,) + zeros
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(ref_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return em, f1, precision, recall

# def evaluate(
#     tokenizer,
#     model,
#     device,
#     temperature=0.1,
#     top_p=0.75,
#     top_k=40,
#     num_beams=4,
#     max_new_tokens=256,
#     **kwargs,
# ):
#     em_sum = 0
#     em_count = 0
#     data_list = joblib.load('/apdcephfs/share_47076/shchxu/ir_datasets/odqa/nq/nq-test-ref.qa')
#     for i in data_list:
#         temp_dic = {}
#         temp_dic['instruction'] = i[0] + ' Reference is ' +i[1]
#         temp_dic['input'] = ''
#         temp_dic['output'] = i['positive_ctxs'][0]['text']
#         #prompt = generate_prompt(instruction, input)
#         prompt = 'Please tell me more about: ' + temp_dic['instruction'] + ' Answer is: '
#         inputs = tokenizer(prompt, return_tensors="pt")
#         input_ids = inputs["input_ids"].to(device)
#         generation_config = GenerationConfig(
#             temperature=temperature,
#             top_p=top_p,
#             top_k=top_k,
#             num_beams=num_beams,
#             **kwargs,
#         )
#         with torch.no_grad():
#             generation_output = model.generate(
#                 input_ids=input_ids,
#                 generation_config=generation_config,
#                 return_dict_in_generate=True,
#                 output_scores=True,
#                 max_new_tokens=max_new_tokens,
#             )
#         s = generation_output.sequences[0]
#         output = tokenizer.decode(s)
#         print(output)
#         print('-----------------------------------------')
#         em = 0
#         answer_list = i['answer_list']
#         output_answer = output.split('Answer is:')[1]
#         for answer in answer_list[0]:
#             em, f1, precision, recall = eval_ans(output_answer, answer)
#             if em > 0:
#                 break
#         em_sum += em
#         em_count += 1
#         print(em_count)
#     print('EM is {}'.format(em_sum / em_count))

def main():
    args = parse_args()

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    ds_config = get_train_ds_config(offload=args.offload,
                                    stage=args.zero_stage,
                                    enable_tensorboard=args.enable_tensorboard,
                                    tb_path=args.tensorboard_path,
                                    tb_name="step1_model")
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
        ) * args.gradient_accumulation_steps
    ds_config['fp16']['enabled'] = False
    print(ds_config)
    ds_config['bf16'] = {'enabled': True}
    print(ds_config)

    # If passed along, set the training seed now.
    set_random_seed(args.seed)

    torch.distributed.barrier()

    # load_hf_tokenizer will get the correct tokenizer and set padding tokens based on the model family
    print('args.model_name_or_path is {}'.format(args.model_name_or_path))
    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    tokenizer.padding_side = 'left'
    if 'llama' in args.model_name_or_path:
        tokenizer.pad_token_id = 0


    print(tokenizer.padding_side)
    print(tokenizer.pad_token_id)
    print(tokenizer.eos_token_id)
    print(tokenizer.bos_token_id)

    #tokenizer.add_tokens(['[REFERENCE]','[QUERY]','[MASK_PASSAGE]','[TRACE]'])
    tokenizer.add_tokens(['[MASK_PASSAGE]'])


    model = create_llama_model(LlamaForCausalLM,
                               args.model_name_or_path,
                               tokenizer,
                               ds_config,
                               disable_dropout=args.disable_dropout
                               )


    if args.lora_dim > 0:
        model = convert_linear_layer_to_lora(model, args.lora_module_name,
                                             args.lora_dim)
        if args.only_optimize_lora:
            model = only_optimize_lora_parameters(model)
            model = make_model_gradient_checkpointing_compatible(model)
    for name, param in model.named_parameters():
        print(name,param.requires_grad)

    # prepare data
    # data_list = joblib.load('/apdcephfs/share_47076/shchxu/ir_datasets/matching_data_800w_len=5_sentences_filter_doct5query_final')
    data_list = joblib.load(args.uns_data_path)
    train_dataset = SupervisedDataset(tokenizer=tokenizer,data_type='train',data_list=data_list)

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    if args.local_rank == -1:
        train_sampler = SequentialSampler(train_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=data_collator,
                                  sampler=train_sampler,
                                  shuffle=False,
                                  batch_size=args.per_device_train_batch_size)

    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model, args.weight_decay, args.lora_learning_rate)

    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              betas=(0.9, 0.95))

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Train!
    print_rank_0("***** Running training *****", args.global_rank)

    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank)
        model.train()
        loss_step = 0
        loss_mask = 0
        loss_ground = 0
        loss_copy = 0
        step_mask = 0
        step_ground = 0
        step_copy = 0
        for step, data_list in enumerate(train_dataloader):
            ### data start ###
            instances_new = []
            data_tag = 0
            for i in range(len(data_list)):
                if step%10 < 4:  # Correct and Complete
                    data_tag = 0
                    special_token = '[REFERENCE]'
                    passage_list = data_list[i][2]
                    score_list = data_list[i][3]
                    passage_ids_list = []
                    for passage in passage_list:
                        input_passage_text = (special_token + ' ' + passage)
                        output_passage, output_all_token = tokenize(input_passage_text, '',
                                                                    tokenizer)
                        output_passage_token_ids = output_passage['input_ids']
                        label_ids = output_passage_token_ids
                        #print(label_ids)

                        mask_probability = 0.3 #process 30% tokens

                        masked_indices = [i for i in range(int(len(output_passage_token_ids) / 3),len(output_passage_token_ids)) if
                                          random.random() < mask_probability]
                        masked_ids = []
                        idx = 0
                        masked_idx = []
                        while idx < len(output_passage_token_ids):
                            if (not (idx in masked_indices)) or (output_passage_token_ids[idx] in special_token_list):
                                masked_ids.append(output_passage_token_ids[idx])
                                idx += 1
                            else:  # Process two tokens consecutively
                                rand_num = random.random()
                                if rand_num < 0.5:  # [MASK]
                                    masked_ids.append(32000)
                                elif rand_num > 0.5 and rand_num < 0.6:  # Keep
                                    masked_ids.append(output_passage_token_ids[idx])
                                else:
                                    masked_ids.append(random.randint(3, 31999))  # Replace
                                masked_idx.append(idx)
                                idx += 1
                                if idx < len(output_passage_token_ids) and (
                                not (output_passage_token_ids[idx] in special_token_list)):
                                    rand_num = random.random()
                                    if rand_num < 0.5:  # [MASK]
                                        masked_ids.append(32000)
                                    elif rand_num > 0.5 and rand_num < 0.6:  # Keep
                                        masked_ids.append(output_passage_token_ids[idx])
                                    else:
                                        masked_ids.append(random.randint(3, 31999))  # Replace
                                    masked_idx.append(idx)
                                    idx += 1
                        passage_ids_list.append((label_ids, masked_ids))

                    query_ids = tokenize(
                        'Complete this text according to the above [REFERENCE]: ' + data_list[i][1], '',
                        tokenizer)[0]['input_ids'][1:] # remove start token
                    selected_s_i = 0
                    for s_i in range(len(score_list)):
                        if score_list[s_i] == 0:
                            selected_s_i = s_i
                            break
                    trace_ids = tokenize(
                        'This content is generated according to my knowledge and [REFERENCE] number {}'.format(selected_s_i), '',
                        tokenizer)[0]['input_ids'][1:] # remove start token
                    start_generation = random.randint(int(len(query_ids) / 2),int((3 * len(query_ids)) / 4))
                    answer_label_ids = [IGNORE_INDEX]*start_generation + query_ids[start_generation:] + trace_ids
                    input_passage_ids = [1]
                    origin_ids = [1]
                    for item in passage_ids_list:
                        input_passage_ids += item[1][1:]
                        origin_ids += item[0][1:]
                    input_ids = input_passage_ids + query_ids + trace_ids
                    labels = [IGNORE_INDEX] * len(input_passage_ids) + answer_label_ids

                elif step%10 >= 4 and step%10 < 8:  # Contextual Stimulation
                    data_tag = 1
                    special_token = '[REFERENCE]'
                    passage_list = data_list[i][2]
                    selected_passage = '[QUERY] ' + data_list[i][1]
                    score_list = data_list[i][3]
                    score_list_fuben = []
                    for s in score_list:
                        if not s == 0:
                            score_list_fuben.append(s)
                    if len(score_list) > 1:
                        score_list = score_list_fuben
                    passage_ids_list = []
                    for passage in passage_list:
                        input_passage_text = (special_token + ' ' + passage)
                        output_passage, output_all_token = tokenize(input_passage_text, selected_passage,
                                                                    tokenizer)
                        output_passage_token_ids = output_passage['input_ids']
                        label_ids = output_passage_token_ids
                        if passage == data_list[i][1]:
                            selected_ids = label_ids
                        else:
                            passage_ids_list.append(label_ids)

                    query_ids = tokenize(
                        'Complete this text according to the above [REFERENCE]: ' + data_list[i][1], '',
                        tokenizer)[0]['input_ids'][1:] # remove start token
                    selected_s_i = 0
                    for s_i in range(len(score_list)):
                        if score_list[s_i] == 0:
                            selected_s_i = s_i
                            break
                    trace_ids = tokenize(
                        'This content is generated according to my knowledge'.format(selected_s_i), '',
                        tokenizer)[0]['input_ids'][1:] # remove start token
                    start_generation = random.randint(int(len(query_ids) / 2),int((3 * len(query_ids)) / 4))
                    answer_label_ids = [IGNORE_INDEX]*start_generation + query_ids[start_generation:] + trace_ids
                    input_passage_ids = [1]
                    origin_ids = [1]
                    for item in passage_ids_list:
                        input_passage_ids += item[1:]
                        origin_ids += item[1:]
                    input_ids = input_passage_ids + query_ids + trace_ids
                    labels = [IGNORE_INDEX] * len(input_passage_ids) + answer_label_ids

                else:  # Select and Copy
                    data_tag = 2
                    special_token = '[REFERENCE]'
                    passage_list = data_list[i][2]
                    selected_passage = '[QUERY] ' + data_list[i][1]
                    score_list = data_list[i][3]
                    passage_ids_list = []
                    for passage in passage_list:
                        input_passage_text = (special_token + ' ' + passage)
                        output_passage, output_all_token = tokenize(input_passage_text, selected_passage,
                                                                    tokenizer)
                        output_passage_token_ids = output_passage['input_ids']
                        label_ids = output_passage_token_ids
                        passage_ids_list.append(label_ids)
                    query_ids = tokenize(
                        'Complete this text according to the above [REFERENCE]: ' + data_list[i][1], '',
                        tokenizer)[0]['input_ids'][1:] # remove start token
                    selected_s_i = 0
                    for s_i in range(len(score_list)):
                        if score_list[s_i] == 0:
                            selected_s_i = s_i
                            break
                    trace_ids = tokenize(
                        'This content is generated according to [REFERENCE] number {}'.format(selected_s_i), '',
                        tokenizer)[0]['input_ids'][1:] # remove start token
                    start_generation = random.randint(int(len(query_ids) / 2),int((3 * len(query_ids)) / 4))
                    answer_label_ids = [IGNORE_INDEX]*start_generation + query_ids[start_generation:] + trace_ids
                    input_passage_ids = [1]
                    for item in passage_ids_list:
                        input_passage_ids += item[1:]
                    input_ids = input_passage_ids + query_ids + trace_ids
                    labels = [IGNORE_INDEX] * len(input_passage_ids) + answer_label_ids

                data_temp = {}
                data_temp['input_ids'] = input_ids
                data_temp['attention_mask'] = [1] * len(input_ids)
                data_temp['labels'] = labels
                data_temp['scores'] = score_list
                data_temp['query'] = query_ids
                data_temp['data_tag'] = data_tag
                instances_new.append(data_temp)

            input_ids = []
            attention_mask = []
            labels = []
            scores = []
            max_length = 0
            data_tag = []
            for instance in instances_new:
                if len(instance["input_ids"]) > max_length:
                    max_length = min(args.setting_max_length, len(instance["input_ids"]))
                if len(instance["input_ids"]) > args.setting_max_length:
                    max_length = args.setting_max_length
                    instance["input_ids"] = instance["input_ids"][:args.setting_max_length]
                    instance["attention_mask"] = instance["attention_mask"][:args.setting_max_length]
                    instance["labels"] = instance["labels"][:args.setting_max_length]
                input_ids.append(instance["input_ids"])
                attention_mask.append(instance["attention_mask"])
                labels.append(instance["labels"])
                scores.append(instance['scores'])
                data_tag.append(instance['data_tag'])

            for i in range(len(input_ids)):
                #print(data_tag, len(input_ids[i]), len(labels[i])) #label和input id不想等
                if not len(input_ids[i]) == len(labels[i]):
                    print(input_ids[i])
                    print(labels[i])
                remainder_pad = [tokenizer.pad_token_id] * (max_length - len(input_ids[i]))
                remainder_att = [0] * (max_length - len(input_ids[i]))
                remainder_ign = [IGNORE_INDEX] * (max_length - len(labels[i]))
                if tokenizer.padding_side == 'left':
                    input_ids[i] = remainder_pad + input_ids[i]
                    attention_mask[i] = remainder_att + attention_mask[i]
                    labels[i] = remainder_ign + labels[i]
                elif tokenizer.padding_side == 'right':
                    input_ids[i] = input_ids[i] + remainder_pad
                    attention_mask[i] = attention_mask[i] + remainder_att
                    labels[i] = labels[i] + remainder_ign
                else:
                    raise NotImplementedError('Invalid padding-side setup! Two choices only: left and right. ')

            input_ids = torch.tensor(input_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask)
            try:
                labels = torch.tensor(labels, dtype=torch.long)
            except:
                print(input_ids)
                print(labels)
            batch = dict(
                input_ids=input_ids,
                # labels=labels,
                labels=labels,
                attention_mask=attention_mask,
                scores=scores,
                data_tag=data_tag
            )
            #### data done ####

            batch_input = {}
            for k, v in batch.items():
                if not (k == 'scores' or k == 'data_tag'):
                    batch_input[k] = batch[k]
            batch_input = to_device(batch_input, device)
            outputs = model(**batch_input, output_attentions=True, use_cache=False)
            loss = outputs.loss
            loss_step += loss
            if batch['data_tag'][0] == 0:
                loss_mask += loss
                step_mask += 1
            elif batch['data_tag'][0] == 1:
                loss_ground += loss
                step_ground += 1
            elif batch['data_tag'][0] == 2:
                loss_copy += loss
                step_copy += 1

            if step % 1000 == 0:
                print(
                    f"Epoch: {epoch}, Step: {step}, Rank: {torch.distributed.get_rank()}, loss = {loss_step / 1000}"
                    f"loss mask = {loss_mask / (step_mask+1)} loss ground = {loss_ground / (step_ground+1)} loss copy = {loss_copy / (step_copy+1)}"
                )
                loss_step = 0
                loss_mask = 0
                loss_ground = 0
                loss_copy = 0
                step_mask = 0
                step_ground = 0
                step_copy = 0
                # print(
                #     f"Epoch: {epoch}, Step: {step}, Rank: {torch.distributed.get_rank()}, avg_loss_att = {avg_loss_att}, avg_loss_lm = {avg_loss_lm}"
                # )
            model.backward(loss)
            for name,para in model.named_parameters():
                print(name,para.grad)
            model.step()

            if args.output_dir is not None and step % 5000 == 0 and step > 0:
                print_rank_0('saving the final model ...', args.global_rank)
                model = convert_lora_to_linear_layer(model)

                if args.global_rank == 0:
                    save_hf_format(model, tokenizer, args, sub_folder='epoch_{}_step_{}'.format(epoch,step))

                if args.zero_stage == 3:
                    # For zero stage 3, each gpu only has a part of the model, so we need a special save function
                    save_zero_three_model(model,
                                          args.global_rank,
                                          save_dir=os.path.join(
                                              args.output_dir, 'epoch_{}_step_{}'.format(epoch,step)),
                                          zero_stage=args.zero_stage)
        model.tput_timer.update_epoch_count()

    if args.output_dir is not None:
        print_rank_0('saving the final model ...', args.global_rank)
        model = convert_lora_to_linear_layer(model)

        if args.global_rank == 0:
            save_hf_format(model, tokenizer, args, sub_folder='final')

        if args.zero_stage == 3:
            # For zero stage 3, each gpu only has a part of the model, so we need a special save function
            save_zero_three_model(model,
                                  args.global_rank,
                                  save_dir=os.path.join(
                                      args.output_dir, 'final'),
                                  zero_stage=args.zero_stage)
    #evaluate(tokenizer, model, device)
if __name__ == "__main__":
    main()
