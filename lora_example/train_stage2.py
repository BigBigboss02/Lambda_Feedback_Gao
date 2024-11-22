#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence speech recognition.
"""
# You can also adapt this script on your own sequence to sequence speech
# recognition task. Pointers for this are left as comments.

import logging
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union, Tuple
import typing
import datasets
datasets.config.HF_DATASETS_OFFLINE = True

import transformers
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaConfig, WhisperConfig, WhisperFeatureExtractor
from transformers.deepspeed import is_deepspeed_zero3_enabled
# from src.lxxmodify import load_speech_text_paired_dataset
from src.speech_text_paired_dataset import load_speech_text_paired_dataset, SpeechTextPairedDataCollator
from src.modeling_blsp import BlspModel
from src.modeling_whisper_encoder import WhisperEncoder
from src.configuration_blsp import BlspConfig
import json
# from modelscope import AutoTokenizer, AutoModel, snapshot_download
from peft import LoraConfig, get_peft_model
from peft import PeftModel, PeftConfig
logger = logging.getLogger(__name__)
from transformers import AutoTokenizer, AutoModelForCausalLM

@dataclass
class LoraArguments:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: typing.List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj"]
    )
    # lora_weight_path: str = "/data/jbn/pretrained_models/vicuna_stage1_correction"#"/data/jbn/pretrained_models/new_atom/checkpoint-1500"
    bias: str = "none"
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    llama_model: str = field(
        default="/data/jbn/pretrained_models/stage1", metadata={"help": "the path of base model"}
    )
    whisper_model: str = field(
        default="openai/whisper-small", metadata={"help": "the path of whisper model"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    data: str = field(
        metadata={
            "help": "the root to load dataset"
        },
    )
    manifest_files: str = field(
        default="",
        metadata={
            "help": "The name of the training unit text paired set split to use."
        },
    )



def main():
    # 1. Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, LoraArguments))
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()

    # 2. Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Dataset parameters {data_args}")

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # 3. Detecting last checkpoint and eventually continue from last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)
    
    # 4. Load tokenizer
    #  
    tokenizer = LlamaTokenizer.from_pretrained(model_args.llama_model)
    # tokenizer = AutoTokenizer.from_pretrained(model_args.llama_model)
    extractor = WhisperFeatureExtractor.from_pretrained(model_args.whisper_model)
    
    ## 5. Load dataset

    instruction=""
    dataset = load_speech_text_paired_dataset(
        dataroot=data_args.data,
        manifest_files=data_args.manifest_files,
        tokenizer=tokenizer,
        # max_length=data_args.max_length,
        instruction=instruction
    )

    # 6. Load pretrained model
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    whisper_config = WhisperConfig.from_pretrained(model_args.whisper_model)
    #下面这一步去除所有的llama 在初始化BLSPcofig的时候，因为我已经在init函数去除llama的参数了，所以不用care
    llama_config = LlamaConfig.from_pretrained(model_args.llama_model)
    # lora_m_config = PeftConfig.from_pretrained(model_args.lora_config)
    # PeftModel, PeftConfig
    blsp_config = BlspConfig(
        whisper_config.to_dict(),
        llama_config.to_dict()
    )
    #/root/.cache/modelscope/hub/ZhipuAI/chatglm3-6b-base

    model = BlspModel(blsp_config)
    
    model.whisper_model = WhisperEncoder.from_pretrained(model_args.whisper_model)

    model.llama_model = LlamaForCausalLM.from_pretrained(model_args.llama_model, _fast_init=not is_deepspeed_zero3_enabled())

    lora_config = LoraConfig(
    r=lora_args.lora_r,
    lora_alpha=lora_args.lora_alpha,
    target_modules=lora_args.lora_target_modules,
    lora_dropout=lora_args.lora_dropout,
    bias=lora_args.bias,
    # lora_weight_path = lora_args.lora_weight_path,
    task_type="CAUSAL_LM",
    )
    
    model.llama_model = get_peft_model(model.llama_model, lora_config)
    # model.llama_model = PeftModel.from_pretrained(model.llama_model, '/data/jbn/pretrained_models/stage1_onlyCMI_15')

    for name, param in model.whisper_model.named_parameters():
        param.requires_grad = False
    for name, param in model.llama_model.named_parameters():
        param.requires_grad = False

    # 6. Define data collator
    data_collator = SpeechTextPairedDataCollator(
        pad_id=tokenizer.pad_token_id,
        sampling_rate=extractor.sampling_rate,
        extractor=extractor
    )


    # 7. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # 8. Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    results = {}
    # 9. Save tokenizer for inference load
    tokenizer.save_pretrained(training_args.output_dir)
    extractor.save_pretrained(training_args.output_dir)

    # 试图尝试额外save一下llm
    llm_path = training_args.output_dir + '/llm'
    if not os.path.exists(llm_path):
        os.makedirs(llm_path)
    model.llama_model.save_pretrained(llm_path)
    
    return results


if __name__ == "__main__":
    main()