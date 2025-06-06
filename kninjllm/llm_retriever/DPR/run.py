import logging
import os
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)

from arguments import ModelArguments, DataArguments, \
    RetrieverTrainingArguments as TrainingArguments
from data import TrainDatasetForEmbedding, SameDatasetTrainDataset, EmbedCollator
from modeling import BiEncoderModel
from trainer import BiTrainer
logger = logging.getLogger(__name__)

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    training_args.learning_rate = 2e-5
    training_args.fp16 = True
    training_args.num_train_epochs = 1
    training_args.gradient_checkpointing = True
    training_args.output_dir = "TEMP_TRAIN/DPR/ModelSave"
    training_args.logging_steps = 10
    model_args.output_dir = "TEMP_TRAIN/DPR/ModelSave"
    
    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    # Set seed
    set_seed(training_args.seed)

    num_labels = 1
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
        trust_remote_code=True,
    )
    if data_args.model_type == "LLM":
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
        #trust_remote_code=True,
    )
    logger.info('Config: %s', config)

    model = BiEncoderModel(model_name=model_args.model_name_or_path,
                           model_type=data_args.model_type,
                           normlized=training_args.normlized,
                           sentence_pooling_method=training_args.sentence_pooling_method,
                           negatives_cross_device=training_args.negatives_cross_device,
                           temperature=training_args.temperature,
                           use_inbatch_neg=training_args.use_inbatch_neg,
                           lora_tune=training_args.lora_tune,
                           use_flash_attn=training_args.use_flash_attn,
                           )

    if training_args.fix_position_embedding:
        for k, v in model.named_parameters():
            if "position_embeddings" in k:
                logging.info(f"Freeze the parameters for {k}")
                v.requires_grad = False

    if data_args.same_task_within_batch:
        logger.info('Using SameDatasetTrainDataset')
        train_dataset = SameDatasetTrainDataset(tokenizer=tokenizer,
                                                args=data_args, 
                                                batch_size=training_args.per_device_train_batch_size, 
                                                seed=training_args.seed, 
                                                num_processes=training_args.world_size,
                                                process_index=training_args.process_index)
        training_args.per_device_train_batch_size = 1
        training_args.dataloader_num_workers = 0    # avoid multi-processes
    else:
        train_dataset = TrainDatasetForEmbedding(tokenizer=tokenizer, args=data_args)

    data_collator = EmbedCollator(
        tokenizer,
        args=data_args,
        query_max_len=data_args.query_max_len,
        passage_max_len=data_args.passage_max_len,
    )
    
    trainer = BiTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    # Training
    if training_args.resume_from_ckpt != None:
        trainer.train(resume_from_checkpoint=training_args.resume_from_ckpt)
    else:
        trainer.train()
    trainer.save_model()
    # For convenience, we also re-save the tokenizer to the same directory,
    # so that you can share your model easily on huggingface.co/models =)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
