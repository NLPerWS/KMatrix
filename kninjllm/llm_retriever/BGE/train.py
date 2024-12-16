import subprocess
import os

def train(train_data,train_model_path,output_model_name):

    
    command = [
        "torchrun", "--nproc_per_node", "1", "run.py",
        "--output_dir", f"{output_model_name}",
        "--model_name_or_path", train_model_path,
        "--model_type", "BERT",
        "--train_data", train_data,
        "--same_task_within_batch", "True",
        "--use_instruct", "False",
        "--learning_rate", "2e-5",
        "--fp16",
        "--num_train_epochs", "1",
        "--per_device_train_batch_size", "256",
        "--dataloader_drop_last", "True",
        "--normlized", "True",
        "--temperature", "0.02",
        "--query_max_len", "64",
        "--passage_max_len", "236",
        "--train_group_size", "8",
        "--gradient_checkpointing",
        "--deepspeed", "./DsConfig/ds_config_zero1.json",
        "--negatives_cross_device",
        "--logging_steps", "10",
        "--save_steps", "5000",
        "--warmup_ratio", "0.1",
        "--sentence_pooling_method", "cls"
    ]

    
    
    process = subprocess.Popen(command)