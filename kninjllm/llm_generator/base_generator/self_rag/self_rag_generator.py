import shutil
import json
import os
from typing import Any, Dict, List, Optional
from vllm import SamplingParams
from kninjllm.llm_common.component import component
import subprocess
from root_config import RootConfig
from kninjllm.llm_utils.common_utils import RequestOutputToDict,loadModelByCatch


@component
class RagGenerator:

    def __init__(
        self,
        api_key: str = "",
        model_path : str = "",
        executeType : str = "",
        do_log: bool = True,
    ):
        if do_log:
            self.logSaver = RootConfig.logSaver
        else:
            self.logSaver = None
        self.executeType = executeType
        self.model,self.tokenizer = loadModelByCatch(model_name='selfrag',model_path=model_path)
        
        
            
    @component.output_types(final_result=Dict[str, Any])
    def run(
        self,
        query_obj: Dict[str, Any],
        train_data_info:Dict[str, Any] = {},
        dev_data_info:Dict[str, Any] = {},
        sampling_params: SamplingParams=SamplingParams(temperature=0.0, top_p=1, max_tokens=300),
        saveLogFlag = True
    ):
        
        if self.executeType == "infer":
            
            if self.model==None:
                raise ValueError("The model is empty.")
            
            if "question" in query_obj and query_obj['question'] != "":
                prompt = query_obj['question']
                if self.logSaver is not None and saveLogFlag == True:
                    self.logSaver.writeStrToLog("Function -> RagGenerator -> run")
                    self.logSaver.writeStrToLog("Given generator prompt -> : "+prompt)
                
                pred = self.model.generate(prompt, sampling_params)[0]
                pred_dict =  RequestOutputToDict(pred)
                content = pred_dict['outputs'][0]['text']
            
                if self.logSaver is not None and saveLogFlag == True:
                    self.logSaver.writeStrToLog("Returns generator reply : "+content)
            
                final_result = {
                    "prompt":prompt,
                    "content":content,
                    "meta":{"pred":pred_dict},
                    **query_obj
                }  
            else:
                final_result = {
                    "prompt":"",
                    "content":"",
                    "meta":{"pred":{}},
                    **query_obj
                }  
            return {"final_result":final_result}
        
        elif self.executeType == "train":
            print("train ... ")
            MODEL_SIZE = '7B'
            BATCH_SIZE_PER_GPU = 1
            TOTAL_BATCH_SIZE = 128
            GRADIENT_ACC_STEPS = TOTAL_BATCH_SIZE
            
            previous_directory = os.getcwd()  
            temp_dir = os.path.join(RootConfig.root_path,"kninjllm/llm_generator/base_generator/self_rag/retrieval_lm")
            os.chdir(temp_dir)
            
            out_dir_name = "TEMP_SELFRAG_TRAIN_" + str(MODEL_SIZE) + "/"
            output_file = RootConfig.root_path + out_dir_name
            command = (
                f"accelerate launch "
                f"--mixed_precision bf16 "
                f"--num_machines 1 "
                # f"--num_processes {NUM_GPUS} "
                f"--use_deepspeed "
                f"--deepspeed_config_file stage3_no_offloading_accelerate.conf "
                f"finetune.py "
                f"--model_name_or_path {RootConfig.selfRAG_model_path} "
                f"--use_flash_attn "
                f"--tokenizer_name {RootConfig.selfRAG_model_path} "
                f"--use_slow_tokenizer "
                f"--train_file {train_data_info['dataset_path']} " 
                f"--max_seq_length 2048 "
                f"--preprocessing_num_workers 16 "
                f"--per_device_train_batch_size {BATCH_SIZE_PER_GPU} "
                f"--gradient_accumulation_steps {GRADIENT_ACC_STEPS} "
                f"--learning_rate 2e-5 "
                f"--lr_scheduler_type linear "
                f"--warmup_ratio 0.03 "
                f"--weight_decay 0. "
                f"--num_train_epochs 3 "
                f"--output_dir {output_file} "
                f"--with_tracking "
                f"--report_to tensorboard "
                f"--logging_steps 1 "
                f"--use_special_tokens"
            )
            process = subprocess.Popen(command, shell=True)
            os.chdir(previous_directory)
            return {"final_result":{"content":f"train start,please wait... (please check log in log file: {out_dir_name})","meta":{}}} 
        
        else:
            raise ValueError("Unknown executeType: %s" % self.executeType)