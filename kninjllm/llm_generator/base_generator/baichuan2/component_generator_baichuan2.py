from typing import Any, Dict, List, Optional
from vllm import RequestOutput, SamplingParams
from kninjllm.llm_common.component import component
from kninjllm.llm_utils.common_utils import RequestOutputToDict,loadModelByCatch
from root_config import RootConfig

@component
class Baichuan2Generator:

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
        self.saveFlag = False
        self.model,self.tokenizer = loadModelByCatch(model_name='baichuan2',model_path=model_path)
            
        print("Baichuan2 ... ")
    
    @component.output_types(final_result=Dict[str, Any])
    def run(
        self,
        query_obj: Dict[str, Any],
        train_data_info:Dict[str, Any] = {},
        dev_data_info:Dict[str, Any] = {},
    ):
        if self.model==None:
            raise ValueError("The model is empty.")
        
        print("------------------------------  baichuan2  -------------------------")
        
        if "question" in query_obj and query_obj['question'] != "":
            
            prompt = query_obj['question']
            if self.logSaver is not None:
                self.logSaver.writeStrToLog("Function -> Baichuan2Generator -> run")
                self.logSaver.writeStrToLog("Given generator prompt -> : "+prompt)
                
            messages = [{"role": "user", "content": prompt}]
            content = self.model.chat(self.tokenizer, messages)
                
            if self.logSaver is not None:
                self.logSaver.writeStrToLog("Returns generator reply : "+ content )

            final_result = {
                "prompt":prompt,
                "content":content,
                "meta":{"pred":{}},
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
    
        