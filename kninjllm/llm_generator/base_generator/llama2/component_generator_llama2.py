from typing import Any, Dict, List, Optional
from vllm import SamplingParams
from kninjllm.llm_common.component import component
from kninjllm.llm_utils.common_utils import RequestOutputToDict,loadModelByCatch
from root_config import RootConfig

@component
class LLama2Generator:

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
        self.model = loadModelByCatch(model_name='llama2',model_path=model_path)
        
    @component.output_types(final_result=Dict[str, Any])
    def run(
        self,
        query_obj: Dict[str, Any],
        train_data_info:Dict[str, Any] = {},
        dev_data_info:Dict[str, Any] = {},
        sampling_params: Any = SamplingParams(temperature=0.0, top_p=1.0)
    ):
        
        if self.model==None:
            raise ValueError("The model is empty.")
        
        print("------------------------------  llama2  -------------------------")
        
        if "question" in query_obj and query_obj['question'] != "":
            
            prompt = query_obj['question']
            if self.logSaver is not None:
                self.logSaver.writeStrToLog("Function -> LLama2Generator -> run")
                self.logSaver.writeStrToLog("Given generator prompt -> : " + prompt)
                
            pred = self.model.generate(prompt, sampling_params)[0]
            pred_dict =  RequestOutputToDict(pred)
            content = pred_dict['outputs'][0]['text']
            
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
    
        
    