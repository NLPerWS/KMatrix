import time
from typing import Any, Dict, List, Optional
from kninjllm.llm_common.component import component

@component
class saveDataToMemory:

    def __init__(
        self,
        memory_list: List = None
    ):
        self.memory_list = memory_list

    @component.output_types(final_result=Dict[str,Any])
    def run(
        self,
        save_obj: Any,
    ):
        print("-------------------do saver !!!------------------")
        
        if self.memory_list != None:
           self.memory_list.append(save_obj)
        else:
            raise ValueError("saveDataToMemory initparamter is None !!!")
        
        return {"final_result":"save success"}
