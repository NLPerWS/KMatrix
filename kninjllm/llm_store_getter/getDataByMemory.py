from typing import Any, Dict, List, Optional
from kninjllm.llm_common.component import component

@component
class getDataByMemory:

    def __init__(
        self,
        memory_list: List = None
    ):
        self.memory_list = memory_list

    @component.output_types(final_result=List[Any])
    def run(
        self,
        queryIndexList:List[int]=[],
    ):
        if self.memory_list != None:
            final_result = []
            
            if len(queryIndexList)!=1 and len(queryIndexList)!=2:
                raise ValueError("the queryIndexList is out of range.")
            
            if len(queryIndexList) == 1:
                if queryIndexList[0] > 0:
                    final_result = self.memory_list[:queryIndexList[0]]
                elif queryIndexList[0] < 0:
                    final_result = self.memory_list[queryIndexList[0]:] 
                else:
                    final_result = self.memory_list[:] 
            else:
                final_result = self.memory_list[queryIndexList[0]:queryIndexList[1]] 
            
            return {"final_result": final_result}
        
        else:
            raise ValueError("getDataByMemory initparamter is None !!!")
        