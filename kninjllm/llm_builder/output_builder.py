import json
from typing import Any, Dict, List
from kninjllm.llm_common.component import component
from root_config import RootConfig

@component
class OutputBuilder:
    def __init__(
        self,
    ):
        self.logSaver = RootConfig.logSaver
    
    
    @component.output_types(final_result=Dict[str, Any])
    def run(self, params: Any):
        final_result = {
            "content":"",
            "ctxs":[],
            "meta":{}
        }
        
        
        if isinstance(params,list) and  len(params) > 0:
            params = params[0]
        
        
        if isinstance(params,Dict) and "content" in params:
            final_result['content'] = params['content']
        else:
            final_result['content'] = params

        if isinstance(params,Dict) and "meta" in params:
            final_result['meta'] = params['meta']

        
        if "content" not in params and "ctxs" in params:
            for ctx in params['ctxs']:
                one_str = ctx['id'].strip() + "\t" + ctx['title'].strip() + "\t" + ctx['text'].strip() + "\n\n"
                final_result['content'] = final_result['content'] + one_str


        
        if self.logSaver is not None:
            final_result['ctxs'] = self.logSaver.readLogToTxtList()

        return {"final_result":final_result}

