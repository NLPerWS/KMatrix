import json
import time
from typing import Any, Dict, List, Optional
from kninjllm.llm_common.component import component
from root_config import RootConfig

@component
class SaveQueryInterface:

    def __init__(
        self,
    ):
        pass

    @component.output_types(final_result=Dict[str,Any])
    def run(
        self,
        interface_list: List[Any],
    ):
        from_list = ["from kninjllm.llm_knowledgeUploader.utils.interface_execute import InterfaceExecute"]
        domain_mapping = {}
        
        temp_QueryName_list = []
        for interface in interface_list:
            root_class = interface.domain.split('/')[0]
            interface_class = interface.domain.split('/')[1]
            temp_QueryName = f"InterfaceExecute(domain='{interface.domain}',type='{interface.type}',url='{interface.url}')"
            temp_QueryName_list.append(temp_QueryName)
            
            if root_class in domain_mapping:
                domain_mapping[root_class].update({interface_class:temp_QueryName})
            else:
                domain_mapping[root_class] = {interface_class:temp_QueryName}
        
        final_str = ""
        final_str = final_str + "\n".join(from_list)
        final_str += "\n\n"
        final_str += "domain_mapping = "
        final_str += json.dumps(domain_mapping,ensure_ascii=False,indent=4)
        
        for temp_QueryName in temp_QueryName_list:
            final_str = final_str.replace('"'+temp_QueryName+'"',temp_QueryName)
        
        with open(RootConfig.root_path + "kninjllm/llm_knowledgeUploader/utils/interface_config.py",'w',encoding='utf-8') as f:
            f.write(final_str)
        
        return {"final_result":"save success"}
