
import json
import os
from typing import Any, Dict, List
from kninjllm.llm_common.component import component
from kninjllm.llm_knowledgeUploader.utils.interface_execute import InterfaceExecute

@component
class InterfacePreprecess:
    def __init__(self):
        pass
    
    @component.output_types(interface=Any)
    def run(
        self,
        interface_info:Dict[str, Any],
    ):
        interface_domain = interface_info['interface_domain']
        interface_type = interface_info['interface_type']
        search_url = interface_info['search_url']
        interface = InterfaceExecute(domain=interface_domain,type=interface_type,url=search_url)
        return {"interface":interface}
    