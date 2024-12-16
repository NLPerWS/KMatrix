from typing import Dict,Any
from kninjllm.llm_common.component import component

@component
class Local_interface:
    def __init__(
        self,
        interface_domain:str,
        interface_type:str,
        search_url:str,
    ):
        self.interface_domain = interface_domain
        self.interface_type = interface_type
        self.search_url = search_url

    @component.output_types(interface_info=Dict[str,Any])
    def run(self):
        print("------------------Local_interface----------------------")
        return {"interface_info":{"interface_domain":self.interface_domain,"interface_type":self.interface_type,"search_url":self.search_url}}