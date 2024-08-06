from typing import Any, Dict, List

from jinja2 import Template, meta

from kninjllm.llm_common.component import component
from kninjllm.llm_common.serialization import  default_to_dict

@component
class PromptBuilder:

    def __init__(self, template: str):

        self._template_string = template
        self.template = Template(template)
        
        ast = self.template.environment.parse(template)
        template_variables = meta.find_undeclared_variables(ast)
        for var in template_variables:
            component.set_input_type(self, var, Any, "")

    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(self, template=self._template_string)

    @component.output_types(prompt_obj=Dict[str,Any])
    def run(self, 
            query_obj: Dict[str,Any],
            retriever_list: List[List[Dict[str,Any]]]
        ):
        
        if "question" in query_obj:
            question = query_obj['question']
        else:
            question = ""
        
        if len(retriever_list) == 1:
            retriever_list = retriever_list[0]
        else:
            retriever_list = []
        
        prompt = self.template.render({"query_obj":query_obj,"retriever_list":retriever_list})
        prompt_obj = {"question":prompt,"query":question}
        
        print("----------------------------- prompt builder prompt_obj----------------------")
        
        
        return {"prompt_obj":prompt_obj}