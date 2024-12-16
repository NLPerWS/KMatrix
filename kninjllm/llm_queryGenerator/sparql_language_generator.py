from typing import Any, Dict, List, Optional

from kninjllm.llm_common.component import component
from root_config import RootConfig
from kninjllm.llm_generator.base_generator.wikidata_emnlp23.infer import e2esparql_generate_flask

@component
class Sparql_language_generator:

    def __init__(
        self,
    ):
        pass
    
    @component.output_types(final_query=Dict[str, Any])
    def run(
        self,
        query_obj: Dict[str, Any],
    ):
        query = query_obj["question"]
        print("query = input:",query)
        res_query = e2esparql_generate_flask(query)
        
        return {"final_result":res_query}