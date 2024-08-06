import os
from typing import Any, Dict, List, Optional

from kninjllm.llm_common.component import component
from root_config import RootConfig

@component
class KnowledgeSelector:

    def __init__(
        self,
        knowledge_path: str = "",
        knowledge_elasticIndex :str = "",
        knowledge_tag:str = "",
    ):
        self.knowledge_path = knowledge_path
        if self.knowledge_path != "" and not self.knowledge_path.startswith("/"):
            self.knowledge_path = os.path.join(RootConfig.root_path,self.knowledge_path)
        
        self.knowledge_elasticIndex = knowledge_elasticIndex
        self.knowledge_tag = knowledge_tag

    @component.output_types(knowledge_info=Dict[str, Any])
    def run(
        self,
    ):

        return {"knowledge_info": {"knowledge_path":self.knowledge_path,
                                   "knowledge_elasticIndex":self.knowledge_elasticIndex,
                                   "knowledge_tag":self.knowledge_tag}
                }
    