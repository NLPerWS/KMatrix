from copy import deepcopy
import os
from typing import Any, Dict, List, Literal
from more_itertools import windowed
import json
import traceback
from kninjllm.llm_common.document import Document
from kninjllm.llm_common.component import component
from kninjllm.llm_store.elasticsearch_document_store import ElasticsearchDocumentStore
from kninjllm.llm_utils.common_utils import EmbeddingByRetriever

@component
class SaveToElasticSearchDB:

    def __init__(
        self,
        host:str,
        index:str,
        username:str,
        password:str,
        ebbedding_retriever_nameList:List[str] = []
    ):
        self.index = index
        self.store = ElasticsearchDocumentStore(hosts = host,index=index,basic_auth=(username,password))
        self.ebbedding_retriever_nameList = ebbedding_retriever_nameList

    @component.output_types(final_result=Dict[str,Any])
    def run(self, documents:List[Dict[str,Any]]):
        
        # embedding
        documents = EmbeddingByRetriever(documents,self.ebbedding_retriever_nameList)
        
        try:
            documents = list(map(lambda x:Document(**x),documents))
            self.store.write_documents(documents=documents)
        except:
            traceback.print_exc()
        
        return {"final_result":f"save to ES ,index is {self.index}"}