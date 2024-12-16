from typing import Any, Dict, List, Optional

from kninjllm.llm_utils.common_utils import loadKnowledgeByCatch
from kninjllm.llm_common.errors import DeserializationError
from kninjllm.llm_common.document import Document
from kninjllm.llm_common.component import component
from kninjllm.llm_common.serialization import default_from_dict, default_to_dict
from kninjllm.llm_store.in_memory_store import InMemoryDocumentStore
from root_config import RootConfig

@component
class InMemoryBM25Retriever:

    def __init__(
        self,
        top_k = 10,
        model_path:str="",
        executeType:str="",
        filters: Optional[Dict[str, Any]] = None,
        scale_score: bool = False,
        document_store: InMemoryDocumentStore = None
    ):
        self.logSaver = RootConfig.logSaver
        self.document_store = document_store
        self.filters = filters
        self.top_k = top_k
        self.scale_score = scale_score

        if top_k <= 0:
            raise ValueError(f"top_k must be greater than 0. Currently, the top_k is {top_k}")


    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"document_store": type(self.document_store).__name__}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        docstore = self.document_store.to_dict()
        # print("docstore",docstore)
        return default_to_dict(
            self, document_store=docstore, filters=self.filters, top_k=self.top_k, scale_score=self.scale_score
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InMemoryBM25Retriever":
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized component.
        """
        init_params = data.get("init_parameters", {})
        if "document_store" not in init_params:
            raise DeserializationError("Missing 'document_store' in serialization data")
        if "type" not in init_params["document_store"]:
            raise DeserializationError("Missing 'type' in document store's serialization data")
        data["init_parameters"]["document_store"] = InMemoryDocumentStore.from_dict(
            data["init_parameters"]["document_store"]
        )
        return default_from_dict(cls, data)

  
    @component.output_types(final_result=List[List[Dict[str,Any]]])
    def run(
        self,
        query_obj: Dict[str,Any] = {},
        knowledge_info : Dict[str,Any] = {},
        test_data_info: Dict[str,Any] = {},
        train_data_info : Dict[str,Any] = {},
        dev_data_info : Dict[str,Any] = {},
    ):
        print("-------------------------------BM25_Retriever-----------------------------------")
        
        if knowledge_info != {} and self.document_store == None:
            knowledge_path = knowledge_info['knowledge_path']
            knowledge_elasticIndex = knowledge_info['knowledge_elasticIndex']
            knowledge_tag = knowledge_info['knowledge_tag']
            knowledge = loadKnowledgeByCatch(knowledge_path=knowledge_path,elasticIndex=knowledge_elasticIndex,tag=knowledge_tag)
            catch_flag = False
            for catch in RootConfig.tempPipeLineKnowledgeCatch:
                if catch['path'] == knowledge_path+knowledge_elasticIndex+"_bm25_obj":
                    catch_flag = True
                    self.document_store = catch['data']
            if catch_flag == False:
                self.document_store = InMemoryDocumentStore()
                doc_write_list = list(map(lambda x:Document(**x),knowledge))
                self.document_store.write_documents(doc_write_list)
                self.document_store.do_embedding()
                RootConfig.tempPipeLineKnowledgeCatch.append({"path":knowledge_path+knowledge_elasticIndex+"_bm25_obj","data":self.document_store})

        if query_obj == {}:
            return {"final_result": [[]]}
        
        if self.document_store == None:
            return {"final_result": [[]]}
        
        if self.logSaver is not None:
            self.logSaver.writeStrToLog("Function -> InMemoryBM25Retriever -> run | Given the search text, return the search content ")
            self.logSaver.writeStrToLog("search input -> :querys: "+str(query_obj))

        final_docs_dicts_list = []
        docs = self.document_store._bm25_retrieval(query=query_obj['question'], top_k=self.top_k)
        selected_keys = ['id', 'content', 'source','score']
        docs_dicts_list = list(map(lambda x: {k: x.__dict__[k] for k in selected_keys if k in x.__dict__}, docs))
        final_docs_dicts_list.append(docs_dicts_list)

        if self.logSaver is not None:
            self.logSaver.writeStrToLog("search returned -> : final_result: "+str(final_docs_dicts_list))
        return {"final_result": final_docs_dicts_list}
    