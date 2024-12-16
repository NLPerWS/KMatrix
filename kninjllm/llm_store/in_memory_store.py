import re
from typing import Any, Dict, Iterable, List, Literal, Optional

import numpy as np
from tqdm.auto import tqdm

from kninjllm.llm_common.serialization import default_from_dict, default_to_dict
from kninjllm.llm_common.document import Document
from kninjllm.llm_common.errors import DocumentStoreError, DuplicateDocumentError
from kninjllm.llm_store.types import DuplicatePolicy
from kninjllm.llm_utils.expit_utils import expit
from kninjllm.llm_store.utils.filter_utils_conver import convert, document_matches_filter
import jieba
import heapq
from rank_bm25 import BM25Okapi

# document scores are essentially unbounded and will be scaled to values between 0 and 1 if scale_score is set to
# True (default). Scaling uses the expit function (inverse of the logit function) after applying a scaling factor
# (e.g., BM25_SCALING_FACTOR for the bm25_retrieval method).
# Larger scaling factor decreases scaled scores. For example, an input of 10 is scaled to 0.99 with BM25_SCALING_FACTOR=2
# but to 0.78 with BM25_SCALING_FACTOR=8 (default). The defaults were chosen empirically. Increase the default if most
# unscaled scores are larger than expected (>30) and otherwise would incorrectly all be mapped to scores ~1.
BM25_SCALING_FACTOR = 8
DOT_PRODUCT_SCALING_FACTOR = 100


class InMemoryDocumentStore:
    """
    Stores data in-memory. It's ephemeral and cannot be saved to disk.
    """

    def __init__(
        self,
    ):
        self.storage: Dict[str, Document] = {}
        self.bm25 = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            bm25_tokenization_regex=self._bm25_tokenization_regex,
            bm25_algorithm=self.bm25_algorithm.__name__,
            bm25_parameters=self.bm25_parameters,
            embedding_similarity_function=self.embedding_similarity_function,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InMemoryDocumentStore":
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized component.
        """
        return default_from_dict(cls, data)

    def count_documents(self) -> int:
        """
        Returns the number of how many documents are present in the DocumentStore.
        """
        return len(self.storage.keys())

    def filter_documents(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Returns the documents that match the filters provided.

        For a detailed specification of the filters, refer to the DocumentStore.filter_documents() protocol documentation.

        :param filters: The filters to apply to the document list.
        :returns: A list of Documents that match the given filters.
        """
        if filters:
            if "operator" not in filters and "conditions" not in filters:
                filters = convert(filters)
            return [doc for doc in self.storage.values() if document_matches_filter(filters=filters, document=doc)]
        return list(self.storage.values())

    def write_documents(self, documents: List[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int:
        """
        Refer to the DocumentStore.write_documents() protocol documentation.

        If `policy` is set to `DuplicatePolicy.NONE` defaults to `DuplicatePolicy.FAIL`.
        """
        if (
            not isinstance(documents, Iterable)
            or isinstance(documents, str)
            or any(not isinstance(doc, Document) for doc in documents)
        ):
            raise ValueError("Please provide a list of Documents.")


        if policy == DuplicatePolicy.NONE:
            policy = DuplicatePolicy.FAIL

        written_documents = len(documents)
        for document in documents:
            if policy != DuplicatePolicy.OVERWRITE and document.id in self.storage.keys():
                if policy == DuplicatePolicy.FAIL:
                    # raise DuplicateDocumentError(f"ID '{document.id}' already exists.")
                    print("ID '{document_id}' already exists")
                    written_documents -= 1
                    continue
                if policy == DuplicatePolicy.SKIP:
                    print("ID '{document_id}' already exists", document_id=document.id)
                    written_documents -= 1
                    continue
            self.storage[document.id] = document
        return written_documents

    def do_embedding(self):
        
        try:
            tokenized_corpus = [list(filter(lambda x: x.replace(" ","") != "", list(jieba.cut(doc.content.lower(), cut_all=False)))) for doc in tqdm(list(self.storage.values()), desc="Tokenizing corpus")]
            self.bm25 = BM25Okapi(tokenized_corpus)
        except:
            self.bm25 = None

    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Deletes all documents with matching document_ids from the DocumentStore.
        :param document_ids: The object_ids to delete.
        """
        for doc_id in document_ids:
            if doc_id not in self.storage.keys():
                continue
            del self.storage[doc_id]

    def delete_all_documents(self) -> None:
        """
        Deletes all documents
        """
        self.storage = {}

    def _bm25_retrieval(
        self, query: str, top_k: int = 10
    ) -> List[Document]:
        
        if not query:
            return []
        if self.bm25 is None:
            return []
        
        # tokenized_query = query.lower().split(" ")
        tokenized_query = list(filter(lambda x: x.replace(" ","") != "", list(jieba.cut(query, cut_all=False))))
        
        doc_scores = self.bm25.get_scores(tokenized_query)
        topk_scores_indices = heapq.nlargest(top_k, enumerate(doc_scores), key=lambda x: x[1])
        return_documents = [(list(self.storage.values())[i], score) for i, score in topk_scores_indices]

        for document, score in return_documents:
            document.score = score
        
        return [document for document, _ in return_documents]


    def embedding_retrieval(
        self,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        scale_score: bool = False,
        return_embedding: bool = False,
    ) -> List[Document]:
        """
        Retrieves documents that are most similar to the query embedding using a vector similarity metric.

        :param query_embedding: Embedding of the query.
        :param filters: A dictionary with filters to narrow down the search space.
        :param top_k: The number of top documents to retrieve. Default is 10.
        :param scale_score: Whether to scale the scores of the retrieved Documents. Default is False.
        :param return_embedding: Whether to return the embedding of the retrieved Documents. Default is False.
        :returns: A list of the top_k documents most relevant to the query.
        """
        if len(query_embedding) == 0 or not isinstance(query_embedding[0], float):
            raise ValueError("query_embedding should be a non-empty list of floats.")

        filters = filters or {}
        all_documents = self.filter_documents(filters=filters)

        documents_with_embeddings = [doc for doc in all_documents if doc.embedding is not None]
        if len(documents_with_embeddings) == 0:
            print(
                "No Documents found with embeddings. Returning empty list. "
                "To generate embeddings, use a DocumentEmbedder."
            )
            return []
        elif len(documents_with_embeddings) < len(all_documents):
            print(
                "Skipping some Documents that don't have an embedding. "
                "To generate embeddings, use a DocumentEmbedder."
            )

        scores = self._compute_query_embedding_similarity_scores(
            embedding=query_embedding, documents=documents_with_embeddings, scale_score=scale_score
        )

        # create Documents with the similarity score for the top k results
        top_documents = []
        for doc, score in sorted(zip(documents_with_embeddings, scores), key=lambda x: x[1], reverse=True)[:top_k]:
            doc_fields = doc.to_dict()
            doc_fields["score"] = score
            if return_embedding is False:
                doc_fields["embedding"] = None
            top_documents.append(Document.from_dict(doc_fields))

        return top_documents

    def _compute_query_embedding_similarity_scores(
        self, embedding: List[float], documents: List[Document], scale_score: bool = False
    ) -> List[float]:
        """
        Computes the similarity scores between the query embedding and the embeddings of the documents.

        :param embedding: Embedding of the query.
        :param documents: A list of Documents.
        :param scale_score: Whether to scale the scores of the Documents. Default is False.
        :returns: A list of scores.
        """

        query_embedding = np.array(embedding)
        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(a=query_embedding, axis=0)

        try:
            document_embeddings = np.array([doc.embedding for doc in documents])
        except ValueError as e:
            if "inhomogeneous shape" in str(e):
                raise DocumentStoreError(
                    "The embedding size of all Documents should be the same. "
                    "Please make sure that the Documents have been embedded with the same model."
                ) from e
            raise e
        if document_embeddings.ndim == 1:
            document_embeddings = np.expand_dims(a=document_embeddings, axis=0)

        if self.embedding_similarity_function == "cosine":
            # cosine similarity is a normed dot product
            query_embedding /= np.linalg.norm(x=query_embedding, axis=1, keepdims=True)
            document_embeddings /= np.linalg.norm(x=document_embeddings, axis=1, keepdims=True)

        try:
            scores = np.dot(a=query_embedding, b=document_embeddings.T)[0].tolist()
        except ValueError as e:
            if "shapes" in str(e) and "not aligned" in str(e):
                raise DocumentStoreError(
                    "The embedding size of the query should be the same as the embedding size of the Documents. "
                    "Please make sure that the query has been embedded with the same model as the Documents."
                ) from e
            raise e

        if scale_score:
            if self.embedding_similarity_function == "dot_product":
                scores = [expit(float(score / DOT_PRODUCT_SCALING_FACTOR)) for score in scores]
            elif self.embedding_similarity_function == "cosine":
                scores = [(score + 1) / 2 for score in scores]

        return scores
