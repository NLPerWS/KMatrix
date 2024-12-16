import hashlib
import io
from dataclasses import asdict, dataclass, field, fields
import json
from typing import Any, Dict, List, Optional, Union
import numpy as np
from numpy import ndarray
from pandas import DataFrame, read_json

from kninjllm.llm_dataclasses.byte_stream import ByteStream


class _BackwardCompatible(type):
    """
    Metaclass that handles Document backward compatibility.
    """

    def __call__(cls, *args, **kwargs):
        
        """
        Called before Document.__init__, will remap legacy fields to new ones.
        Also handles building a Document from a flattened dictionary.
        """
        
        content = kwargs.get("content")
        if isinstance(content, DataFrame):
            kwargs["dataframe"] = content
            del kwargs["content"]

        
        if "content_type" in kwargs:
            del kwargs["content_type"]

        
        if isinstance(embedding := kwargs.get("embedding"), ndarray):
            kwargs["embedding"] = embedding.tolist()
        if isinstance(BGE_embedding := kwargs.get("BGE_embedding"), ndarray):
            kwargs["BGE_embedding"] = BGE_embedding.tolist()   
        if isinstance(DPR_embedding := kwargs.get("DPR_embedding"), ndarray):
            kwargs["DPR_embedding"] = DPR_embedding.tolist()   
        if isinstance(contriever_embedding := kwargs.get("contriever_embedding"), ndarray):
            kwargs["contriever_embedding"] = contriever_embedding.tolist()   
        if isinstance(BERT_embedding := kwargs.get("BERT_embedding"), ndarray):
            kwargs["BERT_embedding"] = BERT_embedding.tolist()   
        if isinstance(E5_embedding := kwargs.get("E5_embedding"), ndarray):
            kwargs["E5_embedding"] = E5_embedding.tolist()   
            
            
        
        if "id_hash_keys" in kwargs:
            del kwargs["id_hash_keys"]

        return super().__call__(*args, **kwargs)


@dataclass
class Document(metaclass=_BackwardCompatible):
    """
    Base data class containing some data to be queried.

    Can contain text snippets, tables, and file paths to images or audios. Documents can be sorted by score and saved
    to/from dictionary and JSON.

    :param id: Unique identifier for the document. When not set, it's generated based on the Document fields' values.
    :param content: Text of the document, if the document contains text.
    :param dataframe: Pandas dataframe with the document's content, if the document contains tabular data.
    :param blob: Binary data associated with the document, if the document has any binary data associated with it.
    :param meta: Additional custom metadata for the document. Must be JSON-serializable.
    :param score: Score of the document. Used for ranking, usually assigned by retrievers.
    """

    id: str = field(default="")
    content: Optional[str] = field(default=None)
    dataframe: Optional[DataFrame] = field(default=None)
    blob: Optional[ByteStream] = field(default=None)
    meta: Dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = field(default=None)
    embedding: Optional[List[float]] = field(default=None)
    BGE_embedding: Optional[List[Any]] = field(default=None)
    contriever_embedding: Optional[List[Any]] = field(default=None)
    DPR_embedding: Optional[List[Any]] = field(default=None)
    BERT_embedding: Optional[List[Any]] = field(default=None)
    E5_embedding: Optional[List[Any]] = field(default=None)
    
    
    header: Optional[List[str]] = field(default=None)
    rows: Optional[List[List[str]]] = field(default=None)
    triples: Optional[Dict[str,Any]] = field(default=None)
    source: Optional[str] = field(default=None)
    
    def __repr__(self):
        fields = []
        if self.content is not None:
            fields.append(
                f"content: '{self.content}'" if len(self.content) < 100 else f"content: '{self.content[:100]}...'"
            )
        if self.dataframe is not None:
            fields.append(f"dataframe: {self.dataframe.shape}")
        if self.blob is not None:
            fields.append(f"blob: {len(self.blob.data)} bytes")
        if len(self.meta) > 0:
            fields.append(f"meta: {self.meta}")
        if self.score is not None:
            fields.append(f"score: {self.score}")
        if self.embedding is not None:
            fields.append(f"embedding: vector of size {len(self.embedding)}")
        if self.BGE_embedding is not None:
            fields.append(f"BGE_embedding: vector of size {len(self.BGE_embedding)}")   
        if self.DPR_embedding is not None:
            fields.append(f"DPR_embedding: vector of size {len(self.DPR_embedding)}")   
        if self.triples is not None:
            fields.append(f"triples: vector of size {len(self.triples)}")   
        if self.source is not None:
            fields.append(f"source: size {len(self.source)}")   
            
        if self.contriever_embedding is not None:
            fields.append(f"contriever_embedding: vector of size {len(self.contriever_embedding)}")   
            
        if self.BERT_embedding is not None:
            fields.append(f"BERT_embedding: vector of size {len(self.BERT_embedding)}")   
        if self.E5_embedding is not None:
            fields.append(f"E5_embedding: vector of size {len(self.E5_embedding)}")   
              
        fields_str = ", ".join(fields)
        return f"{self.__class__.__name__}(id={self.id}, {fields_str})"

    def __eq__(self, other):
        """
        Compares Documents for equality.

        Two Documents are considered equals if their dictionary representation is identical.
        """
        if type(self) != type(other):
            return False
        return self.to_dict() == other.to_dict()

    def __post_init__(self):
        """
        Generate the ID based on the init parameters.
        """
        
        self.id = self.id or self._create_id()

    def _create_id(self):
        """
        Creates a hash of the given content that acts as the document's ID.
        """
        text = self.content or None
        dataframe = self.dataframe.to_json() if self.dataframe is not None else None
        blob = self.blob.data if self.blob is not None else None
        mime_type = self.blob.mime_type if self.blob is not None else None
        meta = self.meta or {}
        data = f"{text}{dataframe}{blob}{mime_type}{meta}"
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    def to_dict(self, flatten=True) -> Dict[str, Any]:
        """
        Converts Document into a dictionary.

        `dataframe` and `blob` fields are converted to JSON-serializable types.
        """
        data = asdict(self)
        if (dataframe := data.get("dataframe")) is not None:
            data["dataframe"] = dataframe.to_json()
        if (blob := data.get("blob")) is not None:
            data["blob"] = {"data": list(blob["data"]), "mime_type": blob["mime_type"]}

        if flatten:
            meta = data.pop("meta")
            return {**data, **meta}

        return data

    def to_dict_embarray(self, flatten=True) -> Dict[str, Any]:
        """
        Converts Document into a dictionary.

        `dataframe` and `blob` fields are converted to JSON-serializable types.
        """
        data = asdict(self)
        if (dataframe := data.get("dataframe")) is not None:
            data["dataframe"] = dataframe.to_json()
        if (blob := data.get("blob")) is not None:
            data["blob"] = {"data": list(blob["data"]), "mime_type": blob["mime_type"]}
            
        if data['DPR_embedding'] != None:
            data['DPR_embedding'] = np.array(data['DPR_embedding'])
            
        if data['BGE_embedding'] != None:
            data['BGE_embedding'] = np.array(data['BGE_embedding'])
            
        if data['contriever_embedding'] != None:
            data['contriever_embedding'] = np.array(data['contriever_embedding'])
            
        if data['BERT_embedding'] != None:
            data['BERT_embedding'] = np.array(data['BERT_embedding'])
        if data['E5_embedding'] != None:
            data['E5_embedding'] = np.array(data['E5_embedding'])

        if flatten:
            meta = data.pop("meta")
            return {**data, **meta}

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """
        Creates a new Document object from a dictionary.

        The `dataframe` and `blob` fields are converted to their original types.
        """
        if (dataframe := data.get("dataframe")) is not None:
            data["dataframe"] = read_json(io.StringIO(dataframe))
        if blob := data.get("blob"):
            data["blob"] = ByteStream(data=bytes(blob["data"]), mime_type=blob["mime_type"])
        
        
        
        meta = data.pop("meta", {})
        
        
        
        flatten_meta = {}
        legacy_fields = ["content_type", "id_hash_keys"]
        document_fields = legacy_fields + [f.name for f in fields(cls)]
        for key in list(data.keys()):
            if key not in document_fields:
                flatten_meta[key] = data.pop(key)

        
        if meta and flatten_meta:
            raise ValueError(
                "You can pass either the 'meta' parameter or flattened metadata keys as keyword arguments, "
                "but currently you're passing both. Pass either the 'meta' parameter or flattened metadata keys."
            )

        
        return cls(**data, meta={**meta, **flatten_meta})

    @property
    def content_type(self):
        """
        Returns the type of the content for the document.

        This is necessary to keep backward compatibility with 1.x.

        :raises ValueError:
            If both `text` and `dataframe` fields are set or both are missing.
        """
        if self.content is not None and self.dataframe is not None:
            raise ValueError("Both text and dataframe are set.")

        if self.content is not None:
            return "text"
        elif self.dataframe is not None:
            return "table"
        raise ValueError("Neither text nor dataframe is set.")
