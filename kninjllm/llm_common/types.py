from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Type, TypeVar, get_args,Dict,Optional,Protocol
from typing_extensions import Annotated, TypeAlias  # Python 3.8 compatibility

from kninjllm.llm_common.document import Document

KNINJLLM_VARIADIC_ANNOTATION = "__kninjllm__variadic_t"

# # Generic type variable used in the Variadic container
T = TypeVar("T")


# Variadic is a custom annotation type we use to mark input types.
# This type doesn't do anything else than "marking" the contained
# type so it can be used in the `InputSocket` creation where we
# check that its annotation equals to CANALS_VARIADIC_ANNOTATION
Variadic: TypeAlias = Annotated[Iterable[T], KNINJLLM_VARIADIC_ANNOTATION]


class _empty:
    """Custom object for marking InputSocket.default_value as not set."""


@dataclass
class InputSocket:
    """
    Represents an input of a `Component`.

    :param name:
        The name of the input.
    :param type:
        The type of the input.
    :param default_value:
        The default value of the input. If not set, the input is mandatory.
    :param is_variadic:
        Whether the input is variadic or not.
    :param senders:
        The list of components that send data to this input.
    """

    name: str
    type: Type
    default_value: Any = _empty
    is_variadic: bool = field(init=False)
    senders: List[str] = field(default_factory=list)

    @property
    def is_mandatory(self):
        return self.default_value == _empty

    def __post_init__(self):
        try:
            # __metadata__ is a tuple
            self.is_variadic = self.type.__metadata__[0] == KNINJLLM_VARIADIC_ANNOTATION
        except AttributeError:
            self.is_variadic = False
        if self.is_variadic:
            # We need to "unpack" the type inside the Variadic annotation,
            # otherwise the pipeline connection api will try to match
            # `Annotated[type, KNINJLLM_VARIADIC_ANNOTATION]`.
            #
            # Note1: Variadic is expressed as an annotation of one single type,
            # so the return value of get_args will always be a one-item tuple.
            #
            # Note2: a pipeline always passes a list of items when a component
            # input is declared as Variadic, so the type itself always wraps
            # an iterable of the declared type. For example, Variadic[int]
            # is eventually an alias for Iterable[int]. Since we're interested
            # in getting the inner type `int`, we call `get_args` twice: the
            # first time to get `List[int]` out of `Variadic`, the second time
            # to get `int` out of `List[int]`.
            self.type = get_args(get_args(self.type)[0])[0]


@dataclass
class OutputSocket:
    """
    Represents an output of a `Component`.

    :param name:
        The name of the output.
    :param type:
        The type of the output.
    :param receivers:
        The list of components that receive the output of this component.
    """

    name: str
    type: type
    receivers: List[str] = field(default_factory=list)

class DuplicatePolicy(Enum):
    NONE = "none"
    SKIP = "skip"
    OVERWRITE = "overwrite"
    FAIL = "fail"
    
class DocumentStore(Protocol):
    """
    Stores Documents to be used by the components of a Pipeline.

    Classes implementing this protocol often store the documents permanently and allow specialized components to
    perform retrieval on them, either by embedding, by keyword, hybrid, and so on, depending on the backend used.

    In order to retrieve documents, consider using a Retriever that supports the DocumentStore implementation that
    you're using.
    """

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this store to a dictionary.
        """
        ...

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentStore":
        """
        Deserializes the store from a dictionary.
        """
        ...

    def count_documents(self) -> int:
        """
        Returns the number of documents stored.
        """
        ...

    def filter_documents(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Returns the documents that match the filters provided.

        Filters are defined as nested dictionaries that can be of two types:
        - Comparison
        - Logic

        Comparison dictionaries must contain the keys:

        - `field`
        - `operator`
        - `value`

        Logic dictionaries must contain the keys:

        - `operator`
        - `conditions`

        The `conditions` key must be a list of dictionaries, either of type Comparison or Logic.

        The `operator` value in Comparison dictionaries must be one of:

        - `==`
        - `!=`
        - `>`
        - `>=`
        - `<`
        - `<=`
        - `in`
        - `not in`

        The `operator` values in Logic dictionaries must be one of:

        - `NOT`
        - `OR`
        - `AND`


        A simple filter:
        ```python
        filters = {"field": "meta.type", "operator": "==", "value": "article"}
        ```

        A more complex filter:
        ```python
        filters = {
            "operator": "AND",
            "conditions": [
                {"field": "meta.type", "operator": "==", "value": "article"},
                {"field": "meta.date", "operator": ">=", "value": 1420066800},
                {"field": "meta.date", "operator": "<", "value": 1609455600},
                {"field": "meta.rating", "operator": ">=", "value": 3},
                {
                    "operator": "OR",
                    "conditions": [
                        {"field": "meta.genre", "operator": "in", "value": ["economy", "politics"]},
                        {"field": "meta.publisher", "operator": "==", "value": "nytimes"},
                    ],
                },
            ],
        }

        :param filters: the filters to apply to the document list.
        :returns: a list of Documents that match the given filters.
        """
        ...

    def write_documents(self, documents: List[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int:
        """
        Writes Documents into the DocumentStore.

        :param documents: a list of Document objects.
        :param policy: the policy to apply when a Document with the same id already exists in the DocumentStore.
            - `DuplicatePolicy.NONE`: Default policy, behaviour depends on the Document Store.
            - `DuplicatePolicy.SKIP`: If a Document with the same id already exists, it is skipped and not written.
            - `DuplicatePolicy.OVERWRITE`: If a Document with the same id already exists, it is overwritten.
            - `DuplicatePolicy.FAIL`: If a Document with the same id already exists, an error is raised.
        :raises DuplicateError: If `policy` is set to `DuplicatePolicy.FAIL` and a Document with the same id already exists.
        :returns: The number of Documents written.
            If `DuplicatePolicy.OVERWRITE` is used, this number is always equal to the number of documents in input.
            If `DuplicatePolicy.SKIP` is used, this number can be lower than the number of documents in the input list.
        """
        ...

    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Deletes all documents with a matching document_ids from the DocumentStore.

        Fails with `MissingDocumentError` if no document with this id is present in the DocumentStore.

        :param document_ids: the object_ids to delete
        """
        ...
