from copy import deepcopy
from typing import List, Literal
from more_itertools import windowed

from kninjllm.llm_common.document import Document
from kninjllm.llm_common.component import component

@component
class DocumentSplitter:
    """
    Splits a list of text documents into a list of text documents with shorter texts.

    Splitting documents with long texts is a common preprocessing step during indexing.
    This allows Embedders to create significant semantic representations
    and avoids exceeding the maximum context length of language models.
    """

    def __init__(
        self,
        split_by: Literal["word", "sentence", "page", "passage","charListAndmaxLength"] = "charListAndmaxLength",
        split_length: int = 200,
        split_overlap: int = 0,
        split_chars:List[str]=["\n","。","？","！","，",".","?","!",","]
    ):
        """
        :param split_by: The unit by which the document should be split. Choose from "word" for splitting by " ",
            "sentence" for splitting by ".", "page" for splitting by "\\f" or "passage" for splitting by "\\n\\n".
        :param split_length: The maximum number of units in each split.
        :param split_overlap: The number of units that each split should overlap.
        """
        self.split_by = split_by
        if split_by not in ["word", "sentence", "page", "passage","charListAndmaxLength"]:
            raise ValueError("split_by must be one of 'word', 'sentence', 'page','passage', or 'charListAndmaxLength'.")
        if split_length <= 0:
            raise ValueError("split_length must be greater than 0.")
        self.split_length = split_length
        if split_overlap < 0:
            raise ValueError("split_overlap must be greater than or equal to 0.")
        self.split_overlap = split_overlap
        self.split_chars = split_chars

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        """
        Splits documents by the unit expressed in `split_by`, with a length of `split_length`
        and an overlap of `split_overlap`.

        :param documents: The documents to split.

        :returns: A dictionary with the following key:
            - `documents`: List of documents with the split texts. A metadata field "source_id" is added to each
            document to keep track of the original document that was split. Other metadata are copied from the original
            document.

        :raises TypeError: if the input is not a list of Documents.
        :raises ValueError: if the content of a document is None.
        """

        if not isinstance(documents, list) or (documents and not isinstance(documents[0], Document)):
            raise TypeError("DocumentSplitter expects a List of Documents as input.")

        split_docs = []
        for doc in documents:
            if doc.content is None:
                raise ValueError(
                    f"DocumentSplitter only works with text documents but document.content for document ID {doc.id} is None."
                )
            units = self._split_into_units(doc.content, self.split_by)
            if self.split_by == "charListAndmaxLength":
                text_splits = self.split_and_merge_string(doc.content, self.split_chars, self.split_length)
            else:
                text_splits = self._concatenate_units(units, self.split_length, self.split_overlap)
            metadata = deepcopy(doc.meta)
            metadata["source_id"] = doc.id
            split_docs += [Document(content=txt, meta=metadata) for txt in text_splits]
        return {"documents": split_docs}

    def _split_into_units(self, text: str, split_by: Literal["word", "sentence", "passage", "page"]) -> List[str]:
        if split_by == "page":
            split_at = "\f"
        elif split_by == "passage":
            split_at = "\n\n"
        elif split_by == "sentence":
            split_at = "."
        elif split_by == "word":
            split_at = " "
        elif split_by == "charListAndmaxLength":
            split_at = " "
        else:
            raise NotImplementedError(
                "DocumentSplitter only supports 'word', 'sentence', 'page' or 'passage' split_by options."
            )
        units = text.split(split_at)
        
        for i in range(len(units) - 1):
            units[i] += split_at
        return units

    def _concatenate_units(self, elements: List[str], split_length: int, split_overlap: int) -> List[str]:
        """
        Concatenates the elements into parts of split_length units.
        """
        text_splits = []
        segments = windowed(elements, n=split_length, step=split_length - split_overlap)
        for seg in segments:
            current_units = [unit for unit in seg if unit is not None]
            txt = "".join(current_units)
            if len(txt) > 0:
                text_splits.append(txt)
        return text_splits

    
    def split_and_merge_string(self,input_string, char_list, max_length):
        """
            Lengths are separated in the order of the character list based on the specified character list and length.
            The length of each separated sentence cannot exceed the specified length, and the end of each sentence must end with the specified character list element.
            If the separated sentences are too long, continue to separate them according to the specified character list until the separated sentences do not exceed the specified length.
            And try to keep each separated sentence as long as possible: if there are consecutive sentences that do not add up to the maximum length, merge them into one sentence.
        """        
        result = []
        current_sentence = ''
        for char in input_string:
            current_sentence += char
            if char in char_list or len(current_sentence) >= max_length:
                result.append(current_sentence)
                current_sentence = ''
        if current_sentence:
            result.append(current_sentence)
        merged_result = []
        current_merged_sentence = ''
        for sentence in result:
            if len(current_merged_sentence) + len(sentence) <= max_length:
                current_merged_sentence += sentence
            else:
                merged_result.append(current_merged_sentence)
                current_merged_sentence = sentence
        if current_merged_sentence:
            merged_result.append(current_merged_sentence)
        return merged_result
