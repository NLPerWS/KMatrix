import os
import time
from typing import Any, List, Literal,Dict
from kninjllm.llm_common.component import component
from kninjllm.llm_utils.common_utils import calculate_hash

@component
class TextChunk_split_rules:

    def __init__(
        self,
        max_length: int = 500,
        split_chars:List[str]=["\n","。","？","！","，",".","?","!",","]
    ):
        self.max_length = max_length
        self.split_chars = split_chars
        

    def split_documents_by_array_n(self, documents):
        split_str_list = []
        for doc in documents:
            doc_id = doc['id']
            id, text, title = doc['content'].split('\t')
            source = doc['source']
            result = []
            current_sentence = ''
            for char in text:
                current_sentence += char
                if char in self.split_chars or len(current_sentence) >= self.max_length:
                    result.append(current_sentence)
                    current_sentence = ''
            if current_sentence:
                result.append(current_sentence)

            current_merged_sentence = ''
            for index,sentence in enumerate(result):
                if len(current_merged_sentence) + len(sentence) <= self.max_length:
                    current_merged_sentence += sentence
                else:
                    new_id = doc_id + "_" +calculate_hash([current_merged_sentence])+str(index)
                    split_str_list.append({
                        "doc_id":new_id,
                        "text":new_id+"\t"+current_merged_sentence +"\t"+ title,
                        "source":source
                    })
                    current_merged_sentence = sentence
            if current_merged_sentence:
                new_id = doc_id + "_" +calculate_hash([current_merged_sentence])
                split_str_list.append({
                    "doc_id":new_id,
                    "text":new_id+"\t"+current_merged_sentence +"\t"+ title,
                    "source":source
                })

        new_documents = []
        for index,s in enumerate(split_str_list):
            id, text, title = s['text'].split('\t')
            new_documents.append({"id":s['doc_id'],"content":s['text'],"source":s['source']})

        return new_documents
        
        
    def split_csv(self,input_list):
        
        split_str_list = []
        cur_time_str = str(time.time())
        for index,input in enumerate(input_list):
            split_str_list.append(input)
        return split_str_list

    @component.output_types(documents=List[Dict[str, Any]])
    def run(self, documents:List[Dict[str, Any]]=[]):

        """
            Separated in the order of the character list based on the specified character list and length.
            The length of each separated sentence cannot exceed the specified length, and the end of each sentence must end with the specified character list element.
            If the separated sentences are too long, continue to separate them according to the specified character list until the separated sentences do not exceed the specified length.
            And try to keep each separated sentence as long as possible: if there are consecutive sentences that do not add up to the maximum length, merge them into one sentence.
        """        
        
        new_documents = self.split_documents_by_array_n(documents)
        
        return {"documents":new_documents}
