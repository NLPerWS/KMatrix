
import json
import os
from typing import Any, Dict, List
from kninjllm.llm_common.component import component
import pandas as pd
from kninjllm.llm_utils.common_utils import calculate_hash
from kninjllm.llm_utils.common_utils import loadKnowledgeByCatch
import csv
import PyPDF2
from docx import Document

@component
class TextPreprecess:
    def __init__(self):
        pass
    
    def check_tab_separated_length(self,arr):
        for line in arr:
            parts = line.split('\t')
            if len(parts) != 3:
                return False
        return True
    
    def parse_file(self,file_path,chunk_size):
        file_extension = os.path.splitext(file_path)[1].lower()
        text = ""
        if file_extension == '.txt' or file_extension == '.md':
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
        elif file_extension == '.pdf':
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text()
            
        elif file_extension == '.tsv':
            with open(file_path, 'r', encoding='utf-8') as file:
                reader = csv.reader(file, delimiter='\t')
                text = '\n'.join('\t'.join(row) for row in reader)
        elif file_extension == '.doc' or file_extension == '.docx':
            doc = Document(file_path)
            text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        else:
            raise ValueError('Unsupported file format')

        text = text.replace("\t"," ")
        return [text]
    
    
    @component.output_types(documents=List[Dict[str, Any]])
    def run(
        self,
        path:Any,
    ):
        if isinstance(path,dict):
            this_path = path['knowledge_path']
        elif isinstance(path,str):
            this_path = path
        else:
            raise ValueError("Preprecess paramter error ...")
        
        finalJsonObjList = []
        if this_path != "":
            if os.path.isdir(this_path):
                for fileName in os.listdir(this_path):
                    filepath = os.path.join(this_path,fileName)     
                    chunks = self.parse_file(filepath,500)
                    for index,s in enumerate(chunks):
                        id = fileName+"_"+calculate_hash([s])
                        content = id+"\t"+s+"\t"+"None"
                        finalJsonObjList.append({"id":id,"content":content,"source":"文本"})
                    
            else:
                filepath = this_path
                fileName = os.path.basename(filepath)
                chunks = self.parse_file(filepath,500)
                for index,s in enumerate(chunks):
                    id = fileName+"_"+calculate_hash([s])
                    content = id+"\t"+s+"\t"+"None"
                    finalJsonObjList.append({"id":id,"content":content,"source":"文本"})
            
        else:
            knowledge = loadKnowledgeByCatch(knowledge_path="",elasticIndex=path['knowledge_elasticIndex'],tag="文本")
            finalJsonObjList.extend(knowledge)
        
        return {"documents":finalJsonObjList}
    