
import json
import os
from typing import Any, Dict, List
from kninjllm.llm_common.component import component
import pandas as pd
from kninjllm.llm_utils.common_utils import calculate_hash

@component
class TablePreprecess:
    def __init__(self):
        pass
    
    def execute_one_file(self,filepath):
        finalJsonObjList = []
        
        if filepath.endswith(".xlsx") or filepath.endswith(".xls"):
            xl = pd.ExcelFile(filepath)
            sheet_names = xl.sheet_names
            for sheet_name in sheet_names:
                headers = []
                rows = []
                if type == 'row_column':
                    df = pd.read_excel(filepath, sheet_name=sheet_name)
                    for index,row in df.iterrows():
                        temp_data_list = []
                        skip_first_column = True
                        for column_name, cell_value in row.items():
                            if skip_first_column:
                                skip_first_column = False
                                continue
                            temp_data_list.append(cell_value)
                            if column_name not in headers:
                                headers.append(column_name)
                        rows.append(temp_data_list)

                elif type == 'column':
                    df = pd.read_excel(filepath, sheet_name=sheet_name)
                    for index,row in df.iterrows():
                        temp_data_list = []
                        for column_name, cell_value in row.items():
                            temp_data_list.append(cell_value)
                            if column_name not in headers:
                                headers.append(column_name)
                        rows.append(temp_data_list)


                elif type == 'row':
                    df = pd.read_excel(filepath, sheet_name=sheet_name,header=None)
                    headers = df[df.columns[0]].tolist()
                    for column in df.columns[1:]:
                        rows.append(df[column].tolist())
                else:
                    return []
                
                headers = list(map(lambda x:str(x).replace("\t", "   "),headers))
                rows = [[str(item).replace("\t", "   ") for item in sublist] for sublist in rows]
                finalJsonObj = {"id":calculate_hash(headers+[str(rows)]),"header":headers,"rows":rows,"source":"表格"}
                finalJsonObjList.append(finalJsonObj)

        elif filepath.endswith(".jsonl"):
            with open(filepath, 'r',encoding='utf-8') as file:
                for line in file:
                    data = json.loads(line)
                    if "id" not in data and "table_id" in data:
                        data['id'] = data['table_id']
                    finalJsonObjList.append({**data})
                    
        elif filepath.endswith(".json"):
            with open(filepath, 'r',encoding='utf-8') as file:
                data_list = json.load(file)
            finalJsonObjList.extend(data_list)
        else:
            raise ValueError('Unsupported file format')
        
        return finalJsonObjList
    
    @component.output_types(documents=List[Dict[str, Any]])
    def run(
        self,
        path:Any,
        type:str="column"
    ):
        if isinstance(path,dict):
            if path['knowledge_elasticIndex'] != '':
                path = path['knowledge_elasticIndex']
            else:
                path = path['knowledge_path']
        elif isinstance(path,str):
            path = path
        else:
            raise ValueError("Preprecess paramter error ...")
        
        if path == "":
            raise ValueError("path is empty")
        
        finalJsonObjList = []
        
        if os.path.isdir(path):
            for fileName in os.listdir(path):
                filepath = os.path.join(path,fileName)
                temp_list = self.execute_one_file(filepath)
                finalJsonObjList.extend(temp_list)
        else:
            finalJsonObjList = self.execute_one_file(path)
            
        print("---------------- finalJsonObjList -------------",len(finalJsonObjList))
        return {"documents":finalJsonObjList}
    