import json
import importlib
from root_config import RootConfig
def retrieve_knowledge(domain, input, data_point):
    do_import = importlib.import_module('kninjllm.llm_knowledgeUploader.utils.interface_config')
    importlib.reload(do_import)
    domain_mapping = do_import.domain_mapping

    # input is a string
    knowl = {}
    # If not in mapping, automatically use "factual"
    domain = [x if x in domain_mapping else "factual" for x in domain]
    # Remove duplicates
    domain = list(dict.fromkeys(domain))
    for x in domain:
        knowl[x] = {}
        domain_sources = domain_mapping[x]
        for y in domain_sources:
            print("--- Retrieving knowledge from", x, y)
            # res = domain_sources[y](x+"/"+y).run({"question":input,"data_point":data_point})['final_result']
            res = domain_sources[y].execute(input=input)['final_result']
            input = res['input']
            query = res['processed_query']
            tmp_knowl = res['knowl']
            
            data_point["input_111"]=input
            data_point["query_111"]=query
            # print(tmp_knowl)
            knowl[x][y] = tmp_knowl

    return knowl

def knowl_is_empty(knowl):
    for x in knowl:
        for y in knowl[x]:
            if knowl[x][y] != '':
                return False
    return True