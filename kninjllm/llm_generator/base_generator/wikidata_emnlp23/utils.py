import requests
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
from kninjllm.llm_common.version import __version__ as kninjllm_version
from vllm import SamplingParams
from kninjllm.llm_utils.common_utils import unset_proxy,loadModelByCatch
from root_config import RootConfig


def get_es_clent():
    client = Elasticsearch(
        RootConfig.ES_HOST,
        headers={"user-agent": f"kninjllm-py-ds/{kninjllm_version}"},
        basic_auth=(RootConfig.ES_USERNAME,RootConfig.ES_PASSWORD)
    )
    return client

def insert_one_to_es(client,index,data):
    if not client.indices.exists(index=index):
        client.indices.create(index=index)
    client.index(index=index, body=data,refresh=True)
    return "ok"
    
def delete_many_by_es(client,index,data):
    if not client.indices.exists(index=index):
        client.indices.create(index=index)
    query = {
        "query": {
            "match": data
        }
    }
    
    docs = scan(client, query=query, index=index)
    
    for doc in docs:
        client.delete(index=index, id=doc['_id'],refresh=True)
        
def find_one_by_es(client,index,data):
    if not client.indices.exists(index=index):
        client.indices.create(index=index)
    
    query = {
        "query": {
            "match": data
        }
    }
    res = client.search(index=index,body=query)
    if len(res['hits']['hits']) == 0:
        return None
    return res['hits']['hits'][0]['_source']
    

def find_by_es(client,index):
    if not client.indices.exists(index=index):
        client.indices.create(index=index)
    
    """
     Scroll API 
    """
    res_list = []
    
    
    scroll = '5m'  
    query = {"query": {"match_all": {}}}  

    
    response = client.search(
        index=index,
        scroll=scroll,
        size=10000,  
        body=query
    )

    
    while True:
        if len(response['hits']['hits']) == 0:
            break
        res_list.extend(hit['_source'] for hit in response["hits"]["hits"])
        response = client.scroll(scroll_id=response['_scroll_id'], scroll=scroll)

    return res_list



def get_query_sparql(url,headers=None,params=None):
    unset_proxy()
    try:
        res = requests.get(url=url, params=params,headers=headers,timeout=5)
    except:
        try:
            
            
            res = requests.post(url="http://ave0lv6oah9g.guyubao.com/do_request_sparql",headers = {'Content-Type': 'application/json'},json={
                "url":url,
                "headers":headers,
                "params":params
            })
        except:
            res = None
        
    return res


def proxy_query_wiki(query):
    unset_proxy()
    res = requests.post(url="http://ave0lv6oah9g.guyubao.com/do_query_wiki",headers = {'Content-Type': 'application/json'},json={'query':query}).json()['data']
    return res
    

def natural_to_sparql(prompt):
    sampling_params = SamplingParams(temperature=0, top_p=1,max_tokens=200,stop=["</s>", "\n"])
    model = loadModelByCatch(model_name='wikisp',model_path=RootConfig.WikiSP_model_path)
    pred = model.generate(prompt, sampling_params)[0]
    content = pred.outputs[0].text
    return content
