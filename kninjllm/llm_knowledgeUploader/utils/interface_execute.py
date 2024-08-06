import traceback
from root_config import RootConfig
from kninjllm.llm_utils.common_utils import set_proxy,unset_proxy
from kninjllm.llm_queryGenerator.natural_language_generator import Natural_language_generator
import time
import json
import requests
import os
import time
import traceback
from SPARQLWrapper import SPARQLExceptions,SPARQLWrapper, JSON
from kninjllm.llm_common.component import component
from kninjllm.llm_generator.base_generator.wikidata_emnlp23.utils import get_query_sparql,proxy_query_wiki
from kninjllm.llm_queryGenerator.sparql_language_generator import Sparql_language_generator
from kninjllm.llm_retriever.contriever.Contriever_retriever import Contriever_Retriever

class InterfaceExecute:
    def __init__(self,domain, type, url):
        self.domain = domain
        self.type = type
        self.url = url

        # factual/wikipedia     @wikipedia
        # factual/wikidata      https://query.wikidata.org/sparql
        # medical/uptodate      @uptodate
        # medical/flashcard     veggiebird/medical-flashcards
        # physical/physicsclassroom     @physicsclassroom
        # physical/scienceqa_phy        veggiebird/physics-scienceqa
        # biology/ck12          @ck12
        # biology/scienceqa_bio         veggiebird/biology-scienceqa

        if type == "local":
            if not self.url.startswith("/"):
                read_path = os.path.join(RootConfig.root_path,self.url)
            else:
                read_path = self.url
            if os.path.isfile(read_path) and read_path.endswith(".jsonl"):
                try:
                    with open(read_path,'r',encoding='utf-8') as f:
                        self.data_list = [json.loads(line) for line in f.readlines()]
                except:
                    raise ValueError(f"this file not exist {read_path}")
            else:
                self.data_list = []

            self.contriever = Contriever_Retriever(model_path=RootConfig.contriever_model_path,executeType="infer",top_k=1)
        else:
            self.contriever = None
            
        
      
    def execute_by_google(self,input):

        def execute_google_query(query,search_char):
            if search_char == "wikipedia":
                search_char = "https://en.wikipedia.org/wiki"
            
            unset_proxy()
            time.sleep(1)
            url = "https://google.serper.dev/search"
            payload = json.dumps({
            "q": query
            })
            headers = {
            'X-API-KEY': RootConfig.google_search_key,
            'Content-Type': 'application/json'
            }
            response = requests.request("POST", url, headers=headers, data=payload)
            # if response.COD
            if response.status_code==200:
                results=response.json()
            else:
                raise Exception("google key error!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            knowl = ""
            if "answer_box" in results:
                if "snippet" in results["answer_box"]:
                    knowl += results["answer_box"]["snippet"]
                    knowl += " "
            # organic answers
            if "organic" in results:
                organic_result=results["organic"]
                organic_result_sort=[]
                for single_organic_result in organic_result:
                    if search_char in single_organic_result["link"]:
                        organic_result_sort.append(single_organic_result)
                # yield maximun 3 snippets
                if len(knowl) == 0:
                    # if no answer box, yield maximun 3 snippets
                    num_snippets = min(1, len(organic_result_sort))
                else:
                    num_snippets = min(0, len(organic_result_sort))
                
                organic_result_sort = sorted(organic_result_sort,key = lambda i:len(i),reverse=True)
                    
                for i in range(num_snippets):
                    if "snippet" in organic_result_sort[i]:
                        knowl += organic_result_sort[i]["snippet"]
                        knowl += "\n"
            return knowl

        # this url is  @ck12 or @uptodate
        print("Generate query...")
        # 注释：查询query生成器
        query = Natural_language_generator().run(query_obj={"question":input})['final_result']
        # 注释：查询query生成器
        # print(query)
        print("Retrieve  knowledge...")
        # 注释：查询接口
        knowl = execute_google_query(query+" "+self.url,self.url.replace("@",""))
        # 注释：查询接口
        # print(knowl)
        return input,input+" "+self.url,knowl


    def execute_by_wiki(self,input):

        def query_wiki(query):
            unset_proxy()
            while True:
                try:
                    # # 直接访问
                    sparql = SPARQLWrapper(self.url)
                    sparql.setQuery(query)
                    sparql.setTimeout(10)
                    sparql.setReturnFormat(JSON)
                    results = sparql.query().convert()
                    item_labels = []
                    if "results" in results:
                        for result in results["results"]["bindings"][:10]:
                            item_labels.append(result)
                    return item_labels
                except:
                    traceback.print_exc()
                    try:
                        res = proxy_query_wiki(query)
                        return res
                    except:
                        traceback.print_exc()
                        time.sleep(65)
                        print("query wiki error sleep 65 minutes")
                        # return []

        def get_entity_name(entity_id):
            
            url = f"https://www.wikidata.org/w/api.php?action=wbgetentities&format=json&ids={entity_id}"
            # response = requests.get(url)
            response = get_query_sparql(url=url)
            
            data = response.json()

            # Extract the entity name
            entity = data["entities"][entity_id]
            # print("entity = data[entities][entity_id]:",entity)
            try:
                entity_name = entity["labels"]["en"]["value"]  # Assuming you want the English name
            except:
                entity_name=""
                print("entity has no label")

            return entity_name

        def get_wiki_info(list_of_info):
            
            info_list = []
            # print("get_wiki_info(list_of_info):",list_of_info)
            for i in range(len(list_of_info)):
                tmp_info = list_of_info[i]
                
                if len(tmp_info) == 1:

                    tmp_value=tmp_info["x"]['value']
                    
                    if 'http://www.wikidata.org/' in tmp_value:
                        info_list.append(get_entity_name(tmp_value.split('/')[-1]))
                    else:
                        info_list.append(tmp_value)
                else:
                    # ans_1 ans_2
                    try:
                        tmp_info=list(tmp_info.values())
                        # print("tmp_info=list(tmp_info.values()):",tmp_info)
                        
                        info_list.append(get_entity_name(tmp_info[0]['value'].split('/')[-1]))
                        info_list.append(get_entity_name(tmp_info[1]['value'].split('/')[-1]))
                    except:
                        print("tmp_info is []")
                        pass
                    
            
            # convert list to string
            opt = ''
            for i in info_list:
                opt += i
                opt += ', '
                
            return opt[:-2]+'.'

        def execute_wiki_query(query, processed_query):
            knowl = ""
            info = query_wiki(processed_query)
                
            if len(info) != 0:
                tmp_answer = get_wiki_info(info)
                knowl += processed_query.strip()
                knowl += " Answer: "
                knowl += tmp_answer.strip()
            return knowl


        # endpoint_url = "https://query.wikidata.org/sparql"
        print("Generate query...")
        query = input
        processed_query = Sparql_language_generator().run(query_obj={"question":query})['final_result']
        # 注释：查询query生成器
        print("Generate query result:",processed_query)
        print("Retrieve knowledge...")
        time.sleep(1)
        knowl = execute_wiki_query(query, processed_query)
        return input,processed_query,knowl
        
    def execute_by_local(self,input):
        self.contriever.searchDataList = self.data_list
        res = self.contriever.run(query_obj = {"question":input})['final_result'][0][0]
        knowl = res['content'].split("\t")[1]
        # print(res_str)
        return input,input,knowl
                
    def execute(self,input):
        if self.type == "google":
            input,processed_query,knowl = self.execute_by_google(input)
        elif self.type == "wiki":
            input,processed_query,knowl = self.execute_by_wiki(input)
        elif self.type == "local":
            input,processed_query,knowl = self.execute_by_local(input)
        else:
            input = input
            processed_query = input
            knowl = ""
        print("interface retriever know :\n",knowl)
        
        return {"final_result":{"input":input,"processed_query":processed_query,"knowl":knowl}}