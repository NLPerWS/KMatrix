import os
from root_config import RootConfig
os.environ["CUDA_VISIBLE_DEVICES"] = RootConfig.CUDA_VISIBLE_DEVICES
import sys
import json
import importlib
import traceback
from flask import Flask, jsonify, request
from flask_cors import *
import shutil
from typing import Any, Dict, List, Optional
import gc
import ray
import torch
from kninjllm.llm_utils.common_utils import CustomEncoder
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
from kninjllm.llm_utils.component_template import get_RootClassParams_do_newNode,get_class_template_common,get_class_template_controller
from kninjllm.llm_common.log_saver import LogSaver
from kninjllm.llm_pipeline.pipeline import Pipeline

class Kninjllm_Flask:
    
    def __init__(self,flask_name,
                 pipelineRootDataDirPath:str,
                 initJsonConfigDataPath:str,
                ):
        
        # init flask_app
        self.app = Flask(flask_name)
        self.app = Flask(__name__, static_url_path='')
        self.app.config['JSON_AS_ASCII'] = False
        
        CORS(self.app, resources={r"/*": {"origins": "*"}}, send_wildcard=True)

        # init dir
        self.pipelineRootDataDir = pipelineRootDataDirPath
        if not self.pipelineRootDataDir.endswith("/"):
            self.pipelineRootDataDir = self.pipelineRootDataDir + "/"
        if not os.path.exists(self.pipelineRootDataDir):
            os.makedirs(self.pipelineRootDataDir)
        
        self.initJsonConfigDataPath = initJsonConfigDataPath
        if not os.path.exists(self.initJsonConfigDataPath):
            raise Exception(" initJsonConfigDataPath file not found")
        if not self.initJsonConfigDataPath.endswith(".json"):
            raise Exception(" initJsonConfigPath data file must be jsonfile")
        
        # init fuction
        @self.app.get('/getInitConfig')
        def getInitConfig():
            try:
                with open(self.initJsonConfigDataPath,'r',encoding='utf-8') as f:
                    data = json.load(f)
                return jsonify({"data": data, "code": 200})
            except Exception as e:
                traceback.print_exc()
                return jsonify({"data": str(e), "code": 500})
            
        @self.app.post('/uploadComponentConfig')
        def uploadComponentConfig():
            try:
                jsondata = request.get_json()
                data = jsondata["data"]
                action = jsondata["action"]
                updateNode = jsondata["updateNode"]

                # print("------------updateNode-------------")
                # print(updateNode)

                if action == 'save':
                    # common
                    if "parent_type" in updateNode and  len(updateNode['components']) ==0:
                        newUpdateNode = get_RootClassParams_do_newNode(updateNode,data)
                        for one_con in data:
                            if one_con['type'] == updateNode['parent_type']:
                                one_con['children'].append(newUpdateNode)
                        filled_template = get_class_template_common(newUpdateNode)
                    # controller
                    else:
                        newUpdateNode = get_RootClassParams_do_newNode(updateNode,data)
                        for one_con in data:
                            if one_con['type'].endswith("Controller"):
                                one_con['children'].append(newUpdateNode)
                        filled_template = get_class_template_controller(newUpdateNode)
                                
                    with open(self.initJsonConfigDataPath,'w',encoding='utf-8') as f:
                        json.dump(data,f,ensure_ascii=False)  
                    
                    if not os.path.exists(newUpdateNode['codeFilePath']):
                        os.makedirs(os.path.dirname(newUpdateNode['codeFilePath']), exist_ok=True)  
                        with open(newUpdateNode['codeFilePath'], 'w') as file:
                            file.write(filled_template) 

                    print("save !!!")
                elif action == 'del':
                    for one_con in data:
                        if one_con['type'] == updateNode['parent_type']:
                            for child in one_con['children']:
                                if child['type'] == updateNode['type']:
                                    del child
                    with open(self.initJsonConfigDataPath,'w',encoding='utf-8') as f:
                        json.dump(data,f,ensure_ascii=False)
                    if os.path.exists(updateNode['codeFilePath']):
                        os.remove(updateNode['codeFilePath'])
                    print("del !!!")
                else:
                    return jsonify({"data": "Incorrect instruction", "code": 500})
                    
                return jsonify({"data": "OK", "code": 200})
            
            except Exception as e:
                traceback.print_exc()
                return jsonify({"data": str(e), "code": 500})
            

        @self.app.get('/getAllJsonData')
        def getAllJsonData():
            try:
                jsonDataList = []
                for type_dir in os.listdir(self.pipelineRootDataDir):
                    children_list = []
                    for one_pipeline_path in os.listdir(self.pipelineRootDataDir + type_dir):
                        for dataFile in os.listdir(self.pipelineRootDataDir + type_dir + "/" + one_pipeline_path):
                            if not dataFile.startswith("result_") and not dataFile.startswith("variable_") and not dataFile.startswith("logging"):
                                with open(self.pipelineRootDataDir + type_dir + "/" + one_pipeline_path + "/" + dataFile, "r",encoding='utf-8') as f:
                                    thisOriginJsonData = json.load(f)
                                children_list.append({
                                    "value":dataFile,
                                    "label":dataFile,
                                    "fileName": dataFile,
                                    "dir_filePath": self.pipelineRootDataDir + type_dir + "/" + one_pipeline_path + "/",
                                    "data": thisOriginJsonData
                                })
                    jsonDataList.append({
                        "value":type_dir,
                        "label":type_dir,
                        "children":children_list
                    })
                        
                return jsonify({"data": jsonDataList, "code": 200})
            except Exception as e:
                traceback.print_exc()
                return jsonify({"data": str(e), "code": 500})
            
        @self.app.post('/getComponentCodeByFilePath')
        def getComponentCodeByFilePath():
            try:
                jsondata = request.get_json()
                codeFilePath = jsondata["codeFilePath"]
                with open(codeFilePath,'r',encoding='utf-8') as f:
                    codeData = f.read()
                return jsonify({"data": codeData, "code": 200})
            except Exception as e:
                traceback.print_exc()
                return jsonify({"data": str(e), "code": 500})
            
            
        @self.app.post('/setComponentCodeByFilePath')
        def setComponentCodeByFilePath():
            try:
                jsondata = request.get_json()
                code = jsondata["code"]
                codeFilePath = jsondata["codeFilePath"]
                with open(codeFilePath,'w',encoding='utf-8') as f:
                    f.write(code)
                return jsonify({"data": "OK", "code": 200})
            except Exception as e:
                traceback.print_exc()
                return jsonify({"data": str(e), "code": 500})
            

        @self.app.post('/startRunPipelineByFileName')
        def startRunPipelineByFileName():
            try:
                jsondata = request.get_json()
                data = jsondata["data"]
                pipelineType = data['pipeLineType']
                print(pipelineType)
                pipeLineName = jsondata["pipeLineName"]
                execute = jsondata["execute"]
                thisDirPath = self.pipelineRootDataDir + pipelineType + "/" +  pipeLineName + "/"
                
                if not os.path.exists(thisDirPath):
                    os.makedirs(thisDirPath)
                with open(thisDirPath + pipeLineName +".json", "w",encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False)
                    
                if execute == "save":
                    return jsonify({"data": "save ok", "code": 200})
                    
                elif execute == "run" or execute == "Deployment" or execute == "dev":
                    thisResultFilePath = thisDirPath + "result_" + pipeLineName + ".json"
                    thisLogFilePath =  thisDirPath + "logging_" + pipeLineName + ".txt"
                    thisVariableDataPath = thisDirPath + "variable_" + pipeLineName + ".jsonl"
                    
                    with open(thisResultFilePath,'w',encoding='utf-8') as f:
                        f.write("[]")
                    with open(thisLogFilePath,'w',encoding='utf-8') as f:
                        f.write("")
                    with open(thisVariableDataPath,'w',encoding='utf-8') as f:
                        f.write("")
                    
                    resultData = self.changeJsonDataToPipeline(pipeLineName,data,thisVariableDataPath,thisLogFilePath,execute)

                    with open(thisResultFilePath, 'w', encoding='utf-8') as f:
                        json.dump(resultData, f, cls=CustomEncoder, ensure_ascii=False)
                        
                    return jsonify({"data": resultData, "code": 200})
                else:
                    return jsonify({"data": "Incorrect instruction", "code": 500})
                
            except Exception as e:
                traceback.print_exc()
                return jsonify({"data": str(e), "code": 500})
            
                
        @self.app.post('/delPipelineByFileName')
        def delPipelineByFileName():
            try:
                jsondata = request.get_json()
                pipeLineName = jsondata["pipeLineName"]
                pipeLineType = jsondata["pipeLineType"]
                thisDirPath = self.pipelineRootDataDir + pipeLineType +"/"+ pipeLineName + "/"
                print("thisDirPath \n",thisDirPath)
                
                if os.path.exists(thisDirPath):
                    shutil.rmtree(thisDirPath)
                    return jsonify({"data": "del ok", "code": 200})   
                else:
                    return jsonify({"data": "file does not exist", "code": 500})   
            except Exception as e:
                traceback.print_exc()
                return jsonify({"data": str(e), "code": 500})

                
        @self.app.post('/getPipelineResultData')
        def getPipelineResultData():
            try:
                jsondata = request.get_json()
                pipeLineName = jsondata["pipeLineName"]
                pipeLineType = jsondata["pipeLineType"]
                thisDirPath = self.pipelineRootDataDir + pipeLineType + "/" + pipeLineName + "/"
                thisResultFilePath = thisDirPath + "result_" + pipeLineName + ".json"
                if os.path.exists(thisDirPath) and os.path.exists(thisResultFilePath):
                    with open(thisResultFilePath, "r",encoding='utf-8') as f:
                        resultData = json.load(f)
                    return jsonify({"data": resultData, "code": 200})   
                else:
                    return jsonify({"data": "pipeline has not been executed or has not been executed yet, please wait...", "code": 500})   
            except Exception as e:
                traceback.print_exc()
                return jsonify({"data": str(e), "code": 500})
            
            
        @self.app.post('/getPipelineTempVariableData')
        def getPipelineTempVariableData():
            try:
                jsondata = request.get_json()
                pipeLineName = jsondata["pipeLineName"]
                pipeLineType = jsondata["pipeLineType"]
                thisDirPath = self.pipelineRootDataDir + pipeLineType + "/" + pipeLineName + "/"
                thisResultFilePath = thisDirPath + "variable_" + pipeLineName + ".jsonl"
                if os.path.exists(thisDirPath) and os.path.exists(thisResultFilePath):
                    json_list = []
                    with open(thisResultFilePath, "r",encoding='utf-8') as f:
                        for line in f:
                            json_list.append(json.loads(line))
                        
                    return jsonify({"data": json_list, "code": 200})   
                else:
                    return jsonify({"data": "pipeline has not yet been implemented...", "code": 500})   
            except Exception as e:
                traceback.print_exc()
                return jsonify({"data": str(e), "code": 500})
        

        @self.app.get('/initCatch')
        def initCatch():
            del RootConfig.tempPipeLineKnowledgeCatch
            del RootConfig.tempModelCatch
            RootConfig.tempPipeLineKnowledgeCatch = []
            RootConfig.tempModelCatch = []
            destroy_model_parallel()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            ray.shutdown()
            return jsonify({"data": "ok", "code": 200})   
            
        @self.app.route('/uploadKnowledge', methods=['POST'])
        def uploadKnowledge():
            savePath = request.form.get("savePath")
            if savePath == "":
                raise ValueError("Upload path is empty")
            if not savePath.endswith("/"):
                savePath = savePath + "/"
            if 'files' not in request.files:
                return jsonify({"data": 'no file is selected', "code": 500})
            files = request.files.getlist('files')
            if len(files) == 0:
                return jsonify({"data": 'no file is selected', "code": 500})
            
            if not os.path.exists(savePath):
                os.makedirs(savePath)
            
            if os.path.exists(savePath):
                for filename in os.listdir(savePath):
                    file_path = os.path.join(savePath, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path) 
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f'Failed to delete {file_path}. Reason: {e}')
            
            for file in files:
                filename = file.filename 
                file.save(savePath+filename)  
                
            return jsonify({"data": "upload success", "code": 200})
            
        
    def changeJsonDataToPipeline(self,pipelineName,jsonData,variableDataPath,thisLogFilePath,execute):
        if thisLogFilePath != "":
            RootConfig.logSaver = LogSaver(logpath = thisLogFilePath)
            RootConfig.logSaver.initLog()
        else:
            RootConfig.logSaver = None
            
        pipeLine = Pipeline()
        
        nodeList = jsonData['nodeList']
        lineList = jsonData['lineList']
        multiplexer_data = ""
        multiplexerName = "Multiplexer"
        for node in nodeList:
            # ----------------------------------------------------------- Multiplexer -----------------------------------------------------------------------------
            if node['type'] == "Multiplexer":
                from kninjllm.llm_common.mutiplexer import Multiplexer
                multiplexer_data = list(filter(lambda x: x["name"]=="value",node['inputParams']))[0]['value']
                multiplexerName = node['name']
                pipeLine.add_component(instance=Multiplexer(),name=node['name'])
            
            # ----------------------------------------------------------- builder -----------------------------------------------------------------------------
            # Builder -> PromptBuilder
            elif node['type'] == "PromptBuilder":
                from kninjllm.llm_builder.prompt_builder import PromptBuilder
                template = list(filter(lambda x: x["name"]=="template",node['initParams']))[0]['value']
                if template == "":
                    raise ValueError("PromptBuilder Initialization parameters are missing, please check ...")
                pipeLine.add_component(instance=PromptBuilder(template=template),name=node['name'])
                
            elif node['type'] == "OutputBuilder":
                from kninjllm.llm_builder.output_builder import OutputBuilder
                pipeLine.add_component(instance=OutputBuilder(),name=node['name'])
                
            # ----------------------------------------------------------- Generator -----------------------------------------------------------------------------
            # LLM -> openai
            elif node['type'] == "OpenAIGenerator":
                from kninjllm.llm_generator.close_generator.openai_generator import OpenAIGenerator
                api_key = list(filter(lambda x: x["name"]=="api_key",node['initParams']))[0]['value']
                if api_key == "":
                    api_key = RootConfig.openai_api_key
                executeType = list(filter(lambda x: x["name"]=="executeType",node['initParams']))[0]['value']
                if executeType == "":
                    raise ValueError("OpenAIGenerator Initialization parameters are missing, please check ...")
                pipeLine.add_component(instance=OpenAIGenerator(api_key=api_key,executeType=executeType),name=node['name'])
            
            # LLM -> selfRag
            elif node['type'] == "RagGenerator":
                from kninjllm.llm_generator.base_generator.self_rag.self_rag_generator import RagGenerator
                model_path = list(filter(lambda x: x["name"]=="model_path",node['initParams']))[0]['value']
                executeType = list(filter(lambda x: x["name"]=="executeType",node['initParams']))[0]['value']
                if model_path == "":
                    model_path = RootConfig.selfRAG_model_path
                if executeType == "":
                    raise ValueError("RagGenerator Initialization parameters are missing, please check ...")
                pipeLine.add_component(instance=RagGenerator(model_path=model_path,executeType=executeType),name=node['name'])
            # LLM -> LLAMA2
            elif node['type'] == "LLama2Generator":
                from kninjllm.llm_generator.base_generator.llama2.component_generator_llama2 import LLama2Generator
                model_path = list(filter(lambda x: x["name"]=="model_path",node['initParams']))[0]['value']
                executeType = list(filter(lambda x: x["name"]=="executeType",node['initParams']))[0]['value']
                if model_path == "":
                    model_path = RootConfig.llama2_model_path
                if executeType == "":
                    raise ValueError("RagGenerator Initialization parameters are missing, please check ...")
                pipeLine.add_component(instance=LLama2Generator(model_path=model_path,executeType=executeType),name=node['name'])
                
            elif node['type'] == "Baichuan2Generator":
                from kninjllm.llm_generator.base_generator.baichuan2.component_generator_baichuan2 import Baichuan2Generator
                model_path = list(filter(lambda x: x["name"]=="model_path",node['initParams']))[0]['value']
                executeType = list(filter(lambda x: x["name"]=="executeType",node['initParams']))[0]['value']
                if model_path == "":
                    model_path = RootConfig.baichuan2_model_path
                if executeType == "":
                    raise ValueError("RagGenerator Initialization parameters are missing, please check ...")
                pipeLine.add_component(instance=Baichuan2Generator(model_path=model_path,executeType=executeType),name=node['name'])
                
           # ------------------------------------------------------query generator--------------------------------------------------------------------
            elif node["type"] == "SparqlGenerator":
                from kninjllm.llm_queryGenerator.sparql_language_generator import Sparql_language_generator
                pipeLine.add_component(instance=Sparql_language_generator(),name=node['name'])
            elif node["type"] == "NaturalGenerator":
                from kninjllm.llm_queryGenerator.natural_language_generator import Natural_language_generator
                pipeLine.add_component(instance=Natural_language_generator(),name=node['name'])
        
            # ----------------------------------------------------------- Retriever -----------------------------------------------------------------------------
            # Retriever -> Bm25MemoryRetriever 
            elif node['type'] == "Bm25MemoryRetriever":
                from kninjllm.llm_retriever.in_memory import InMemoryBM25Retriever
                top_k = list(filter(lambda x: x["name"]=="top_k",node['initParams']))[0]['value']
                executeType = list(filter(lambda x: x["name"]=="executeType",node['initParams']))[0]['value']
                if top_k == "" or executeType == "":
                    raise ValueError("Bm25MemoryRetriever Initialization parameters are missing, please check ...")
                pipeLine.add_component(instance=InMemoryBM25Retriever(top_k=int(top_k),executeType=executeType),name=node['name'])
                
            elif node['type'] == "Bm25EsRetriever":
                from kninjllm.llm_retriever.elasticsearch import ElasticsearchBM25Retriever
                top_k = list(filter(lambda x: x["name"]=="top_k",node['initParams']))[0]['value']
                executeType = list(filter(lambda x: x["name"]=="executeType",node['initParams']))[0]['value']
                if top_k == "" or executeType == "":
                    raise ValueError("ElasticsearchBM25Retriever Initialization parameters are missing, please check ...")
                pipeLine.add_component(instance=ElasticsearchBM25Retriever(top_k=int(top_k),executeType=executeType),name=node['name'])
                
            elif node['type'] == "ContrieverRetriever":
                from kninjllm.llm_retriever.contriever.Contriever_retriever import Contriever_Retriever
                model_path = list(filter(lambda x: x["name"]=="model_path",node['initParams']))[0]['value']
                if model_path == "":
                    model_path = RootConfig.contriever_model_path
                top_k = list(filter(lambda x: x["name"]=="top_k",node['initParams']))[0]['value']
                executeType = list(filter(lambda x: x["name"]=="executeType",node['initParams']))[0]['value']
                if top_k == "" or executeType == "":
                    raise ValueError("ContrieverRetriever Initialization parameters are missing, please check ...")
                pipeLine.add_component(instance=Contriever_Retriever(model_path=model_path,executeType=executeType,top_k=int(top_k)),name=node['name'])
            
            elif node['type'] == "DPR_retriever":
                from kninjllm.llm_retriever.DPR.DPR_retriever import DPR_Retriever
                model_path = list(filter(lambda x: x["name"]=="model_path",node['initParams']))[0]['value']
                if model_path == "":
                    model_path = RootConfig.DPR_model_path
                top_k = list(filter(lambda x: x["name"]=="top_k",node['initParams']))[0]['value']
                executeType = list(filter(lambda x: x["name"]=="executeType",node['initParams']))[0]['value']
                if top_k == "" or executeType == "":
                    raise ValueError("DPR_retriever Initialization parameters are missing, please check ...")
                pipeLine.add_component(instance=DPR_Retriever(model_path=model_path,executeType=executeType,top_k=int(top_k)),name=node['name'])
            
            elif node['type'] == "BGE_retriever":
                from kninjllm.llm_retriever.BGE.BGE_retriever import BGE_Retriever
                model_path = list(filter(lambda x: x["name"]=="model_path",node['initParams']))[0]['value']
                if model_path == "":
                    model_path = RootConfig.BGE_model_path
                top_k = list(filter(lambda x: x["name"]=="top_k",node['initParams']))[0]['value']
                executeType = list(filter(lambda x: x["name"]=="executeType",node['initParams']))[0]['value']
                if top_k == "" or executeType == "":
                    raise ValueError("BGE_retriever Initialization parameters are missing, please check ...")
                pipeLine.add_component(instance=BGE_Retriever(model_path=model_path,executeType=executeType,top_k=int(top_k)),name=node['name'])
                
            elif node['type'] == "E5_retriever":
                from kninjllm.llm_retriever.E5.E5_retriever import E5_Retriever
                model_path = list(filter(lambda x: x["name"]=="model_path",node['initParams']))[0]['value']
                if model_path == "":
                    model_path = RootConfig.E5_model_path
                top_k = list(filter(lambda x: x["name"]=="top_k",node['initParams']))[0]['value']
                executeType = list(filter(lambda x: x["name"]=="executeType",node['initParams']))[0]['value']
                if top_k == "" or executeType == "":
                    raise ValueError("E5_retriever Initialization parameters are missing, please check ...")
                pipeLine.add_component(instance=E5_Retriever(model_path=model_path,executeType=executeType,top_k=int(top_k)),name=node['name'])
                
            elif node['type'] == "BERT_retriever":
                from kninjllm.llm_retriever.BERT.BERT_retriever import BERT_Retriever
                model_path = list(filter(lambda x: x["name"]=="model_path",node['initParams']))[0]['value']
                if model_path == "":
                    model_path = RootConfig.BERT_model_path
                top_k = list(filter(lambda x: x["name"]=="top_k",node['initParams']))[0]['value']
                executeType = list(filter(lambda x: x["name"]=="executeType",node['initParams']))[0]['value']
                if top_k == "" or executeType == "":
                    raise ValueError("BERT_retriever Initialization parameters are missing, please check ...")
                pipeLine.add_component(instance=BERT_Retriever(model_path=model_path,executeType=executeType,top_k=int(top_k)),name=node['name'])
                
            # ----------------------------------------------------------- Router -----------------------------------------------------------------------------
            # Router -> ConditionalRouter 
            elif node['type'].endswith("Router"):
                from kninjllm.llm_router.conditional_router import ConditionalRouter
                routes_str = list(filter(lambda x: x["name"]=="routes",node['initParams']))[0]['value']
                max_loop_count = list(filter(lambda x: x["name"]=="max_loop_count",node['initParams']))[0]['value']
                if routes_str == "":
                    raise ValueError("Router Initialization parameters are missing, please check ...")
                pipeLine.add_component(instance=ConditionalRouter(max_loop_count=int(max_loop_count),routes=eval(routes_str)),name=node['name'])
                
            # --------------------------------------------------------- dataSetLoader -----------------------------------------------------------------------
            elif node['type'].startswith("DataSet"):
                from kninjllm.llm_dataloader.dataSetLoader import DataSetLoader
                dataset_path = list(filter(lambda x: x["name"]=="dataset_path",node['initParams']))[0]['value']
                if dataset_path == "":
                    raise ValueError("DataSet Initialization parameters are missing, please check ...")
                pipeLine.add_component(instance=DataSetLoader(dataset_path=dataset_path),name=node['name'])
            
            # --------------------------------------------------------- KnowledgeLoader -----------------------------------------------------------------------
            elif node['type'].startswith("Knowledge_") or node['type'].startswith("KnowledgeCombiner"):
                from kninjllm.llm_dataloader.knowledgeLoader import KnowledgeLoader
                tag = list(filter(lambda x: x["name"]=="tag",node['initParams']))[0]['value']
                knowledge_path = list(filter(lambda x: x["name"]=="knowledge_path",node['initParams']))[0]['value']
                elasticIndex = list(filter(lambda x: x["name"]=="elasticIndex",node['initParams']))[0]['value']
                if knowledge_path == "" and elasticIndex == "":   
                    raise ValueError(f"Please set up a knowledge source folder or database link: {node['name']}")
                pipeLine.add_component(instance=KnowledgeLoader(knowledge_path=knowledge_path,knowledge_elasticIndex=elasticIndex,knowledge_tag=tag),name=node['name'])
                
            # --------------------------------------------------------- KnowledgeSelector -----------------------------------------------------------------------
            elif node['type'] == "KnowledgeSelector":
                from kninjllm.llm_dataloader.KnowledgeSelector import KnowledgeSelector
                pipeLine.add_component(instance=KnowledgeSelector(),name=node['name'])

            # --------------------------------------------------------- knowledge Upload -----------------------------------------------------------------------
            elif node['type'] == "Upload_Text" or node['type'] == "Upload_Table" or node['type'] == "Upload_KG":
                from kninjllm.llm_knowledgeUploader.KnowledgeUploader import KnowledgeUploader
                savePath = list(filter(lambda x: x["name"]=="savePath",node['initParams']))[0]['value']
                if savePath == "":
                    raise ValueError("Uploader Initialization parameters are missing, please check ...")
                if not savePath.endswith("/"):
                    savePath = savePath + "/"
                pipeLine.add_component(instance=KnowledgeUploader(path=savePath),name=node['name'])
            # --------------------------------------------------------- Interface -----------------------------------------------------------------------
            elif node['type'] == "Upload_interface":
                from kninjllm.llm_knowledgeUploader.Local_interface import Local_interface
                interface_domain = list(filter(lambda x: x["name"]=="interface_domain",node['initParams']))[0]['value']
                if len(interface_domain.split('/')) != 2:
                    raise ValueError("interface_domain Format error, please check ...")
                interface_type = list(filter(lambda x: x["name"]=="interface_type",node['initParams']))[0]['value']
                if interface_type not in ["wiki","google","local"]:
                    raise ValueError("Unsupported interface_type, please check ...")
                search_url = list(filter(lambda x: x["name"]=="search_url",node['initParams']))[0]['value']
                if interface_domain == "" or interface_type == "" or search_url == "" :
                    raise ValueError("Local_interface Initialization parameters are missing, please check ...")
                pipeLine.add_component(instance=Local_interface(interface_domain=interface_domain,interface_type=interface_type,search_url=search_url),name=node['name'])
                
            # --------------------------------------------------------- Preprecess -----------------------------------------------------------------------

            elif node['type'] == "InterfacePreprecess":
                from kninjllm.llm_preprocess.InterfacePreprecess import InterfacePreprecess
                pipeLine.add_component(instance=InterfacePreprecess(),name=node['name'])

            elif node['type'] == "TablePreprecess":
                from kninjllm.llm_preprocess.TablePreprecess import TablePreprecess
                pipeLine.add_component(instance=TablePreprecess(),name=node['name'])
                
            elif node['type'] == "KgPreprecess":
                from kninjllm.llm_preprocess.KgPreprecess import KGPreprecess
                pipeLine.add_component(instance=KGPreprecess(),name=node['name'])
            
            elif node['type'] == "TextPreprecess":
                from kninjllm.llm_preprocess.TextPreprecess import TextPreprecess
                pipeLine.add_component(instance=TextPreprecess(),name=node['name'])


            # -------------------------------------------------------------Unified------------------------------------------------------------------------------
            elif node['type'] == "UnifiedVerbalizer":
                from kninjllm.llm_linearizer.LinearizerToText import LinearizerToText
                max_length = list(filter(lambda x: x["name"]=="max_length",node['initParams']))[0]['value']
                max_length = int(max_length)
                knowledge_line_count = list(filter(lambda x: x["name"]=="knowledge_line_count",node['initParams']))[0]['value']
                if knowledge_line_count == "":
                    raise ValueError("UnifiedVerbalizer Initialization parameters are missing, please check ...")
                knowledge_line_count = int(knowledge_line_count)
                pipeLine.add_component(instance=LinearizerToText(knowledge_line_count=knowledge_line_count,max_length=max_length,valueList=[],count=0),name=node['name'])

            elif node['type'] == "UnifiedQuerier":
                from kninjllm.llm_linearizer.UnifiedInterface import UnifiedInterface
                knowledge_line_count = list(filter(lambda x: x["name"]=="knowledge_line_count",node['initParams']))[0]['value']
                if knowledge_line_count == "":
                    raise ValueError("UnifiedQuerier Initialization parameters are missing, please check ...")
                knowledge_line_count = int(knowledge_line_count)
                pipeLine.add_component(instance=UnifiedInterface(knowledge_line_count=knowledge_line_count,max_length=0,valueList=[],count=0),name=node['name'])


            # --------------------------------------------------------- saveToDataBase -----------------------------------------------------------------------
            elif node['type'] == "SaveToElasticSearchDB":
                from kninjllm.llm_store_saver.SaveToElasticSearchDB import SaveToElasticSearchDB
                host = RootConfig.ES_HOST
                index = list(filter(lambda x: x["name"]=="index",node['initParams']))[0]['value']
                username = RootConfig.ES_USERNAME
                password = RootConfig.ES_PASSWORD
                ebbedding_retriever_nameList = list(filter(lambda x: x["name"]=="ebbedding_retriever_nameList",node['initParams']))[0]['value']
                ebbedding_retriever_nameList = json.loads(ebbedding_retriever_nameList)
                if index == "":
                    raise ValueError("SaveToElasticSearchDB Initialization parameters are missing, please check ...")
                pipeLine.add_component(instance=SaveToElasticSearchDB(host=host,index=index,username=username,password=password,ebbedding_retriever_nameList=ebbedding_retriever_nameList),name=node['name'])
                
            elif node['type'] == "SaveToServer":
                from kninjllm.llm_store_saver.SaveToServer import SaveToServer
                savePath = list(filter(lambda x: x["name"]=="savePath",node['initParams']))[0]['value']
                if savePath == "":
                    raise Exception("savePath is empty")
                pipeLine.add_component(instance=SaveToServer(savePath=savePath),name=node['name'])
                
            elif node['type'] == "SaveQueryInterface":
                from kninjllm.llm_store_saver.SaveQueryInterface import SaveQueryInterface
                pipeLine.add_component(instance=SaveQueryInterface(),name=node['name'])
            
            # --------------------------------------------------------- evaluation -----------------------------------------------------------------------
            elif node["type"] == "Evaluator":
                from kninjllm.llm_evaluation.evaluator import Evaluator
                pipeLine.add_component(instance=Evaluator(),name=node['name'])
                 
            # --------------------------------------------- controller -----------------------------------------------------------------------------
            # LLM controller 
            elif node['type'] == "LongRagController":
                from kninjllm.llm_controller.control_self_rag_long_demo import SelfRagLongDemoController
                pipeLine.add_component(instance=SelfRagLongDemoController(variableDataPath = variableDataPath),name=node['name'])
                
            elif node['type'] == "ShortRagController":
                from kninjllm.llm_controller.control_self_rag_short_demo import SelfRagShortDemoController
                pipeLine.add_component(instance=SelfRagShortDemoController(variableDataPath = variableDataPath),name=node['name'])
                
            elif node['type'] == "CokController":
                from kninjllm.llm_controller.control_cok import CokController
                pipeLine.add_component(instance=CokController(),name=node['name'])
            
            else:
                raise Exception(f"This node is not registered and cannot be joined to pipeline. Please check the node:  {node['type']}")

        for link in lineList:
            fromNode = list(filter(lambda x:x['id']==link['from'],nodeList))[0]
            toNode = list(filter(lambda x:x['id']==link['to'],nodeList))[0]
            fromNodeOutputParam = link['label_from']
            toNodeInputParam = link['label_to']
            pipeLine.connect(f"{fromNode['name']}.{fromNodeOutputParam}",f"{toNode['name']}.{toNodeInputParam}")
        
        if pipeLine.get_component(multiplexerName) != None:
            result = pipeLine.run({
                multiplexerName: {"value":multiplexer_data},
            })
        else:
            result = pipeLine.run({})

        return result
        
    # run 
    def run(self, host, port):
        self.app.run(host=host, port=port)

if __name__ == "__main__":
    
    my_flask_app = Kninjllm_Flask(flask_name='my_flask_app',
                                  pipelineRootDataDirPath=RootConfig.root_path + "dir_pipeline_data",
                                  initJsonConfigDataPath=RootConfig.root_path + "dir_init_config/init_config_data.json")
    my_flask_app.run(host='0.0.0.0', port=int(RootConfig.SERVER_PORT))
    