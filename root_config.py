import os


class RootConfig:
    # global variables
    logSaver = None
    tempModelCatch = []
    tempPipeLineKnowledgeCatch = []
    
    # ---------------------------------- The following configurations need to be modified -----------------------------------
    # Absolute path of the project. If it is started by docker, keep /app/KMatrix
    root_path = "/app/KMatrix"
    if not root_path.endswith("/"):
        root_path = root_path + "/"
    
    # Server PORT
    SERVER_PORT = "8002"
    
    # ES connection address
    ES_HOST = "http://127.0.0.1:9200"
    ES_USERNAME = "elastic"
    ES_PASSWORD = "yourpassword"
    
    # GPU used in the project
    CUDA_VISIBLE_DEVICES = "0"
    
    # OPENAI api_key
    openai_api_key = "sk-xxxxx"
    openai_model_version = "gpt-4o"
    
    # google_search_key     You can go to https://serpapi.com/ to get it
    google_search_key = "xxxxx"
    
    
    # Proxy (used for openai api calls)
    HTTP_PROXY = "http://xxxx:xxxx"
    HTTPS_PROXY = "http://xxxx:xxxx"
    
    # ----------------------------------The following configurations do not need to be modified -----------------------------------
    
    model_path = root_path
    # self_rag model
    selfRAG_model_path = model_path + "dir_model/generator/selfrag_llama2_7b"
    if not os.path.exists(selfRAG_model_path):
        print("selfRAG Does not exist locally, requires online loading...")
        selfRAG_model_path = "selfrag/selfrag_llama2_7b"    
    # llama2 model
    llama2_model_path =  model_path + "dir_model/generator/Llama-2-7b-chat-hf"
    if not os.path.exists(llama2_model_path):
        print("llama2 Does not exist locally, requires online loading...")
        llama2_model_path = "meta-llama/Llama-2-7b-chat-hf"
    # baichuan2 model
    baichuan2_model_path =  model_path + "dir_model/generator/Baichuan2-13B-Chat"
    if not os.path.exists(baichuan2_model_path):
        print("baichuan2 Does not exist locally, requires online loading...")
        baichuan2_model_path = "baichuan-inc/Baichuan2-13B-Chat"    

    NED_model_path = model_path + "dir_model/generator/NED_model"
    if not os.path.exists(NED_model_path):
        print("NED Does not exist locally, requires online loading...")
        NED_model_path = "wikipedia_model_with_numbers"    

    WikiSP_model_path = model_path + "dir_model/generator/WikiSP_sparql_model"
    if not os.path.exists(WikiSP_model_path):
        print("WikiSP Does not exist locally, requires online loading...")
        WikiSP_model_path = "stanford-oval/llama-7b-wikiwebquestions-qald7"  

    # retriever_model_path
    # BGE model
    BGE_model_path = model_path + "dir_model/retriever/BGE/BGE-Reproduce"
    if not os.path.exists(BGE_model_path):
        print("BGE Does not exist locally, requires online loading...")
        BGE_model_path = "BAAI/bge-large-en-v1.5"
    # contriever model
    contriever_model_path = model_path + "dir_model/retriever/contriever/contriever_msmarco_model"
    
    if not os.path.exists(contriever_model_path):
        print("contriever Does not exist locally, requires online loading...")
        contriever_model_path = "nthakur/contriever-base-msmarco"
    # DPR model
    DPR_model_path = model_path + "dir_model/retriever/DPR/facebook-dpr-question_encoder-multiset-base"
    if not os.path.exists(DPR_model_path):
        print("DPR Does not exist locally, requires online loading...")
        DPR_model_path = "sentence-transformers/facebook-dpr-question_encoder-multiset-base"
        
    # BERT model
    BERT_model_path = model_path + "dir_model/retriever/BERT/Bert-base"
    if not os.path.exists(BERT_model_path):
        print("BERT Does not exist locally, requires online loading...")
        BERT_model_path = "google-bert/bert-base-uncased"

    # E5 model
    E5_model_path = model_path + "dir_model/retriever/E5/e5-mistral-7b-instruct"
    if not os.path.exists(E5_model_path):
        print("E5 Does not exist locally, requires online loading...")
        E5_model_path = "intfloat/e5-mistral-7b-instruct"
        
    # # query - model
    wikidata_base_model_path = "LLama_2_7b_hf"
    wikidata_peft_model_path = "llama-2-7b-sparql-8bit"
    scienceqa_model_path = "princeton-nlp/sup-simcse-roberta-large"
    
    # verbalizer model
    verbalizer_model_path = model_path + "dir_model/verlizer/t5-large_T-F_ID-T/val_avg_bleu=54.5600-step_count=3.ckpt"
    if not os.path.exists(verbalizer_model_path):
        print("verbalizer Does not exist locally, requires online loading...")
        verbalizer_model_path = ""   
        
    T5_model_path = model_path + "dir_model/verlizer/t5-large"
    if not os.path.exists(T5_model_path):
        print("T5 Does not exist locally, requires online loading...")
        T5_model_path = "google-t5/t5-large"   
