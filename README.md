# ⚡KMatrix:  A Flexible Heterogeneous Knowledge Enhancement Toolkit for Large Language Model



<img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace%20Datasets-27b3b4.svg"> <img src="https://img.shields.io/npm/l/vue-web-terminal.svg"><img src="https://img.shields.io/badge/made_with-Python-blue">



KMatrix is a flexible heterogeneous knowledge enhancemant toolkit for LLMs.  Our toolkit contains seven stages to complete knowledge-enhanced generation task. All stages are implemented based on our modular component definitions.  Meanwhile, we design a control-logic flow diagram to combine components. The key features of KMatrix include: 

1. Unified enhancement of heterogeneous knowledge: KMatrix uses both verbalizing-retrieval and parsing-query methods to support unified enhancement of heterogeneous knowledge (like free-textual knowledge, tables, knowledge graphs, etc).
2. Systematical adaptive enhancement methods integration: KMatrix offers comprehensive adaptive enhancement methods including retrieval timing judgment and knowledge source selection. 
3. High customizability and easy combinability: our modularity and control-logic flow diagram design flexibly supports the entire lifecycle of various complex Knowledge Enhanced Large Language Models (K-LLMs) systems, including training, evaluation, and deployment.
4. Comprehensive evaluation of K-LLMs systems enhanced by heterogeneous knowledge: we integrate a rich collection of representative K-LLMs knowledge, datasets, and methods, and provide performance analysis of heterogeneous knowledge enhancement. 

![image](images/kmatrix_system.png)



## :wrench: Installation 

To get started with KMatrix, simply clone it from Github and install (requires Python 3.7+ ,  Python 3.10 recommended): 


    $ git clone https://github.com/NLPerWS/KMatrix.git
    
    # It is recommended to use a virtual environment for installation
    $ conda create -n KMatrix python=3.10
    $ conda activate KMatrix
    
    # Install backend environment
    $ cd KMatrix
    $ pip install -r requirements.txt
    
    # Install Frontend environment
    # You need a node environment, and nvm is recommended for node environment management
    # Recommended node environments: 16.20.2
    # You can refer to the fellowing websites to install nvm
    # https://nvm.uihtm.com/#nvm-linux 
    # https://github.com/nvm-sh/nvm
    # After installing the node environment, execute:
    $ cd easy-flow
    $ npm install
    
    # Then, you need to install some third-party tools required by our toolkit
    # Install ES database using Docker
    $ docker pull elasticsearch:8.11.1
    $ docker run -idt  \
        -p 9200:9200 -p 9300:9300 \
        -e "discovery.type=single-node" \
        -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" \
        -e "xpack.security.enabled=true" \
        -e "xpack.security.enrollment.enabled=true" \
        -e "ELASTIC_PASSWORD=yourpassword" \
        -v $(pwd)/elasticsearch_data:/usr/share/elasticsearch/data \
        -v $(pwd)/esplugins:/usr/share/elasticsearch/plugins \
        --name elasticsearch elasticsearch:8.11.1
        
    We upload knowledge and the datasets to ModelScope: https://modelscope.cn/datasets/zhangyujie/KMatrix_Rep/files
    Use the following commands to download:
    $ git lfs install
    $ git clone https://www.modelscope.cn/datasets/zhangyujie/KMatrix_Rep.git
    you can download it, it contains:
    dir_dataset/
    dir_knowledge/
    dir_model/
    use the downloaded folders to replace the original folders in KMatrix directory.
    
    # You need to import knowledge into ES database for Retrieval, our knowledge samples locate in dir_knowledge/local_knowledge_verlized/, you need to create three indexes in ES named 'wikipedia','wikidata','wikitable', and their corresponding files are respectively Wikipedia/wikipedia.jsonl, wikidata/wiki_data.jsonl, wikitable/wikitable.jsonl. After that, you can use retrieval model to retrieve the knowledge in the database.




## :rocket: Quick Start


    If you have successfully installed the environment, a quick start will be easy.
    
    1. Set configurations that needs to be modified in the root_config.py file located in the project root directory, if necessary. Set the SERVER_HOST in easy-flow/src/assets/js/host.js to the IP address of deployment server.
    
    2. Start the toolkit by executing following command: 
    $ cd KMatrix/easy-flow
    $ npm install
    $ npm run dev
    $ cd KMatrix
    $ python flask_server.py
    Visit KMatrix toolkit using the browser: http://yourserverip:8000
    
    3. Construct and Execute Flow diagram
    You can construct K-LLMs systems using our modular component and control-logic flow diagram, and execute it. Details of K-LLMs systems construction can be found in toolkit usage. You can use a flow diagram we have built (a K-LLMs system actively querying multiple knowledge interfaces) for a quick start:
    Click the [Use exising diagram] drop-down box, select Deployment/v16_cok_de_diagram, and then click the [Deploy diagram] button to start the deployment. After the deployment completes, enter your question in the question box and click [send] to generate reasoning steps and answer.



