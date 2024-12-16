from kninjllm.llm_knowledgeUploader.utils.interface_execute import InterfaceExecute

domain_mapping = {
    "factual": {
        "wikipedia": InterfaceExecute(domain='factual/wikipedia',type='google',url='@wikipedia'),
        "wikidata": InterfaceExecute(domain='factual/wikidata',type='wiki',url='https://query.wikidata.org/sparql')
    },
    "biology": {
        "ck12": InterfaceExecute(domain='biology/ck12',type='google',url='@ck12'),
        "scienceqa_bio": InterfaceExecute(domain='biology/scienceqa_bio',type='local',url='dir_knowledge/online_interface/biology_only_output.jsonl')
    },
    "medical": {
        "uptodate": InterfaceExecute(domain='medical/uptodate',type='google',url='@uptodate'),
        "flashcard": InterfaceExecute(domain='medical/flashcard',type='local',url='dir_knowledge/online_interface/medical_only_output.jsonl')
    },
    "physical": {
        "physicsclassroom": InterfaceExecute(domain='physical/physicsclassroom',type='google',url='@physicsclassroom'),
        "scienceqa_phy": InterfaceExecute(domain='physical/scienceqa_phy',type='local',url='dir_knowledge/online_interface/physics_only_output.jsonl')
    }
}