from kninjllm.llm_knowledgeUploader.utils.interface_execute import InterfaceExecute

domain_mapping = {
    "factual": {
        "wikipedia": InterfaceExecute(domain='factual/wikipedia',type='google',url='@wikipedia'),
        "wikidata": InterfaceExecute(domain='factual/wikidata',type='wiki',url='https://query.wikidata.org/sparql')
    },
    "biology": {
        "scienceqa_bio": InterfaceExecute(domain='biology/scienceqa_bio',type='local',url='dir_knowledge/online_interface/biology_only_output.jsonl')
    },
    "medical": {
        "flashcard": InterfaceExecute(domain='medical/flashcard',type='local',url='dir_knowledge/online_interface/medical_only_output.jsonl')
    },
    "physical": {
        "scienceqa_phy": InterfaceExecute(domain='physical/scienceqa_phy',type='local',url='dir_knowledge/online_interface/physics_only_output.jsonl')
    }
}