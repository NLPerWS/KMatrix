import traceback

import time
import re
from urllib.parse import urlencode
import traceback
from jinja2 import Environment, FileSystemLoader, select_autoescape


from root_config import RootConfig
from kninjllm.llm_utils.common_utils import loadModelByCatch

from kninjllm.llm_generator.base_generator.wikidata_emnlp23.utils import get_query_sparql,natural_to_sparql,get_es_clent,insert_one_to_es,delete_many_by_es,find_one_by_es,find_by_es
from kninjllm.llm_generator.base_generator.wikidata_emnlp23.mention_heuristics import location_search


# ES index
client = get_es_clent()
name_to_pid_mapping = "name_to_pid_mapping"
qid_name_mapping = "name_to_qid_mapping"

def fill_template(template_file, prompt_parameter_values={}):
    jinja_environment = Environment(loader=FileSystemLoader(RootConfig.root_path + 'kninjllm/llm_generator/base_generator/wikidata_emnlp23/prompts/'),
                    autoescape=select_autoescape(), trim_blocks=True, lstrip_blocks=True, line_comment_prefix='#')
    template = jinja_environment.get_template(template_file)

    filled_prompt = template.render(**prompt_parameter_values)
    filled_prompt = '\n'.join([line.strip() for line in filled_prompt.split('\n')]) # remove whitespace at the beginning and end of each line
    return filled_prompt

# use
def get_name_from_qid(qid):

    # candidate = qid_name_mapping.find_one({"qid" : qid})
    candidate = find_one_by_es(client=client,index=qid_name_mapping,data={"qid":qid})
    
    # print(candidate)
    if candidate:
        return candidate["name"]
    
    else:
        time.sleep(1)
        # include the wd:Q part
        url = 'https://query.wikidata.org/sparql'
        query = '''
            SELECT ?label
            WHERE {{
            {} rdfs:label ?label.
            FILTER(LANG(?label) = "en").
            }}
        '''.format(qid)
        print("processing QID {}".format(qid))
        while True:
            try:
                # r = requests.get(url, params = {'format': 'json', 'query': query})
                r = get_query_sparql(url=url,params={'format': 'json', 'query': query})
                r.raise_for_status()
                break
            except:
                traceback.print_exc()
                time.sleep(100)
                print("wikidata query sleep 100s")
        try:
            name = r.json()["results"]["bindings"][0]["label"]["value"]
            
            
            print("Found {} with name {}".format(qid, name))
            # qid_name_mapping.insert_one({
            #         "qid": qid,
            #         "name": name
            #     }
            # )
            insert_one_to_es(client=client,index=qid_name_mapping,data={"qid": qid,"name": name})
        
            return name
        except Exception as e:
            return None


# use
def do_ned_for_dev_new(query,refined_model,mode= "refined"):
    if mode == "refined":
        refined = refined_model
        def refined_ned(utterance):
            spans = refined.process_text(utterance)
            output = set()
            for span in spans:
                if span.predicted_entity.wikidata_entity_id:
                    qid = span.predicted_entity.wikidata_entity_id
                    wikidata_name = get_name_from_qid("wd:" + qid)
                    if wikidata_name is not None:
                        output.add((wikidata_name, qid))
            
            return output    
        
        utterance = query
        pid_mapping_list = list(refined_ned(utterance))
        return pid_mapping_list

    elif mode == "oracle":
        pass

    else:
        raise ValueError


def evaluate_dev_new(query,pid_mapping_list):
    
    utterance = query
    _input = fill_template('property-name-gen.input', {
        "query": utterance,
        "qid_list_tuples": pid_mapping_list
    })
    _instruction = fill_template('property-name-gen.instruction')
    prompt = "Below is an instruction that describes a task, paired with an input that provides further context.\nWrite a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:".format(_instruction, _input)
    response = natural_to_sparql(prompt)
    return response

# use
def execute_predicted_sparql_new(sparql):
    
    # print(sparql)
    sparql = sparql.replace("wdt:instance_of/wdt:subclass_of", "wdt:P31/wdt:P279")
    # print(sparql)
    
    url = 'https://query.wikidata.org/sparql'
    extracted_property_names =  [x[1] for x in re.findall(r'(wdt:|p:|ps:|pq:)([a-zA-Z_\(\)(\/_)]+)(?![1-9])', sparql)]
    #print(extracted_property_names)
    pid_replacements = {}
    for replaced_property_name in extracted_property_names:
        # if not name_to_pid_mapping.find_one({"name" : replaced_property_name}):
        if find_one_by_es(client=client,index=name_to_pid_mapping,data={"name":replaced_property_name}) == None:
            
            i = replaced_property_name.replace('_', ' ').lower()
            pid_query = """
                SELECT ?property ?propertyLabel WHERE {
                ?property rdf:type wikibase:Property .
                ?property rdfs:label "%s"@en .
                SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
            }"""% i
            
            time.sleep(1)
            while True:
                try:
                    # response = requests.get(url, params={'format': 'json', 'query': pid_query})
                    response = get_query_sparql(url='https://query.wikidata.org/sparql',params={'format': 'json', 'query': pid_query})
                    response.raise_for_status()
                    data = response.json()
                    break
                except:
                    traceback.print_exc()
                    time.sleep(100)
                    print("wikidata query sleep 100s")
            if 'results' in data and 'bindings' in data['results'] and len(data['results']['bindings']) > 0:
                # Extract the property ID from the response
                property_id = data['results']['bindings'][0]['property']['value']
                property_id = property_id.replace('http://www.wikidata.org/entity/', '')
                
                print("inserting {} for {}".format(replaced_property_name, property_id))
                # name_to_pid_mapping.insert_one({
                #     "name": replaced_property_name,
                #     "pid": property_id
                # })
                insert_one_to_es(client=client,index=name_to_pid_mapping,data={"name":replaced_property_name,"pid":property_id})
                
            else:
                url = "https://www.wikidata.org/w/api.php"
                params = {
                    "action": "wbsearchentities",
                    "search": i,
                    "language": "en",
                    "limit": 20,
                    "format": "json",
                    "type": "property"
                }
                encoded_url = url + "?" + urlencode(params)
                # print(encoded_url)
                time.sleep(1)
                while True:
                    try:
                        # response = requests.get(encoded_url)
                        response = get_query_sparql(url=encoded_url)
                        
                        data = response.json()
                        break
                    except:
                        traceback.print_exc()
                        time.sleep(100)
                        print("wikidata query sleep 100s")
                if "search" in data and len(data["search"]) > 0:
                    property_id = data["search"][0]["id"]
                    print("inserting {} for {} by querying aliases for property".format(replaced_property_name, property_id))
                    # name_to_pid_mapping.insert_one({
                    #     "name": replaced_property_name,
                    #     "pid": property_id
                    # })
                    insert_one_to_es(client=client,index=name_to_pid_mapping,data={"name":replaced_property_name,"pid":property_id})
                else:
                    
                    print("CANNOT FIND PROPERTY: {} for SPARQL {}".format(replaced_property_name, sparql))
                    return sparql

        # pid = name_to_pid_mapping.find_one({"name" : replaced_property_name})["pid"]
        pid = find_one_by_es(client=client,index=name_to_pid_mapping,data={"name":replaced_property_name})["pid"]
        
        pid_replacements[replaced_property_name] = pid
    
    def sub_fcn(match):
        prefix = match.group(1)
        value = match.group(2)
        
        return prefix + pid_replacements[value]
    
    sparql = re.sub(r'(wdt:|p:|ps:|pq:)([a-zA-Z_\(\)(\/_)]+)(?![1-9])', lambda match: sub_fcn(match), sparql)
        
    # next, we need to replace the domain entities
    extracted_entity_names =  [x[1] for x in re.findall(r'(wd:)([a-zA-PR-Z_0-9-]+)', sparql)]
    #print(extracted_entity_names)
    qid_replacements = {}
    for extracted_entity_name in extracted_entity_names:
        if extracted_entity_name in ["anaheim_ca"]:
            # qid_name_mapping.delete_many({
            #     "name": extracted_entity_name
            # })
            delete_many_by_es(client=client,index=qid_name_mapping,data={"name":extracted_entity_name})
            
        
        found = False
        
        # temp_qid_list = qid_name_mapping.find()
        temp_qid_list = find_by_es(client=client,index=qid_name_mapping)
        
        for i in temp_qid_list:
            if i["name"] == extracted_entity_name and "qid" in i:
                found = True
                qid_replacements[extracted_entity_name] = i["qid"]
            elif i["name"].lower().replace(' ', '_').replace('/','_').replace('-', '_') == extracted_entity_name and "qid" in i:
                found = True
                qid_replacements[extracted_entity_name] = i["qid"]
                
        if not found:
            try_location = location_search(extracted_entity_name.replace("_", " "))
            if try_location is not None:
                try_location = "wd:" + try_location
                print("inserting {} for {}".format(try_location, extracted_entity_name))
                # qid_name_mapping.insert_one({
                #     "name": extracted_entity_name,
                #     "qid": try_location
                # })
                insert_one_to_es(client=client,index=qid_name_mapping,data={"name":extracted_entity_name,"qid":try_location})
                
                qid_replacements[extracted_entity_name] = try_location
            else:
                print("CANNOT FIND ENTITY: {} for SPARQL {}".format(extracted_entity_name, sparql))
                return sparql
    
    def sub_entity_fcn(match):
        value = match.group(2)
        return qid_replacements[value]
    
    sparql = re.sub(r'(wd:)([a-zA-PR-Z_0-9-]+)', lambda match: sub_entity_fcn(match), sparql)
        
    # finally, we can execute
    time.sleep(1)
    
    return sparql
    


def e2esparql_generate_flask(query):

    Refined_model = loadModelByCatch(model_name='NED',model_path=RootConfig.NED_model_path)
    print("Refined_model load success!!!")

    pid_mapping_list=do_ned_for_dev_new(query,Refined_model)
    print("pid_mapping_list:",pid_mapping_list)
    
    sparql_with_pid=evaluate_dev_new(query,pid_mapping_list)
    print("sparql_with_pid:",sparql_with_pid)
    
    
    sparql=execute_predicted_sparql_new(sparql_with_pid)
    
    
    return sparql
    