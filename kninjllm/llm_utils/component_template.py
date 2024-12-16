def get_RootClassParams_do_newNode(thisNode,jsondata):
    
    # common
    if "parent_type" in thisNode:
        for one_con in jsondata:
            if one_con['type'] == thisNode['parent_type']:
                one_child = one_con['children'][0]
                thisNode['initParams'] = one_child['initParams']
                thisNode['inputParams'] = one_child['inputParams']
                thisNode['outputParams'] = one_child['outputParams']
                if thisNode['parent_type'] == "KnowledgeUploader":
                    thisNode['type'] = thisNode['type'] + "_Interface"
                    thisNode['name'] = thisNode['name'] + "_Interface"
                return thisNode
    # controller
    else:
        return thisNode
        
def get_class_template_common(thisNode):
    
    print("-------------thisNode----------------")
    print(thisNode)
    
    final_str = ""
    codeTemplate = '''
from kninjllm.llm_common.component import component
from root_config import RootConfig
from kninjllm.llm_utils.common_utils import set_proxy,unset_proxy

@component
class {type}: 
    def __init__(self, {init_params}):
        {do_init_params}

    # do infer ....
    def infer(self,):
        pass

    # do evaluate ....
    def evaluate(self,):
        pass

    # do train ....
    def train(self,):
        pass

    @component.output_types({output_params})
    def run(self,{input_params}):
        # do something ...

        # example return
        return {return_params}
    '''
    init_params = ', '.join([f"{param['name']}=\"{param['value']}\"" for param in thisNode['initParams']])
    do_init_params = '\n\t\t'.join([f"self.{param['name']}={param['name']}" for param in thisNode['initParams']])
    input_params = ', '.join([f"{param['name']}:{param['type']}=\"{param['value']}\"" for param in thisNode['inputParams']])
    output_params = ', '.join([f"{param['name']}={param['type']}" for param in thisNode['outputParams']])
    return_params = ', '.join([f"\"{param['name']}\":\"\"" for param in thisNode['outputParams']])
    return_params = "{" + return_params + "}"
    if thisNode['parent_type'] == "KnowledgeUploader":
        type = "Querier"
    else:
        type = thisNode['type']
    filled_template = codeTemplate.format(type=type, init_params=init_params,do_init_params=do_init_params, input_params=input_params, output_params=output_params,return_params=return_params)
    return filled_template 


def get_class_template_controller(thisNode):
    # controller
    init_template = '''
from kninjllm.llm_common.component import component
{import_params}

{init_common_params}

@component
class {type}: 
    def __init__(self, {init_params}):
        {do_init_params}

    @component.output_types({output_params})
    def run(self,{input_params}):
        # do something ...

        # example return
        return {return_params}
        '''

    import_params = '\n'.join([f"from {param['codeFilePath'].replace('/','.').replace('.py','')} import {param['type']}" for param in thisNode['tempControllerValueConfig']])

    final_init_common_params = ""
    for common_compent in thisNode['tempControllerValueConfig']:
        common_params = ', '.join([f"{param['name']}=\"{param['value']}\"" for param in common_compent['initParams']])
        init_common_params = f"{common_compent['name']} = {common_compent['type']}({common_params}) \n'''\n {common_compent['description']} \n'''"


        final_init_common_params = final_init_common_params + "\n\n" +  init_common_params

    init_params = ', '.join([f"{param['name']}=\"{param['value']}\"" for param in thisNode['initParams']])
    init_params = init_params + " , " + ', '.join([f"{param['name']}={param['name']}" for param in thisNode['tempControllerValueConfig']])
    do_init_params = "\n\t\t"
    do_init_params = do_init_params + '\n\t\t'.join([f"self.{param['name']}={param['name']}" for param in thisNode['initParams']])
    do_init_params = do_init_params + '\n\t\t' + '\n\t\t'.join([f"self.{param['name']}={param['name']}" for param in thisNode['tempControllerValueConfig']])
    input_params = ', '.join([f"{param['name']}:{param['type']}=\"{param['value']}\"" for param in thisNode['inputParams']])
    output_params = ', '.join([f"{param['name']}={param['type']}" for param in thisNode['outputParams']])
    return_params = ', '.join([f"\"{param['name']}\":\"\"" for param in thisNode['outputParams']])
    return_params = "{" + return_params + "}"
    filled_template = init_template.format(type=thisNode['type'],
                                            init_params=init_params,do_init_params=do_init_params, input_params=input_params, output_params=output_params,return_params=return_params,
                                        import_params=import_params,init_common_params=final_init_common_params)

    return filled_template