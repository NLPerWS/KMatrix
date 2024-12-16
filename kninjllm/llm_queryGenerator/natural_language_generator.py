from typing import Any, Dict, List, Optional
from kninjllm.llm_common.component import component
from root_config import RootConfig
from openai import OpenAI, Stream
from kninjllm.llm_utils.common_utils import set_proxy,unset_proxy

@component
class Natural_language_generator:

    def __init__(
        self,
    ):

        self.np_parsing_demonstration = """
        Write a Question for given Declarative Statement.

        Declarative Statement: the 1993 World Champion figure skater's home country is Canada.
        Question: What is the home country of the 1993 World Champion figure skater?
        """


    def nl_parser(self,input_content,model_name=RootConfig.openai_model_version):

        prompt_question_input=self.np_parsing_demonstration+"\nDeclarative Statement: "+input_content+"\nQuestion: "
        set_proxy()
        Question_return=self.call_openai_api(model_name, prompt_question_input, max_tokens=1024, temperature=0, n=1)
        unset_proxy()
        
        Question_return=[x.message.content.strip() for x in Question_return[0].choices][0]

        return Question_return

    def call_openai_api(self,model, input_text, max_tokens=256, temperature=0, n=1):

        error_times = 0
        client = OpenAI(api_key=RootConfig.openai_api_key)

        while error_times < 5:
            try:
                if "text-davinci" in model:
                    # InstructGPT models, text completion
                    model="gpt-3.5-turbo-instruct"
                    response=client.completions.create(
                        model=model,
                        prompt=input_text,
                        max_tokens=max_tokens,
                        seed=0,
                        temperature=temperature,
                        n=n,
                    )
                    return [response, response.choices[0].text]
                elif "gpt-" in model:
                    # ChatGPT models, chat completion
                    response = client.chat.completions.create(
                        model=model,
                        messages = [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": input_text}
                        ],
                        seed=0,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        n=n,
                        timeout=60,
                        # request_timeout=60,
                    )
                    
                    return [response, response.choices[0].message.content]
                else:
                    raise Exception("Invalid model name")
            except Exception as e:
                print('Retry due to:', e)
                error_times += 1
                continue
            
        return None
    
    @component.output_types(final_query=Dict[str, Any])
    def run(
        self,
        query_obj: Dict[str, Any],
    ):
        # res_query = query_obj["question"]
        res_query = self.nl_parser(query_obj["question"])

        return {"final_result":res_query}
        