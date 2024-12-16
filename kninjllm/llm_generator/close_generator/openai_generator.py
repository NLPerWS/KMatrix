import dataclasses
import json
import os
import re
import subprocess
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Union

from openai import OpenAI, Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
# from kninjllm.llm_common.document import Document
from kninjllm.llm_common.component import component
from kninjllm.llm_common.serialization import default_from_dict, default_to_dict
from kninjllm.llm_dataclasses import ChatMessage, StreamingChunk

from kninjllm.llm_utils.secret_utils import Secret,deserialize_secrets_inplace
from kninjllm.llm_utils.callable_serialization import deserialize_callable,serialize_callable
from root_config import RootConfig
from kninjllm.llm_utils.common_utils import set_proxy,unset_proxy

@component
class OpenAIGenerator:

    def __init__(
        self,
        api_key: str = "",
        model_path : str = "",
        executeType : str = "",
        tempModelCatch: list = [],
        do_log: bool = True,
        model: str = RootConfig.openai_model_version,
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
        api_base_url: Optional[str] = None,
        organization: Optional[str] = None,
        system_prompt: Optional[str] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        # tableQA
        prompt_path = None, 
        prompt_name = None, 
        max_tokens = None
    ):

        self.api_key = api_key
        if do_log:
            self.logSaver = RootConfig.logSaver
        else:
            self.logSaver = None
        self.model = model
        self.generation_kwargs = generation_kwargs or {}
        self.system_prompt = system_prompt
        self.streaming_callback = streaming_callback
        self.api_base_url = api_base_url
        self.organization = organization
        
        # self.client = OpenAI(api_key=api_key, organization=organization, base_url=api_base_url)
        # tableQA  kg  and  test_to_sql 
        if prompt_path != None or prompt_name != None:
            self.history_messages = []
            self.history_contents = []
            self.max_tokens = max_tokens
            self.prompt = self.load_prompt_template(prompt_path, prompt_name)
            self.idx_mapping = {"0": "first", "1": "second", "2": "third", "3": "fourth", "4": "fifth", "5": "sixth",
                                "6": "seventh",
                                "7": "eighth", "8": "ninth", "9": "tenth"}

    # ----------tableQA start------------------
    def get_response_v1(self, input_text, turn_type):
        message = self.create_message_v1(input_text, turn_type)
        self.history_contents.append(message['content'])
        self.history_messages.append(message)
        if self.logSaver is not None:
            self.logSaver.writeStrToLog("Function -> OpenAIGenerator -> get_response_v1(self, input_text, turn_type)")
            self.logSaver.writeStrToLog(f"Given generator prompt -> : {self.history_messages}")
        
        message = self.query_API_to_get_message(self.history_messages)
        self.history_contents.append(message.content)
        self.history_messages.append(message)
        response = message.content
        if self.logSaver is not None:
            self.logSaver.writeStrToLog("Returns generator reply -> : " + str(response))
        return response

    def create_message_v1(self, input_text, turn_type):
        if turn_type == "columns_select":
            template = self.prompt['columns_select']
            columns, question = input_text
            # question = question.capitalize()
            input_text = template.format(question=question, columns=columns)
        elif turn_type == 'rows_select':
            template = self.prompt['rows_select']
            selected_cols, rows, question = input_text
            # question = question.capitalize()
            input_text = template.format(selected_columns=selected_cols, rows=rows, question=question)
        elif turn_type == "ask_final_answer_or_next_question":
            question, serialized_table = input_text
            template = self.prompt['ask_final_answer_or_next_question']
            input_text = template.format(table=serialized_table, question=question)
        else:
            raise NotImplementedError
        message = {'role': 'user', 'content': input_text}
        return message


    def parse_result(self, result, turn_type):
        content = result['content'].strip()
        if turn_type in ["initial", "question_template"]:
            if "should be" in content:
                content = content.split("should be")[1].strip()
                if content.startswith('"') and content.endswith('"'):
                    content = content[1:-1]
                else:
                    matchObj = re.search(r'"(.*?)"', content)
                    if matchObj is not None:
                        content = matchObj.group()
                        content = content[1:-1]
                    else:
                        content = content.strip().strip('"')
                        print("Not exactly parse, we directly use content: %s" % content)

        return content

    def reset_history(self):
        self.history_messages = []
        self.history_contents = []

    def reset_history_messages(self):
        self.history_messages = []

    def load_prompt_template(self, prompt_path, prompt_name):
        if prompt_path.endswith(".json"):
            with open(prompt_path, "rb") as f:
                prompt = json.load(f)
            return prompt[prompt_name]

    # ----------tableQA end------------------
    
    
    # ----------kg  start----------------------------
    def get_response_kg(self, input_text, turn_type, tpe_name=None):
        if self.args.debug:
            message = self.create_message_kg(input_text, turn_type, tpe_name)
            self.history_messages.append(message)
            self.history_contents.append(message['content'])
            print("query API to get message:\n%s" % message['content'])
            # message = self.query_API_to_get_message(self.history)
            # self.history.append(message)
            # response = self.parse_result_kg(message)
            response = input("input the returned response:")
        else:
            message = self.create_message_kg(input_text, turn_type, tpe_name)
            self.history_messages.append(message)
            self.history_contents.append(message['content'])
            message = self.query_API_to_get_message(self.history_messages)
            self.history_messages.append(message)
            self.history_contents.append(message['content'])
            response = self.parse_result_kg(message, turn_type)
        return response

    def get_response_v1_kg(self, input_text, turn_type, tpe_name=None):
        if self.args.debug:
            message = self.create_message_v1_kg(input_text, turn_type)
            self.history_messages.append(message)
            self.history_contents.append(message['content'])
            print("query API to get message:\n%s" % message['content'])
            # message = self.query_API_to_get_message(self.history)
            # self.history.append(message)
            # response = self.parse_result_kg(message)
            response = input("input the returned response:")
        else:
            message = self.create_message_v1_kg(input_text, turn_type)
            self.history_messages.append(message)
            self.history_contents.append(message['content'])
            message = self.query_API_to_get_message(self.history_messages)
            self.history_messages.append(message)
            self.history_contents.append(message['content'])
            response = self.parse_result_v1_kg(message, turn_type)
        return response

    def create_message_kg(self, input_text, turn_type, tpe_name):
        if turn_type == "initial":  # the initial query
            instruction = self.prompt[turn_type]['instruction']
            template = self.prompt[turn_type]['init_template']
            self.question = input_text
            input_text = instruction + template.format(question=input_text, tpe=tpe_name)
        elif turn_type == "continue_template":
            input_text = self.prompt[turn_type]
        elif turn_type == "question_template":
            template = self.prompt[turn_type]
            input_text = template.format(idx=self.idx_mapping[input_text])
        elif turn_type == "answer_template":
            template = self.prompt[turn_type]
            if len(input_text) > 0:
                input_text = template["valid"].format(facts=input_text)
            else:
                input_text = template["invalid"]
        elif turn_type == "final_query_template":
            template = self.prompt[turn_type]
            input_text = template.format(question=self.question)
        else:
            raise NotImplementedError
        message = {'role': 'user', 'content': input_text}
        return message

    def create_message_v1_kg(self, input_text, turn_type):
        if turn_type == "instruction":  # the initial query
            instruction = self.prompt['instruction']
            input_text = instruction
        elif turn_type == "init_relation_rerank":
            template = self.prompt['init_relation_rerank']
            question, tpe, can_rels = input_text
            input_text = template.format(question=question, tpe=tpe, relations=can_rels)
        elif turn_type == "ask_question":
            template = self.prompt['ask_question']
            idx, relations = input_text
            idx = self.idx_mapping[idx]
            input_text = template.format(idx=idx, relations=relations)
        elif turn_type == "ask_answer":
            facts = input_text
            template = self.prompt['ask_answer']
            input_text = template.format(facts=facts)
        elif turn_type == "ask_final_answer_or_next_question":
            question, serialized_facts = input_text
            template = self.prompt['ask_final_answer_or_next_question']
            input_text = template.format(facts=serialized_facts, question=question)
        elif turn_type == "condition":
            input_text = self.prompt['continue_template']['condition']
        elif turn_type == "continue":
            input_text = self.prompt['continue_template']['continue']
        elif turn_type == "stop":
            input_text = self.prompt['continue_template']['stop']
        elif turn_type == 'relation_rerank':
            template = self.prompt['relation_rerank']
            question, can_rels = input_text
            input_text = template.format(question=question, relations=can_rels)
        else:
            raise NotImplementedError
        message = {'role': 'user', 'content': input_text}
        return message

    def query_API_to_get_message(self, messages,max_output_len=300):
        while True:
            try:
                set_proxy()
                self.client = OpenAI(api_key=self.api_key, organization=self.organization, base_url=self.api_base_url)
                res = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0,
                    max_tokens=max_output_len,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                unset_proxy()
                return res.choices[0].message
            except Exception as e:
                unset_proxy()
                traceback.print_exc()
                time.sleep(30)

    def parse_result_kg(self, result, turn_type):
        content = result['content'].strip()
        if turn_type in ["initial", "question_template"]:
            if "should be" in content:
                content = content.split("should be")[1].strip()
                if content.startswith('"') and content.endswith('"'):
                    content = content[1:-1]
                else:
                    matchObj = re.search(r'"(.*?)"', content)
                    if matchObj is not None:
                        content = matchObj.group()
                        content = content[1:-1]
                    else:
                        content = content.strip().strip('"')
                        print("Not exactly parse, we directly use content: %s" % content)

        return content

    def parse_result_v1_kg(self, result, turn_type):
        content = result['content'].strip()
        if turn_type in ["ask_question", "continue"]:
            if "the simple question:" in content:
                content = content.split("the simple question:")[1].strip()
                if content.startswith('"') and content.endswith('"'):
                    content = content[1:-1]
                else:
                    matchObj = re.search(r'"(.*?)"', content)
                    if matchObj is not None:
                        content = matchObj.group()
                        content = content[1:-1]
                    else:
                        content = content.strip().strip('"')
                        print("Not exactly parse, we directly use content: %s" % content)

        return content

    def parse_result_v2_kg(self, result, turn_type):
        content = result['content'].strip()

        return content

    def reset_history_kg(self):
        self.history_messages = []
        self.history_contents = []

    def reset_history_messages_kg(self):
        self.history_messages = []

    def reset_history_contents_kg(self):
        self.history_contents = []

    def load_prompt_template_kg(self, prompt_path, prompt_name):
        if prompt_path.endswith(".json"):
            with open(prompt_path, "rb") as f:
                prompt = json.load(f)
            return prompt[prompt_name]

    def get_response_v2_kg(self, input_text, turn_type):
        message = self.create_message_v2_kg(input_text, turn_type)
        self.history_messages.append(message)
        self.history_contents.append(message['content'])
        if self.logSaver is not None:
            self.logSaver.writeStrToLog("Function -> OpenAIGenerator -> get_response_v2_kg(self, input_text, turn_type)")
            self.logSaver.writeStrToLog(f"Given generator prompt -> : {self.history_messages}")
        message = self.query_API_to_get_message(self.history_messages)
        self.history_messages.append(message)
        self.history_contents.append(message.content)
        response = message.content.strip()
        if self.logSaver is not None:
            self.logSaver.writeStrToLog("Returns generator reply -> : " + str(response))
        return response

    def create_message_v2_kg(self, input_text, turn_type):
        if turn_type == "instruction":  # the initial query
            instruction = self.prompt['instruction']
            input_text = instruction
        # ykm
        # elif turn_type == "init_relation_rerank":
        #     template = self.prompt['init_relation_rerank']
        #     can_rels, question, tpe, hop = input_text
        #     if hop == 1:
        #         hop = "first"
        #     elif hop == 2:
        #         hop = "second"
        #     elif hop == 3:
        #         hop = "third"
        #     input_text = template.format(question=question, tpe=tpe, relations=can_rels, hop=hop)
        elif turn_type == "init_relation_rerank":
            template = self.prompt['init_relation_rerank']
            can_rels, question, tpe = input_text
            input_text = template.format(question=question, tpe=tpe, relations=can_rels)
        elif turn_type == "constraints_flag":
            template = self.prompt['constraints_flag']
            question, tpe, selected_relations = input_text
            if len(selected_relations) > 1:
                selected_relations = "are " + ", ".join(selected_relations)
            else:
                selected_relations = "is " + ", ".join(selected_relations)
            input_text = template.format(question=question, tpe=tpe, selected_relations=selected_relations)
        elif turn_type == "ask_final_answer_or_next_question":
            question, serialized_facts = input_text
            template = self.prompt['ask_final_answer_or_next_question']
            input_text = template.format(facts=serialized_facts, question=question)
        elif turn_type == "choose_constraints":
            question, relation_tails, tpe_name = input_text
            template = self.prompt['choose_constraints']
            input_text = template.format(question=question, relation_tails=relation_tails, tpe=tpe_name)
        elif turn_type == "final_query_template":
            template = self.prompt['final_query_template']
            input_text = template.format(question=input_text)
        elif turn_type == 'relation_rerank':
            template = self.prompt['relation_rerank']
            can_rels, question, tpe, selected_relations = input_text
            # if len(selected_relations) > 1:
            #     selected_relations = "are " + ", ".join(selected_relations)
            # else:
            #     selected_relations = "is " + ", ".join(selected_relations)
            selected_relations = "".join(selected_relations)
            input_text = template.format(question=question, relations=can_rels, tpe=tpe,
                                         selected_relations=selected_relations)
        elif turn_type == 'relation_rerank_2hop':
            template = self.prompt['relation_rerank_2hop']
            can_rels, question, tpe, sub_question, selected_relations = input_text
            sub_question = ", ".join(sub_question)
            selected_relations = ", ".join(selected_relations)
            input_text = template.format(question=question, relations=can_rels, tpe=tpe,
                                         first_sub_question=sub_question, first_relation=selected_relations)
        elif turn_type == 'relation_rerank_3hop':
            template = self.prompt['relation_rerank_3hop']
            can_rels, question, tpe, sub_question, selected_relations = input_text
            first_sub_question = sub_question[0]
            second_sub_question = sub_question[1]
            fisrt_relation = selected_relations[0]
            second_relation = selected_relations[1]
            input_text = template.format(question=question, relations=can_rels, tpe=tpe,
                                         first_sub_question=first_sub_question, first_relation = fisrt_relation,
                                         second_sub_question=second_sub_question, second_relation=second_relation)
        elif turn_type == 'direct_ask_final_answer':
            template = self.prompt['direct_ask_final_answer']
            question = input_text
            input_text = template.format(question=question)
        elif turn_type == 'final_answer_organize':
            template = self.prompt['final_answer_organize']
            input_text = template
        else:
            raise NotImplementedError
        message = {'role': 'user', 'content': input_text}
        return message

    # ----------kg  end--------------------------------------
    
    
    # ------------------------ text_to_sql  start ----------------------------
    def get_response_v1_text_to_sql(self, input_text, turn_type, max_output_len):
        message = self.create_message_v1_text_to_sql(input_text, turn_type)
        self.history_messages.append(message)
        self.history_contents.append(message['content'])
        if self.logSaver is not None:
            self.logSaver.writeStrToLog("Function -> OpenAIGenerator -> get_response_v1_text_to_sql(self, input_text, turn_type, max_output_len)")
            self.logSaver.writeStrToLog(f"Given generator prompt -> : {self.history_messages}")
        message = self.query_API_to_get_message(self.history_messages, max_output_len)
        self.history_messages.append(message)
        self.history_contents.append(message.content)
        response = self.parse_result_text_to_sql(message)
        if self.logSaver is not None:
            self.logSaver.writeStrToLog("Returns generator reply -> : " + str(response))
        return response

    def create_message_v1_text_to_sql(self, input_text, turn_type):
        if turn_type == "select_tab":
            template = self.prompt['free_generate']
            question, ser_table_name = input_text
            input_text = template.format(question=question, table=ser_table_name)
        elif turn_type == "reorg_sel_tab":
            template = self.prompt['table_column_select_reorganize']
            input_text = template
        elif turn_type == "ask_final_answers":
            question, ser_table_name, ser_fks = input_text
            if len(ser_fks) > 1:
                template = self.prompt['ask_final_answers']['has_fk']
                input_text = template.format(question=question, table=ser_table_name, fk=ser_fks)
            else:
                template = self.prompt['ask_final_answers']['no_fk']
                input_text = template.format(question=question, table=ser_table_name)
        else:
            raise NotImplementedError
        message = {'role': 'user', 'content': input_text}
        return message


    def parse_result_text_to_sql(self, result):
        content = result.content.strip()

        return content

    def reset_history_text_to_sql(self):
        self.history_messages = []
        self.history_contents = []

    def reset_history_messages_text_to_sql(self):
        self.history_messages = []

    def reseta_history_contents_text_to_sql(self):
        self.history_contents = []

    def load_prompt_template_text_to_sql(self, prompt_path, prompt_name):
        if prompt_path.endswith(".json"):
            with open(prompt_path, "rb") as f:
                prompt = json.load(f)
            return prompt[prompt_name]
    # ------------------------ text_to_sql  end ----------------------------
    
    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self.model}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns:
            The serialized component as a dictionary.
        """
        callback_name = serialize_callable(self.streaming_callback) if self.streaming_callback else None
        return default_to_dict(
            self,
            model=self.model,
            streaming_callback=callback_name,
            api_base_url=self.api_base_url,
            generation_kwargs=self.generation_kwargs,
            system_prompt=self.system_prompt,
            api_key=self.api_key.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OpenAIGenerator":
        """
        Deserialize this component from a dictionary.

        :param data:
            The dictionary representation of this component.
        :returns:
            The deserialized component instance.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        init_params = data.get("init_parameters", {})
        serialized_callback_handler = init_params.get("streaming_callback")
        if serialized_callback_handler:
            data["init_parameters"]["streaming_callback"] = deserialize_callable(serialized_callback_handler)
        return default_from_dict(cls, data)


    def get_res_content(self,prompt):
        set_proxy()
        message = ChatMessage.from_user(prompt)
        if self.system_prompt:
            messages = [ChatMessage.from_system(self.system_prompt), message]
        else:
            messages = [message]

        # update generation kwargs by merging with the generation kwargs passed to the run method
        generation_kwargs = {**self.generation_kwargs}

        # adapt ChatMessage(s) to the format expected by the OpenAI API
        openai_formatted_messages = self._convert_to_openai_format(messages)
        
        self.client = OpenAI(api_key=self.api_key, organization=self.organization, base_url=self.api_base_url)
        completion: Union[Stream[ChatCompletionChunk], ChatCompletion] = self.client.chat.completions.create(
            model=self.model,
            messages=openai_formatted_messages,  # type: ignore
            stream=self.streaming_callback is not None,
            **generation_kwargs,
        )

        completions: List[ChatMessage] = []
        if isinstance(completion, Stream):
            num_responses = generation_kwargs.pop("n", 1)
            if num_responses > 1:
                raise ValueError("Cannot stream multiple responses, please set n=1.")
            chunks: List[StreamingChunk] = []
            chunk = None

            # pylint: disable=not-an-iterable
            for chunk in completion:
                if chunk.choices and self.streaming_callback:
                    chunk_delta: StreamingChunk = self._build_chunk(chunk)
                    chunks.append(chunk_delta)
                    self.streaming_callback(chunk_delta)  # invoke callback with the chunk_delta
            completions = [self._connect_chunks(chunk, chunks)]
        elif isinstance(completion, ChatCompletion):
            completions = [self._build_message(completion, choice) for choice in completion.choices]
        # before returning, do post-processing of the completions
        for response in completions:
            self._check_finish_reason(response)
        content = "".join([message.content for message in completions])
        unset_proxy()
        return content

    @component.output_types(final_result=Dict[str, Any])
    def run(self,
            query_obj: Dict[str, Any],
            train_data_info:Dict[str, Any] = {},
            dev_data_info:Dict[str, Any] = {},
        ):
        print("------------------------------  OPENAI  -------------------------")
        
        if "question" in query_obj and query_obj['question'] != "":
            
            prompt = query_obj['question']
            if self.logSaver is not None:
                self.logSaver.writeStrToLog("Function -> OpenAIGenerator -> run")
                self.logSaver.writeStrToLog("Given generator prompt -> : " + prompt)
                
            content = self.get_res_content(prompt)

            if self.logSaver is not None:
                self.logSaver.writeStrToLog("Returns generator reply -> : " + content)
            
            final_result = {
                "prompt":prompt,
                "content":content,
                "meta":{"pred":{}},
                **query_obj
            } 
        
        else:
            final_result = {
                "prompt":"",
                "content":"",
                "meta":{"pred":{}},
                **query_obj
            }  
        
        return {"final_result":final_result}

    def _convert_to_openai_format(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """
        Converts the list of ChatMessage to the list of messages in the format expected by the OpenAI API.

        :param messages:
            The list of ChatMessage.
        :returns:
            The list of messages in the format expected by the OpenAI API.
        """
        openai_chat_message_format = {"role", "content", "name"}
        openai_formatted_messages = []
        for m in messages:
            message_dict = dataclasses.asdict(m)
            filtered_message = {k: v for k, v in message_dict.items() if k in openai_chat_message_format and v}
            openai_formatted_messages.append(filtered_message)
        return openai_formatted_messages

    def _connect_chunks(self, chunk: Any, chunks: List[StreamingChunk]) -> ChatMessage:
        """
        Connects the streaming chunks into a single ChatMessage.
        """
        complete_response = ChatMessage.from_assistant("".join([chunk.content for chunk in chunks]))
        complete_response.meta.update(
            {
                "model": chunk.model,
                "index": 0,
                "finish_reason": chunk.choices[0].finish_reason,
                "usage": {},  # we don't have usage data for streaming responses
            }
        )
        return complete_response

    def _build_message(self, completion: Any, choice: Any) -> ChatMessage:
        """
        Converts the response from the OpenAI API to a ChatMessage.

        :param completion:
            The completion returned by the OpenAI API.
        :param choice:
            The choice returned by the OpenAI API.
        :returns:
            The ChatMessage.
        """
        # function or tools calls are not going to happen in non-chat generation
        # as users can not send ChatMessage with function or tools calls
        chat_message = ChatMessage.from_assistant(choice.message.content or "")
        chat_message.meta.update(
            {
                "model": completion.model,
                "index": choice.index,
                "finish_reason": choice.finish_reason,
                "usage": dict(completion.usage),
            }
        )
        return chat_message

    def _build_chunk(self, chunk: Any) -> StreamingChunk:
        """
        Converts the response from the OpenAI API to a StreamingChunk.

        :param chunk:
            The chunk returned by the OpenAI API.
        :returns:
            The StreamingChunk.
        """
        # function or tools calls are not going to happen in non-chat generation
        # as users can not send ChatMessage with function or tools calls
        choice = chunk.choices[0]
        content = choice.delta.content or ""
        chunk_message = StreamingChunk(content)
        chunk_message.meta.update({"model": chunk.model, "index": choice.index, "finish_reason": choice.finish_reason})
        return chunk_message

    def _check_finish_reason(self, message: ChatMessage) -> None:
        """
        Check the `finish_reason` returned with the OpenAI completions.
        If the `finish_reason` is `length`, log a warning to the user.

        :param message:
            The message returned by the LLM.
        """
        if message.meta["finish_reason"] == "length":
            print(
                "The completion for index {index} has been truncated before reaching a natural stopping point. "
                "Increase the max_tokens parameter to allow for longer completions.",
                index=message.meta["index"],
                finish_reason=message.meta["finish_reason"],
            )
        if message.meta["finish_reason"] == "content_filter":
            print(
                "The completion for index {index} has been truncated due to the content filter.",
                index=message.meta["index"],
                finish_reason=message.meta["finish_reason"],
            )
