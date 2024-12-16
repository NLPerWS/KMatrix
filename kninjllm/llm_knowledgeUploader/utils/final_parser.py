import os
import json
from kninjllm.llm_knowledgeUploader.utils.knowl_query import retrieve_knowledge
from kninjllm.llm_knowledgeUploader.utils.other_prompts import hotpotqa_s1_prompt_demonstration, hotpotqa_s2_edit_prompt_demonstration
from root_config import RootConfig

class Parser:
    
    def __init__(self,data_list):
        self.data = data_list
        self.logSaver = RootConfig.logSaver
        self.s1_prompt_demonstration = hotpotqa_s1_prompt_demonstration
        self.s2_edit_prompt_demonstration = hotpotqa_s2_edit_prompt_demonstration
        
    def get_dataset(self):
        return self.data
    
    def get_retrieve_related_flag(self,generator,rationale,rationale_knowledge):
        
        query="Given two sentences, sentence1 and sentence2, if the two sentences are relevant, please reply 'YES', otherwise, please reply 'NO'. Do not reply with any other content."
        
        
        query=query+"\n\nsentence1: "+rationale+"\nsentence2: "+rationale_knowledge+"\nreply: "
        
    
        # cot_sc_responses = call_openai_api(RootConfig.openai_model_version,query, max_tokens=1024, temperature=0, n=1)
        cot_sc_responses = generator.run(query_obj={"question":query})['final_result']['content']
        
        
        # all_cot_text_response = [x.message.content.strip() for x in cot_sc_responses[0].choices][0]
        all_cot_text_response = [cot_sc_responses.strip()][0]
        
        print("get_retrieve_related_flag:",all_cot_text_response)
        
        if all_cot_text_response=='YES':
            return True
        else:
            return False
    
    def get_question(self, data_point):
    
        data_point["question_all"]=data_point["question"]
        return data_point["question"]
    
    def get_ground_truth(self, data_point):
        return data_point["answer"]
    
    def get_s1_prompt(self, question):
        
        return self.s1_prompt_demonstration + "Q: " + question.strip() + "\nA: "

    def get_s2_edit_prompt(self, generator,rationale, rationale_knowledge):
        
        sentence = self.s2_edit_prompt_demonstration + "Sentence: " + rationale + "\nKnowledge: "
        positive_retrieve_knowledge=""
        Flag=False
        for x in rationale_knowledge:
            for y in rationale_knowledge[x]:
                if rationale_knowledge[x][y] and self.get_retrieve_related_flag(generator,rationale,rationale_knowledge[x][y]):
                # if rationale_knowledge[x][y]:
                    # Flag=True
                    positive_retrieve_knowledge += rationale_knowledge[x][y].strip("\n") + "\n"
        
        sentence=sentence+positive_retrieve_knowledge
        sentence=sentence[:-1]
        sentence += "\nEdited sentence: "
        
        if positive_retrieve_knowledge:
            Flag=True
        return sentence,Flag

    def get_s3_consolidation_prompt(self, question, rationale_1, rationale_2):
        return self.s1_prompt_demonstration + "Q: " + question.strip() + "\nA: First, " + rationale_1 + " Second, " + rationale_2 + " The answer is "
    def get_cot_sc_results(self, data_point, generator, cot_prompt):
       
        generator.generation_kwargs['temperature'] = 0.7
        cot_sc_responses = generator.run(query_obj={"question":cot_prompt})['final_result']['content']
        generator.generation_kwargs['temperature'] = 0
        # try:
        if cot_sc_responses is not None:
            
            # all_cot_text_response = [x.message.content.strip() for x in cot_sc_responses[0].choices]
            all_cot_text_response = [cot_sc_responses.strip()]
            
            all_cot_results = []
            
            data_point["all_cot_text_response"]=all_cot_text_response
            
            for x in all_cot_text_response:
                if "The answer is" in x:
                    all_cot_results.append(x.split("The answer is")[1].strip().lower())
                elif "Therefore, the answer is" in x:
                    all_cot_results.append(x.split("Therefore, the answer is")[1].strip().lower())
                elif "the answer is" in x:
                    all_cot_results.append(x.split("the answer is")[1].strip().lower())
                elif "therefore, the answer is" in x:
                    all_cot_results.append(x.split("therefore, the answer is")[1].strip().lower())
                else:
                    pass
                
            all_cot_results = all_cot_results[:5]
            
            if len(all_cot_results)==0:
                data_point["cot_response"] = all_cot_text_response[0]
                data_point["cot_answer"] = ""
                data_point["cot_sc_score"] = 0
                data_point["cot_sc_response"] = all_cot_text_response[0]
                data_point["cot_sc_answer"] = ""
                
                try:
                    cot_sc_rationale_1 = all_cot_text_response[0].split("Second, ")[0].strip().split("First, ")[1].strip()
                except:
                    cot_sc_rationale_1 = all_cot_text_response[0].split("Second, ")[0].strip().split("First, ")[0].strip()
                
                try:
                    cot_sc_rationale_2 = all_cot_text_response[0].split("Second, ")[1].strip()
                except:
                    cot_sc_rationale_2 = all_cot_text_response[0].split("Second, ")[0].strip()
                
                data_point["cot_sc_rationales"] = [cot_sc_rationale_1,cot_sc_rationale_2]
                
                return data_point
            
            # find the most common answer and indices in all_cot_results
            most_common_answer = max(set(all_cot_results), key = all_cot_results.count)
            most_common_answer_indices = [i for i, x in enumerate(all_cot_results) if x == most_common_answer]
            
            sc_score = float(len(most_common_answer_indices)) / len(all_cot_results)
            
            # use the first answer as cot answer
            cot_answer = all_cot_results[0]
            
            # cot_sc answer and rationales
            cot_sc_text_response = all_cot_text_response[most_common_answer_indices[0]]
            try:
                cot_sc_rationale_1 = cot_sc_text_response.split("Second, ")[0].strip().split("First, ")[1].strip()
            except:
                cot_sc_rationale_1 = cot_sc_text_response.split("Second, ")[0].strip().split("First, ")[0].strip()
                
            # try:  
            if "The answer is" in cot_sc_text_response:
                try:
                    cot_sc_rationale_2 = cot_sc_text_response.split("Second, ")[1].strip().split("The answer is")[0].strip()
                except:
                    cot_sc_rationale_2 = cot_sc_text_response.split("Second, ")[0].strip().split("The answer is")[0].strip() 
            elif "Therefore, the answer is" in cot_sc_text_response:
                try:
                    cot_sc_rationale_2 = cot_sc_text_response.split("Second, ")[1].strip().split("Therefore, the answer is")[0].strip()
                except:
                    cot_sc_rationale_2 = cot_sc_text_response.split("Second, ")[0].strip().split("Therefore, the answer is")[0].strip()
            elif "therefore, the answer is" in cot_sc_text_response:
                try:
                    cot_sc_rationale_2 = cot_sc_text_response.split("Second, ")[1].strip().split("therefore, the answer is")[0].strip()
                except:
                    cot_sc_rationale_2 = cot_sc_text_response.split("Second, ")[0].strip().split("therefore, the answer is")[0].strip()
                    
            else:
                try:
                    cot_sc_rationale_2 = cot_sc_text_response.split("Second, ")[1].strip().split("the answer is")[0].strip() 
                except:
                    cot_sc_rationale_2 = cot_sc_text_response.split("Second, ")[0].strip().split("the answer is")[0].strip() 
                   
            cot_sc_answer = most_common_answer
            
        else:
            raise Exception("Stage 1: OpenAI API call failed")
        
        # store the results
        data_point["cot_response"] = all_cot_text_response[0]
        data_point["cot_answer"] = cot_answer
        data_point["cot_sc_score"] = sc_score
        data_point["cot_sc_response"] = cot_sc_text_response
        data_point["cot_sc_answer"] = cot_sc_answer
        data_point["cot_sc_rationales"] = [cot_sc_rationale_1, cot_sc_rationale_2]

        # self.logSaver.writeStrToLog("生成初始推理步骤及答案: " + str(data_point["cot_response"]))
        self.logSaver.writeStrToLog("Initial Reason Chain and Answer Generate: " + str(data_point["cot_response"]))
        

        return data_point
    
    
    def update_rationales_step_by_step(self, generator, data_point):
        domains = data_point["s1_domains"]
        rationales = [x.strip() for x in data_point["cot_sc_rationales"]]
        rationale_1 = rationales[0]
        rationale_2 = rationales[1]

        print("****** Editing Rationale 1 ...")
        # retrieve knowledge for rationale 1 first
        rationale_1_knowl = retrieve_knowledge(domains, rationale_1, data_point)


        # self.logSaver.writeStrToLog("针对第一跳推理，执行知识查询:")
        self.logSaver.writeStrToLog("Knowledge Query and Correction for First-Hop Reason:")
        # self.logSaver.writeStrToLog("第一跳推理内容: " + str(rationale_1) + " , 返回的知识: " + str(rationale_1_knowl))
        self.logSaver.writeStrToLog("Knowledge: " + str(rationale_1_knowl))

        # edit rationale 1 based on rationale 1_knowl
        s2_edit_prompt_rationale_1,FLAG = self.get_s2_edit_prompt(generator,rationale_1, rationale_1_knowl)
        
        
        data_point["FLAG_1"]=FLAG
        # print(s2_edit_prompt_rationale_1)
        if FLAG:
            # edited_rationale_1 = call_openai_api(model, s2_edit_prompt_rationale_1, max_tokens=1024, temperature=0, n=1)
            edited_rationale_1 = generator.run(query_obj={"question":s2_edit_prompt_rationale_1})['final_result']['content']
            
            # edited_rationale_1=[x.message.content.strip() for x in edited_rationale_1[0].choices][0]
            edited_rationale_1 = [edited_rationale_1.strip()][0]
            
            new_rationale_2_prompt = self.s1_prompt_demonstration + "Q: " + data_point["question_all"].strip() + "\nA: First, " + edited_rationale_1 + " Second, "

            # new_rationale_2_source = call_openai_api(model, new_rationale_2_prompt, max_tokens=1024, temperature=0, n=1)
            new_rationale_2_source = generator.run(query_obj={"question":new_rationale_2_prompt})['final_result']['content']
            
            # new_rationale_2_source=[x.message.content.strip() for x in new_rationale_2_source[0].choices][0] 
            new_rationale_2_source = [new_rationale_2_source.strip()][0] 
        
            if "The answer is" in new_rationale_2_source:
                new_rationale_2 = new_rationale_2_source.split("The answer is")[0].strip()
            elif "Therefore, the answer is" in new_rationale_2_source:
                new_rationale_2 = new_rationale_2_source.split("Therefore, the answer is")[0].strip()
            elif "therefore, the answer is" in new_rationale_2_source:
                new_rationale_2 = new_rationale_2_source.split("therefore, the answer is")[0].strip()
            else:
                new_rationale_2 = new_rationale_2_source.split("the answer is")[0].strip()
                
                
            data_point["new_rationale_2_prompt"] = new_rationale_2_prompt
        
            data_point["new_rationale_2_source"] = new_rationale_2_source
        else:
            edited_rationale_1=rationale_1
            new_rationale_2=rationale_2
            
        # self.logSaver.writeStrToLog("使用第一跳知识查询结果，校正第一跳推理:" + str(edited_rationale_1))
        self.logSaver.writeStrToLog("First-Hop Reason Correction:" + str(edited_rationale_1))
        
        # self.logSaver.writeStrToLog("使用校正后的第一跳推理，更新第二跳推理:" + str(new_rationale_2))
        self.logSaver.writeStrToLog("Second Reason Update:" + str(new_rationale_2))
            
        print("*** Original rationale 1:", rationale_1)
        print("*** Edited rationale 1:", edited_rationale_1)
        
        print("****** Editing Rationale 2 ...")
        
        print("*** New rationale 2:", new_rationale_2)
        
        data_point["rationale_1_knowl"] = rationale_1_knowl
        
        data_point["s2_edit_prompt_rationale_1"] = s2_edit_prompt_rationale_1
        
        data_point["edited_rationale_1"] = edited_rationale_1
        
        data_point["new_rationale_2"] = new_rationale_2

        # retreive knowledge for rationale 2
        rationale_2_knowl = retrieve_knowledge(domains, new_rationale_2, data_point)

        # self.logSaver.writeStrToLog("针对更新后的第二跳推理，执行知识查询: " )
        self.logSaver.writeStrToLog("Knowledge Query and Correction for Seconde-Hop Reason: " )
        # self.logSaver.writeStrToLog("更新后的第二跳推理内容: " + str(new_rationale_2) + " , 返回的知识: " + str(rationale_2_knowl) )
        self.logSaver.writeStrToLog("Knowledge: " + str(rationale_2_knowl) )


        # edit rationale 2 based on rationale 2_knowl
        s2_edit_prompt_rationale_2,FLAG_2 = self.get_s2_edit_prompt(generator,new_rationale_2, rationale_2_knowl)
        # print(s2_edit_prompt_rationale_2)
        data_point["FLAG_2"]=FLAG_2
        if FLAG_2:
            # edited_rationale_2 = call_openai_api(model, s2_edit_prompt_rationale_2, max_tokens=1024, temperature=0, n=1)
            edited_rationale_2 = generator.run(query_obj={"question":s2_edit_prompt_rationale_2})['final_result']['content']
            
            # edited_rationale_2=[x.message.content.strip() for x in edited_rationale_2[0].choices][0]
            edited_rationale_2 = [edited_rationale_2.strip()][0]
        else:
            edited_rationale_2=new_rationale_2
            
        # self.logSaver.writeStrToLog("使用第二跳知识查询结果，校正第二跳推理:" + str(edited_rationale_2))
        self.logSaver.writeStrToLog("Second-Hop Reason Correction:" + str(edited_rationale_2))
            
        print("*** Original rationale 2:", rationale_2)
        print("*** Edited rationale 2:", edited_rationale_2)

        # store the results
        data_point["rationale_2_knowl"] = rationale_2_knowl
        data_point["s2_edit_prompt_rationale_2"] = s2_edit_prompt_rationale_2
        data_point["edited_rationale_2"] = edited_rationale_2
        
        return data_point
    

        
    def update_rationales_at_once(self,generator, data_point):
        domains = data_point["s1_domains"]
        rationales = [x.strip() for x in data_point["cot_sc_rationales"]]
        rationale_1 = rationales[0]
        rationale_2 = rationales[1]

        print("****** Editing Rationale 1 ...")
        # retrieve knowledge for rationale 1 first
        rationale_1_knowl = retrieve_knowledge(domains, rationale_1, data_point)

        # edit rationale 1 based on rationale 1_knowl
        s2_edit_prompt_rationale_1 = self.get_s2_edit_prompt(generator,rationale_1, rationale_1_knowl)
        # print(s2_edit_prompt_rationale_1)
        
        # edited_rationale_1 = call_openai_api(model, s2_edit_prompt_rationale_1, max_tokens=256, temperature=0, n=1)[1].strip()
        edited_rationale_1 = generator.run(query_obj={"question":s2_edit_prompt_rationale_1})['final_result']['content'].strip()
        
        
        print("*** Original rationale 1:", rationale_1)
        print("*** Edited rationale 1:", edited_rationale_1)
        
        print("****** Editing Rationale 2 ...")        
        
        data_point["rationale_1_knowl"] = rationale_1_knowl
        data_point["edited_rationale_1"] = edited_rationale_1

        # retreive knowledge for rationale 2
        rationale_2_knowl = retrieve_knowledge(domains, rationale_2, data_point)

        # edit rationale 2 based on rationale 2_knowl
        s2_edit_prompt_rationale_2 = self.get_s2_edit_prompt(generator,rationale_2, rationale_2_knowl)
        # print(s2_edit_prompt_rationale_2)
        # edited_rationale_2 = call_openai_api(model, s2_edit_prompt_rationale_2, max_tokens=256, temperature=0, n=1)[1].strip()
        edited_rationale_2 = generator.run(query_obj={"question":s2_edit_prompt_rationale_2})['final_result']['content'].strip()
        
        print("*** Original rationale 2:", rationale_2)
        print("*** Edited rationale 2:", edited_rationale_2)

        # store the results
        data_point["rationale_2_knowl"] = rationale_2_knowl
        data_point["edited_rationale_2"] = edited_rationale_2

        return data_point
    
    def get_ans_text(self,text):
        text=text.strip().lower()[0]
        return text

    def get_final_answer(self, generator, data_point):
        print("****** Edited rationales: ", "First, " + data_point["edited_rationale_1"] + " Second, " + data_point["edited_rationale_2"])
        
        if data_point["edited_rationale_1"]==data_point["cot_sc_rationales"][0]  and data_point["edited_rationale_2"]==data_point["cot_sc_rationales"][1]:
            final_answer=data_point["cot_sc_answer"]
        else:
            s3_answer_consolidation_prompt = self.get_s3_consolidation_prompt(data_point["question_all"], data_point["edited_rationale_1"], data_point["edited_rationale_2"])
            # final_answer = call_openai_api(model, s3_answer_consolidation_prompt, max_tokens=1024, temperature=0, n=1)
            final_answer = generator.run(query_obj={"question":s3_answer_consolidation_prompt})['final_result']['content']
            
            show_final_answer = final_answer
            # self.logSaver.writeStrToLog("使用校正后的第一跳和第二跳推理，生成最终答案:" + str(show_final_answer))
            self.logSaver.writeStrToLog("Final Answer Generate:" + str(show_final_answer))
            
            
            # final_answer=[x.message.content.strip() for x in final_answer[0].choices][0]
            final_answer = [final_answer.strip()][0]
        
            data_point["s3_answer_consolidation_prompt"] = s3_answer_consolidation_prompt
        
        data_point["final_answer"] = final_answer
        return data_point
    
    
