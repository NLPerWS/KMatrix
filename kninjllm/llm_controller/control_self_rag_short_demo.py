from typing import Any, Dict, List 
from vllm import LLM, SamplingParams
import random
import torch
import numpy as np
import json
import argparse
import re
from collections import Counter
import string
from kninjllm.llm_generator.base_generator.self_rag.retrieval_lm.utils import TASK_INST, PROMPT_DICT, load_special_tokens, load_jsonlines, postprocess, fix_spacing,control_tokens
from kninjllm.llm_common.component import component
from root_config import RootConfig
from kninjllm.llm_generator.base_generator.self_rag.self_rag_generator import RagGenerator
from kninjllm.llm_retriever.contriever.Contriever_retriever import Contriever_Retriever


@component
class SelfRagShortDemoController:
    def __init__(self,variableDataPath=""):
        self.topk = 3
        generator=RagGenerator(model_path=RootConfig.selfRAG_model_path,executeType="infer")
        retriever=Contriever_Retriever(top_k=self.topk,executeType="infer",model_path=RootConfig.contriever_model_path)
        self.contriever = retriever
        self.generator = generator
        self.tokenizer = self.generator.tokenizer
        self.variableDataPath = variableDataPath
        self.logSaver = RootConfig.logSaver
        seed = 633
        torch.backends.cudnn.deterministic = True
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    @component.output_types(final_result=List[Dict[str,Any]])
    def run(self,knowledge_info:Dict[str,Any],query_obj:Dict[str,Any] = {},query_list:List[Dict[str,Any]] = []):
        print("-------------------------- run short !!! -----------------------------")
        self.contriever.run(knowledge_info = knowledge_info)
        if query_obj == {} and len(query_list) == 0:
            return {"final_result":[]}
        if len(query_list) == 0:
            query_list = [query_obj]
        parser = argparse.ArgumentParser()
        args = parser.parse_args()
        args.device = "cuda"
        args.world_size = 1
        args.max_new_tokens = 500
        args.threshold = 0.4
        args.beam_width = 1
        args.max_depth = 2
        args.w_rel = 1.0
        args.w_sup = 1.0
        args.w_use = 1.0
        args.metric = "match"
        args.use_groundness = True
        args.use_utility = True
        args.use_seqscore = True
        args.task = ""
        args.dtype = "half"
        args.mode = "adaptive_retrieval"
        query_list = self.preprocess_input_data(query_list, task=args.task)
        
        query_list = list(map(lambda x:{"ctxs":[],**x},query_list))
        retriever_input_list = list(map(lambda x:{"question":x['question']},query_list))
        res_ctxs = self.contriever.run(query_list=retriever_input_list)['final_result']
        for query,res_ctx in zip(query_list,res_ctxs):
            query['ctxs'] = res_ctx
        
        # Get token ids for reflection tokens.
        if self.tokenizer != None:
            ret_tokens, rel_tokens, grd_tokens, ut_tokens = load_special_tokens(
                self.tokenizer, use_grounding=args.use_groundness, use_utility=args.use_utility)
        else:
            ret_tokens = None
            rel_tokens = None
            grd_tokens = None
            ut_tokens = None
        final_result = []
        count = 0
        for query in query_list:
            prompt = PROMPT_DICT["prompt_no_input"].format_map(query)
            evidences = query['ctxs']
            pred, results, do_retrieve = self.call_model_rerank_w_scores_batch(prompt=prompt, evidences=evidences, max_new_tokens=args.max_new_tokens,
                                                                rel_tokens=rel_tokens, ret_tokens=ret_tokens, grd_tokens=grd_tokens, ut_tokens=ut_tokens,
                                                                threshold=args.threshold, use_seqscore=args.use_seqscore,
                                                                w_rel=args.w_rel, w_sup=args.w_sup, w_use=args.w_use, mode=args.mode, closed=args.task in ["fever", "arc_c"])
            count += 1
            if type(pred) is str and pred[0] == "#" or pred[0] == ":":
                pred = pred[1:]
            final_result.append({"question":query['question'],"prompt": prompt,"content":pred,"tag":query['tag'],"ctxs": query['ctxs']})    # "meta":{ "pred": pred,  "results": results}
        if self.variableDataPath != "":
            with open(self.variableDataPath,'a',encoding='utf-8') as f:
                f.write(json.dumps({"component":"SelfRagShortDemoController","result":final_result},ensure_ascii=False)+"\n")
        
        return {"final_result":final_result}

    def postprocess_answer_option_conditioned(self,answer):
        for token in control_tokens:
            answer = answer.replace(token, "")

        if "</s>" in answer:
            answer = answer.replace("</s>", "")
        if "\n" in answer:
            answer = answer.replace("\n", "")

        if "<|endoftext|>" in answer:
            answer = answer.replace("<|endoftext|>", "")

        return answer

    def call_model_rerank_w_scores_batch(self,prompt, evidences, max_new_tokens=15,
                                        ret_tokens=None, rel_tokens=None, grd_tokens=None, ut_tokens=None,
                                        use_seqscore=False, threshold=0.5,
                                        w_rel=1.0, w_sup=1.0, w_use=0.5, mode="adaptive_retrieval", closed=False):
        results = {}
        if mode != "always_retrieve":
            sampling_params = SamplingParams(
                temperature=0.0, top_p=1.0, max_tokens=max_new_tokens, logprobs=32016)
            
            pred = self.generator.run(query_obj={"question":prompt}, sampling_params=sampling_params,saveLogFlag=False)['final_result']['meta']['pred']
            
            pred_token_ids = pred['outputs'][0]['token_ids']
            pred_text = pred['outputs'][0]['text']
            pred_log_probs = pred['outputs'][0]['logprobs']
            
            results["no_retrieval"] = pred_text

        # save relevance token scores
        if mode == "always_retrieve":
            do_retrieve = True

        elif mode == "no_retrieval":
            do_retrieve = False

        else:
            if threshold is not None:
                print("---------------------threshold is not None:-------------------------------",threshold)
                score_dict = {}
                for tok, id in ret_tokens.items():
                    if id not in pred_log_probs[0]:
                        score_dict[tok] = -100
                    prob = pred_log_probs[0][id]
                    score_dict[tok] = float(prob)
                do_retrieve = score_dict["[Retrieval]"] / (
                    score_dict["[Retrieval]"] + score_dict["[No Retrieval]"]) > threshold
            else:
                do_retrieve = "[Retrieval]" in pred
        
        print("-------------------do_retrieve-------------------")
        print(do_retrieve)

        if do_retrieve is True:
            evidence_augmented_inputs = [prompt + "[Retrieval]<paragraph>{0}\n{1}</paragraph>".format(
                para["title"], para["content"]) for para in evidences]
                
            sampling_params = SamplingParams(
                temperature=0.0, top_p=1.0, max_tokens=max_new_tokens, logprobs=5000)
            preds = []
            for input_i in evidence_augmented_inputs:
                # pred = self.generator.run(query_obj = {"question":input_i}, sampling_params=sampling_params)['final_result']['meta']['pred']
                pred = self.generator.run(query_obj = {"question":input_i}, sampling_params=sampling_params,saveLogFlag=False)['final_result']['meta']['pred']
                preds.append(pred)
            
            relevance_score_dict = {}
            grd_score_dict = {}
            ut_score_dict = {}
            overall_scores = {}
            for p_idx, pred in enumerate(preds):

                pred_token_ids = pred['outputs'][0]['token_ids']
                pred_text = pred['outputs'][0]['text']
                pred_log_probs = pred['outputs'][0]['logprobs']
                seq_score = pred['outputs'][0]['cumulative_logprob'] / \
                    max(len(pred['outputs'][0]['token_ids']), 1)


                relevance_score_dict.setdefault(p_idx, {})
                grd_score_dict.setdefault(p_idx, {})
                ut_score_dict.setdefault(p_idx, {})
                # Compute reward scores
                for tok, id in rel_tokens.items():
                    prob = pred_log_probs[0][id] if id in pred_log_probs[0] else -100
                    relevance_score_dict[p_idx][tok] = np.exp(float(prob))

                if grd_tokens is not None:
                    groundness_token_appear_indices = []
                    for tok_idx, tok in enumerate(pred_token_ids):
                        if tok in list(grd_tokens.values()):
                            groundness_token_appear_indices.append(tok_idx)
                            break
                    if len(groundness_token_appear_indices) > 0:
                        idx = groundness_token_appear_indices[0]
                        for token, token_id in grd_tokens.items():
                            prob = pred_log_probs[idx][token_id] if token_id in pred_log_probs[idx] else -100
                            grd_score_dict[p_idx][token] = np.exp(float(prob))

                if ut_tokens is not None:
                    utility_token_appear_indices = []
                    for tok_idx, tok in enumerate(pred_token_ids):
                        if tok in list(ut_tokens.values()):
                            utility_token_appear_indices.append(tok_idx)
                    if len(utility_token_appear_indices) > 0:
                        idx = utility_token_appear_indices[0]
                        for token, token_id in ut_tokens.items():
                            prob = pred_log_probs[idx][token_id] if token_id in pred_log_probs[idx] else -100
                            ut_score_dict[p_idx][token] = np.exp(float(prob))

                relevance_score = relevance_score_dict[p_idx]["[Relevant]"] / (
                    np.sum(list(relevance_score_dict[p_idx].values())))

                if len(grd_score_dict[p_idx]) == 3:
                    gt_sum = np.sum(list(grd_score_dict[p_idx].values()))
                    ground_score = (grd_score_dict[p_idx]["[Fully supported]"] / gt_sum) + 0.5 * (
                        grd_score_dict[p_idx]["[Partially supported]"] / gt_sum)
                else:
                    ground_score = 0.0

                if len(ut_score_dict[p_idx]) == 5:
                    ut_sum = np.sum(list(ut_score_dict[p_idx].values()))
                    ut_scores = [-1, -0.5, 0, 0.5, 1]
                    utility_score = np.sum(
                        [ut_scores[i] * (ut_score_dict[p_idx]["[Utility:{}]".format(i+1)] / ut_sum) for i in range(len(ut_scores))])
                else:
                    utility_score = 0.0

                if use_seqscore is True:
                    final_score = np.exp(seq_score) + w_rel * relevance_score + \
                        w_sup * ground_score + w_use * utility_score
                else:
                    final_score = w_rel * relevance_score + \
                        w_sup * ground_score + w_use * utility_score

                overall_scores[p_idx] = {"final_score": final_score,
                                        "relevance_score": relevance_score,
                                        "ground_score": ground_score,
                                        "utility_score": utility_score,
                                        "relevance_score_dict": relevance_score_dict,
                                        "grd_score_dict": grd_score_dict,
                                        "ut_score_dict": utility_score}
                results["retrieval_{}".format(p_idx)] = {
                    "pred": pred_text, "score": final_score, "ctx": evidences[p_idx]}

        else:
            sampling_params = SamplingParams(
                temperature=0.0, top_p=1.0, max_tokens=max_new_tokens)
            prompt += "[No Retrieval]"

            preds = []
            for one_prompt in [prompt]:
                pred = self.generator.run(query_obj={"question":one_prompt}, sampling_params=sampling_params)['final_result']['meta']['pred']
                preds.append(pred)
            pred = preds[0]['outputs'][0]['text']

        # Aggregating answers
        if len(results) == 1:
            postprocessed_pred = self.postprocess_answer_option_conditioned(pred)
            return postprocessed_pred, results, do_retrieve
        else:
            answer2score = {}
            if closed is True:
                for key, result in results.items():
                    if key == "no_retrieval":
                        continue
                    answer = self.postprocess_answer_option_conditioned(result["pred"])
                    score = result["score"]
                    answer2score.setdefault(answer, 0)
                    answer2score[answer] += score
                sorted_answers = sorted(
                    answer2score.items(), key=lambda x: x[1], reverse=True)
                best_option = sorted_answers[0][0]
            else:
                path2score = {key: item["score"] for key,
                            item in results.items() if key != "no_retrieval"}
                best_path = sorted(path2score.items(),
                                key=lambda x: x[1], reverse=True)[0][0]
                best_option = results[best_path]["pred"]
                print("------------------best_option = results[best_path]-----------------------")
                print(best_path)
                print(results[best_path])
            
            best_path_index = int(best_path.replace("retrieval_",""))
            if self.logSaver is not None:
                self.logSaver.writeStrToLog("Function -> RagGenerator -> run")
                self.logSaver.writeStrToLog("Given generator prompt -> : " + evidence_augmented_inputs[best_path_index])
                self.logSaver.writeStrToLog("Returns generator reply -> : "+best_option)
            
            return best_option, results, do_retrieve

    def process_data_evidences(self,demonstration, top_n):
        prompt = PROMPT_DICT["prompt_no_input"].format_map(demonstration)
        
        evidences = []
        return prompt, evidences

    def preprocess_input_data(self,dataset, task=None):
        new_data = []
        if task in TASK_INST:
            instruction = TASK_INST[task]
        else:
            instruction = None
        for item in dataset:
            
            if "golden_answers" in item and "answers" not in item:
                item['answers'] = item['golden_answers']
            
            if "tag" not in item:
                item['tag'] = ""
            
            if task == "arc_c" and "choices" in item:
                choices = item["choices"]
                answer_labels = {}
                for i in range(len(choices["label"])):
                    answer_key = choices["label"][i]
                    text = choices["text"][i]
                    if answer_key == "1":
                        answer_labels["A"] = text
                    if answer_key == "2":
                        answer_labels["B"] = text
                    if answer_key == "3":
                        answer_labels["C"] = text
                    if answer_key == "4":
                        answer_labels["D"] = text
                    if answer_key in ["A", "B", "C", "D"]:
                        answer_labels[answer_key] = text

                if "D" not in answer_labels:
                    answer_labels["D"] = ""
                choices = "\nA: {0}\nB: {1}\nC: {2}\nD: {3}".format(
                    answer_labels["A"], answer_labels["B"], answer_labels["C"], answer_labels["D"])
                if "E" in answer_labels:
                    choices += "\nE: {}".format(answer_labels["E"])
                item["instruction"] = instruction + \
                    "\n\n### Input:\n" + item["question"] + choices
                item["answers"] = [item["answerKey"]]
            # QA
            elif task == "arc_c" and "choices" not in item:
                item["instruction"] = item['question']
            else:
                prompt = instruction + "\n\n## Input:\n\n" + \
                    item["question"] if instruction is not None else item["question"]
                item["instruction"] = prompt
            
            new_data.append(item)

        return new_data


class metrics:
    def exact_match_score(self,prediction, ground_truth):
        return (self.normalize_answer(prediction) == self.normalize_answer(ground_truth))

    def metric_max_over_ground_truths(self,metric_fn, prediction, ground_truths):
        scores_for_ground_truths = []
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(score)
        return max(scores_for_ground_truths)

    def accuracy(self,preds, labels):
        match_count = 0
        for pred, label in zip(preds, labels):
            target = label[0]
            if pred == target:
                match_count += 1

        return 100 * (match_count / len(preds))


    def f1(self,decoded_preds, decoded_labels):
        f1_all = []
        for prediction, answers in zip(decoded_preds, decoded_labels):
            if type(answers) == list:
                if len(answers) == 0:
                    return 0
                f1_all.append(np.max([self.qa_f1_score(prediction, gt)
                            for gt in answers]))
            else:
                f1_all.append(self.qa_f1_score(prediction, answers))
        return 100 * np.mean(f1_all)


    def qa_f1_score(self,prediction, ground_truth):
        prediction_tokens = self.normalize_answer(prediction).split()
        ground_truth_tokens = self.normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1


    def normalize_answer(self,s):
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def find_entity_tags(self,sentence):
        entity_regex = r'(.+?)(?=\s<|$)'
        tag_regex = r'<(.+?)>'
        entity_names = re.findall(entity_regex, sentence)
        tags = re.findall(tag_regex, sentence)

        results = {}
        for entity, tag in zip(entity_names, tags):
            if "<" in entity:
                results[entity.split("> ")[1]] = tag
            else:
                results[entity] = tag
        return results

    def match(self,prediction, ground_truth):
        for gt in ground_truth:
            if gt in prediction:
                return 1
        return 0

metrics_run = metrics()
