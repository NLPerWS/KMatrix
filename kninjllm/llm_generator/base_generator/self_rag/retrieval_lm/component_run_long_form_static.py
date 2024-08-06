import os
import jsonlines
import json
import copy
import re
from transformers import AutoTokenizer
import numpy as np
import json
import argparse
from vllm import LLM, SamplingParams
from utils import TASK_INST, PROMPT_DICT, load_special_tokens, load_jsonlines, postprocess, fix_spacing
from flask import Flask, jsonify, request
from flask_cors import *

class SelfRAGenerator:
    
    def __init__(self,model_name):
        self.model = LLM(model=model_name)
        self.contriever = None
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        
        
    PROMPT_DICT = {
        "prompt_input": (
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
        ),
        "prompt_no_input": (
            "### Instruction:\n{instruction}\n\n### Response:\n"
        ),
        "prompt_no_input_retrieval": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Paragraph:\n{paragraph}\n\n### Instruction:\n{instruction}\n\n### Response:"
        ),
        "prompt_open_instruct": (
            "<user>\n{instruction}\n"
            "<assistant>\n"
        ),
        "prompt_open_instruct_retrieval": (
            "<user>\nReference:{paragraph}\n{instruction}\n"
            "<assistant>\n"
        ),
        "llama_chat_prompt": (
            "[INST]{instruction}[/INST]"
        ),
        "llama_chat_prompt_retrieval": (
            "[INST]{paragraph}\n{instruction}[/INST]"
        ),
    }

    TASK_INST = {"wow": "Given a chat history separated by new lines, generates an informative, knowledgeable and engaging response. ",
                "fever": "Is the following statement correct or not? Say true if it's correct; otherwise say false.",
                "eli5": "Provide a paragraph-length response using simple words to answer the following question.",
                "obqa": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
                "arc_easy": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
                "arc_c": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
                "trex": "Given the input format 'Subject Entity [SEP] Relationship Type,' predict the target entity.",
                "asqa": "Answer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers."}

    rel_tokens_names = ["[Irrelevant]", "[Relevant]"]
    retrieval_tokens_names = ["[No Retrieval]",
                            "[Retrieval]", "[Continue to Use Evidence]"]
    utility_tokens_names = ["[Utility:1]", "[Utility:2]",
                            "[Utility:3]", "[Utility:4]", "[Utility:5]"]
    ground_tokens_names = ["[Fully supported]",
                        "[Partially supported]", "[No support / Contradictory]"]
    other_special_tokens = ["<s>", "</s>", "[PAD]",
                            "<unk>", "<paragraph>", "</paragraph>"]
    control_tokens = ["[Fully supported]", "[Partially supported]", "[No support / Contradictory]", "[No Retrieval]", "[Retrieval]",
                    "[Irrelevant]", "[Relevant]", "<paragraph>", "</paragraph>", "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]"]


    def load_special_tokens(self,tokenizer, use_grounding=False, use_utility=False):
        ret_tokens = {token: tokenizer.convert_tokens_to_ids(
            token) for token in self.retrieval_tokens_names}
        rel_tokens = {}
        for token in ["[Irrelevant]", "[Relevant]"]:
            rel_tokens[token] = tokenizer.convert_tokens_to_ids(token)

        grd_tokens = None
        if use_grounding is True:
            grd_tokens = {}
            for token in self.ground_tokens_names:
                grd_tokens[token] = tokenizer.convert_tokens_to_ids(token)

        ut_tokens = None
        if use_utility is True:
            ut_tokens = {}
            for token in self.utility_tokens_names:
                ut_tokens[token] = tokenizer.convert_tokens_to_ids(token)

        return ret_tokens, rel_tokens, grd_tokens, ut_tokens


    def fix_spacing(self,input_text):
        # Add a space after periods that lack whitespace
        output_text = re.sub(r'(?<=\w)([.!?])(?=\w)', r'\1 ', input_text)
        if output_text.endswith("[Retrieval]"):
            index = output_text.rfind("[Retrieval]")
            output_text = output_text[:index].replace("[Retrieval]", "").replace("[No Retrieval]", "")
            return output_text + "[Retrieval]"
        elif output_text.endswith("[No Retrieval]"):
            index = output_text.rfind("[No Retrieval]")
            output_text = output_text[:index].replace("[No Retrieval]", "").replace("[Retrieval]", "")
            return output_text + "[No Retrieval]"
        else:
            return output_text.replace("[No Retrieval]", "").replace("[Retrieval]", "")


    def postprocess(self,pred):
        special_tokens = ["[Fully supported]", "[Partially supported]", "[No support / Contradictory]", "[No Retrieval]", "[Retrieval]",
                        "[Irrelevant]", "[Relevant]", "<paragraph>", "</paragraph>", "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]"]
        for item in special_tokens:
            pred = pred.replace(item, "")
        pred = pred.replace("</s>", "")

        if len(pred) == 0:
            return ""
        if pred[0] == " ":
            pred = pred[1:]
        return pred


    def load_jsonlines(self,file):
        with jsonlines.open(file, 'r') as jsonl_f:
            lst = [obj for obj in jsonl_f]
        return lst


    def load_file(self,input_fp):
        if input_fp.endswith(".json"):
            input_data = json.load(open(input_fp))
        else:
            input_data = load_jsonlines(input_fp)
        return input_data


    def save_file_jsonl(self,data, fp):
        with jsonlines.open(fp, mode='w') as writer:
            writer.write_all(data)


    def preprocess_input(self,input_data, task):
        if task == "factscore":
            for item in input_data:
                item["instruction"] = item["input"]
                item["output"] = [item["output"]
                                ] if "output" in item else [item["topic"]]
            return input_data

        elif task == "qa":
            for item in input_data:
                if "instruction" not in item:
                    item["instruction"] = item["question"]
                if "answers" not in item and "output" in item:
                    item["answers"] = "output"
            return input_data

        elif task in ["asqa", "eli5"]:
            processed_input_data = []
            for instance_idx, item in enumerate(input_data["data"]):
                prompt = item["question"]
                instructions = TASK_INST[task]
                prompt = instructions + "## Input:\n\n" + prompt
                entry = copy.deepcopy(item)
                entry["instruction"] = prompt
                processed_input_data.append(entry)
            return processed_input_data


    def postprocess_output(self,input_instance, prediction, task, intermediate_results=None):
        if task == "factscore":
            return {"input": input_instance["input"], "output": prediction, "topic": input_instance["topic"], "cat": input_instance["cat"]}

        elif task == "qa":
            input_instance["pred"] = prediction
            return input_instance

        elif task in ["asqa", "eli5"]:
            # ALCE datasets require additional postprocessing to compute citation accuracy.
            final_output = ""
            docs = []
            if "splitted_sentences" not in intermediate_results:
                input_instance["output"] = postprocess(prediction)

            else:
                for idx, (sent, doc) in enumerate(zip(intermediate_results["splitted_sentences"][0], intermediate_results["ctxs"][0])):
                    if len(sent) == 0:
                        continue
                    postprocessed_result = postprocess(sent)
                    final_output += postprocessed_result[:-
                                                        1] + " [{}]".format(idx) + ". "
                    docs.append(doc)
                if final_output[-1] == " ":
                    final_output = final_output[:-1]
                input_instance["output"] = final_output
            input_instance["docs"] = docs
            return input_instance

    def process_arc_instruction(self,item, instruction):
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
        choices = "\nA: {0}\nB: {1}\nC: {2}\nD: {3}".format(answer_labels["A"], answer_labels["B"], answer_labels["C"], answer_labels["D"])
        if "E" in answer_labels:
            choices += "\nE: {}".format(answer_labels["E"])
        processed_instruction = instruction + "\n\n### Input:\n" + item["instruction"] + choices
        return processed_instruction


    def postprocess_answers_closed(self,output, task, choices=None):
        final_output = None
        if choices is not None:
            for c in choices.split(" "):
                if c in output:
                    final_output = c
        if task == "fever" and output in ["REFUTES", "SUPPORTS"]:
            final_output = "true" if output == "SUPPORTS" else "REFUTES"
        if task == "fever" and output.lower() in ["true", "false"]:
            final_output = output.lower()
        if final_output is None:
            return output
        else:
            return final_output


    def run_step_generation_batch(self,model, prompt, paragraphs,  max_new_tokens,
                                rel_tokens=None, grd_tokens=None, ret_tokens=None, ut_tokens=None,
                                threshold=None, w_rel=1.0, w_sup=1.0, w_use=0.5, use_seqscore=False):
        
        if paragraphs is not None:
            aug_prompts = [prompt + "[Retrieval]" + "<paragraph>{}</paragraph>".format(
                paragraph["title"] + "\n" + paragraph["text"]) for paragraph in paragraphs]
        else:
            aug_prompts = [prompt]

        sampling_params = SamplingParams(
            temperature=0.0, top_p=1.0, max_tokens=max_new_tokens, logprobs=32000, )
        
        preds = model.generate(aug_prompts, sampling_params)


        # compute the scores for each generation
        relevance_score_dict = {}
        grd_score_dict = {}
        ut_score_dict = {}
        overall_scores = {}
        final_preds = []
        for p_idx, pred in enumerate(preds):
            pred_token_ids = pred.outputs[0].token_ids
            pred_text = pred.outputs[0].text
            pred_log_probs = pred.outputs[0].logprobs
            seq_score = pred.outputs[0].cumulative_logprob / \
                max(len(pred.outputs[0].token_ids), 1)
            assert len(pred_log_probs) == len(pred_token_ids)

            relevance_score_dict.setdefault(p_idx, {})
            grd_score_dict.setdefault(p_idx, {})
            ut_score_dict.setdefault(p_idx, {})
            # Compute reward scores
            for tok, id in rel_tokens.items():
                if id not in pred_log_probs[0]:
                    prob = -100
                else:
                    prob = np.exp(pred_log_probs[0][id])
                relevance_score_dict[p_idx][tok] = prob

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
                        grd_score_dict[p_idx][token] = np.exp(prob)

            utility_token_appear_indices = []
            if ut_tokens is not None:
                for tok_idx, tok in enumerate(pred_token_ids):
                    if tok in list(ut_tokens.values()):
                        utility_token_appear_indices.append(tok_idx)
                if len(utility_token_appear_indices) > 0:
                    idx = utility_token_appear_indices[0]
                    for token, token_id in grd_tokens.items():
                        prob = pred_log_probs[idx][token_id] if token_id in pred_log_probs[idx] else -100
                        ut_score_dict[p_idx][token] = np.exp(prob)

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
                utility_score = np.sum([ut_scores[i] * (ut_score_dict[p_idx]["[Utility:{}]".format(i+1)] / ut_sum)
                                    if "[Utility:{}]".format(i+1) in ut_score_dict[p_idx] else 0.0 for i in range(0, 5)])
            else:
                utility_score = 0.0

            if use_seqscore is True:
                final_score =np.exp(seq_score) + w_rel * relevance_score + \
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

            # print("-"*30+"pred_text"+"-"*30+"\n",pred_text)
            # print("-"*60)

            # modify and add do retrieve tokens
            if "[No Retrieval]" in pred_text:
                ret_token_appear_indices = []
                substrings = pred_text.split("[No Retrieval]")

                for tok_idx, tok in enumerate(pred_token_ids):
                    if tok == ret_tokens["[No Retrieval]"]:
                        ret_token_appear_indices.append(tok_idx)
                        # substrings
                        print("retrieval_tokens")

                ret_token_score_dict = {}
                retrieval_remap = {}
                for order, idx in enumerate(ret_token_appear_indices):
                    ret_token_score_dict.setdefault(order, {})
                    for tok, tok_id in ret_tokens.items():
                        prob = pred_log_probs[idx][tok_id] if tok_id in pred_log_probs[idx] else -100
                        ret_token_score_dict[order][tok] = np.exp(prob)
                    if ret_token_score_dict[order]["[Retrieval]"] + ret_token_score_dict[order]["[No Retrieval]"] != 0.0:
                        do_retrieve = (ret_token_score_dict[order]["[Retrieval]"] + ret_token_score_dict[order]["[Continue to Use Evidence]"]) / (
                            ret_token_score_dict[order]["[Retrieval]"] + ret_token_score_dict[order]["[No Retrieval]"]) > threshold
                    else:
                        do_retrieve = 0.0
                    if do_retrieve > threshold:
                        retrieval_remap[order] = True
                    else:
                        retrieval_remap[order] = False
                processed_pred = ""
                for substr_i, substring in enumerate(substrings):
                    if substr_i in retrieval_remap and retrieval_remap[substr_i] is True:
                        processed_pred += substring + "[Retrieval]"
                    else:
                        processed_pred += substring + "[No Retrieval]"
                pred_text = processed_pred
                final_preds.append(pred_text)
                # print("-"*30+" no retriever in final_preds"+"-"*30+"\n",pred_text)
                # print("-"*60)
            else:
                final_preds.append(pred_text)
                # print("-"*30+" no retriever not in final_preds"+"-"*30+"\n",pred_text)
                # print("-"*60)

        preds = final_preds
        scores = [overall_scores[p_idx]["final_score"] for p_idx in overall_scores]
        return preds, scores, overall_scores


    def run(self,input_data ):
        ret_tokens, rel_tokens, grd_tokens, ut_tokens = self.load_special_tokens(self.tokenizer)

        default_ctxs = input_data["ctxs"]
        default_ctxs = default_ctxs[0:5]
        
        do_retrieve_type = "always_retrieve"
        
        prompt = input_data["input"]
        processed_prompt = self.PROMPT_DICT["prompt_no_input"].format_map({"instruction": prompt})
        
        # Get token ids for reflection tokens.

        
        prompt = processed_prompt
        model = self.model
        max_new_tokens=300
        ctxs = default_ctxs
        max_depth=7
        threshold=0.2
        beam_width=2
        use_seqscore=False
        w_rel=1.0
        w_sup=1.0
        w_use=0.5
        ignore_cont=False
        
        sampling_params = SamplingParams(
            temperature=0.0, top_p=1, max_tokens=25, logprobs=32000)
        preds = model.generate([prompt], sampling_params)
        pred_log_probs = preds[0].outputs[0].logprobs
        preds = [pred.outputs[0].text.split("\n\n")[0] for pred in preds]
        if "[Retrieval]" not in preds[0]:
            do_retrieve = False
        else:
            if threshold is None:
                do_retrieve = False
            else:
                ret_token_score_dict = {}
                for tok, tok_id in ret_tokens.items():
                    prob = pred_log_probs[0][tok_id]
                    ret_token_score_dict[tok] = np.exp(prob)
                retrieve_prob = ret_token_score_dict["[Retrieval]"] / (
                    ret_token_score_dict["[Retrieval]"] + ret_token_score_dict["[No Retrieval]"])
                do_retrieve = True if retrieve_prob > threshold else False

        if do_retrieve_type == "always_retrieve":
            do_retrieve = True

        # do_retrieve = False
        if do_retrieve is False:
            print("--------------------------------do_retrieve is False-----------------------------------------")
            sampling_params = SamplingParams(
                temperature=0.0, top_p=1, max_tokens=max_new_tokens)
            prompt += "[No Retrieval]"
            preds = model.generate([prompt], sampling_params)
            
            print("-"*30+"preds  !!!"+"-"*30)
            print(preds)
            
            preds = [pred.outputs[0].text.split("\n\n")[0] for pred in preds]
            prediction_tree = {}
            finalResult = {
                "prompt":input_data["input"],
                "result": preds[0],
                "meta": [prediction_tree],
            }   
            return finalResult
        
        elif do_retrieve is True:
            print("--------------------------------do_retrieve is True-----------------------------------------")
            curr_depth = 1
            terminated = False
            node_id = 0
            prediction_tree = {}
            levels = {}
            prediction_tree[node_id] = {"prompt": prompt, "pred": "[Retrieval]",
                                        "processed_pred": "", "score": None, "ctx": None, "parent": None}
            levels[0] = [0]
            # 
            while curr_depth < max_depth:
                levels[curr_depth] = []
                if curr_depth-1 in levels and terminated is False:
                    for node in levels[curr_depth-1]:
                        print("-"*30+"node"+"-"*30+"\n",node)
                        print("-"*60)
                        pred = prediction_tree[node]["pred"]
                        print("-"*30+"pred"+"-"*30+"\n",pred)
                        print("-"*60)
                        if pred == "</s>":
                            terminated = True
                            continue
                        prompt = prediction_tree[node]["prompt"]
                        prev_generation = prediction_tree[node]["processed_pred"]
                        score = prediction_tree[node]["score"]
                        
                        if "[Retrieval]" in pred:
                            retrieval_results = {}
                            ctxs = default_ctxs
                            
                            preds, scores, overall_score_dict = self.run_step_generation_batch(
                                model, prompt + prev_generation, ctxs, max_new_tokens,
                                rel_tokens, ret_tokens=ret_tokens, grd_tokens=grd_tokens, ut_tokens=ut_tokens,
                                threshold=threshold, w_rel=w_rel, w_sup=w_sup, w_use=w_use)
                            
                            print(preds)
                            print("preds len => \n",len(preds))
                            
                            
                            for i, (pred, p_score) in enumerate(zip(preds, scores)):
                                retrieval_results[i] = {
                                    "pred": pred, "score": p_score}
                            for i, result in retrieval_results.items():
                                node_id += 1
                                node_score = result["score"] * \
                                    score if score is not None else result["score"]
                                pred = result["pred"]
                                prediction_tree[node_id] = {"prompt": prompt + prev_generation, "pred": pred,
                                                            "score": node_score, "ctx": ctxs[i], "parent": node,
                                                            "overall_score_dict": overall_score_dict}
                                if "[Retrieval]" in pred:
                                    gen_result_index = pred.index("[Retrieval]")
                                    prev_generation = pred[:gen_result_index]
                                else:
                                    prev_generation = pred
                                prediction_tree[node_id]["processed_pred"] = prev_generation
                                levels[curr_depth].append(node_id)

                    current_rank = levels[curr_depth]
                    print("-"*30+"current_rank"+"-"*30+"\n",current_rank)
                    
                    node2score = {node_id: prediction_tree[node_id]["score"] for node_id in current_rank}
                    print("-"*30+"node2score"+"-"*30+"\n",node2score)
                    top_nodes = sorted(node2score.items(), key=lambda x: x[1], reverse=True)[:beam_width]
                    print("-"*30+"top_nodes"+"-"*30+"\n",top_nodes)
                    levels[curr_depth] = [node[0] for node in top_nodes]
                    print("-"*30+"levels[curr_depth]"+"-"*30+"\n",levels[curr_depth])
                    
                    curr_depth += 1
                else:
                    break

        final_prediction = ""
        parent = 0
        best_selections = {}

        # Traverse from the bottom
        levels = {k: v for k, v in levels.items() if len(v) > 0 and k != 0}
        for path_i, node in enumerate(levels[len(levels)]):
            if node == 0:
                break
            best_selections[path_i] = [node]
            current_node = node
            current_level = curr_depth
            if current_node is None:
                continue
            while current_level > 0 and current_node is not None:
                parent = prediction_tree[current_node]["parent"]
                best_selections[path_i] = [parent] + best_selections[path_i]
                current_node = parent
                current_level += 1

        final_prediction = {}
        splitted_sentences = {}
        original_splitted_sentences = {}
        ctxs = {}
        for path_i, nodes in best_selections.items():
            final_prediction[path_i] = " ".join([prediction_tree[node]["pred"] for node in nodes if node is not None and (
                ignore_cont is False or (ignore_cont is True and "[No support / Contradictory]" not in prediction_tree[node]["pred"]))])
            splitted_sentences[path_i] = [prediction_tree[node]["pred"] for node in nodes if node is not None and (
                ignore_cont is False or (ignore_cont is True and "[No support / Contradictory]" not in prediction_tree[node]["pred"]))]
            original_splitted_sentences[path_i] = [prediction_tree[node]["pred"] for node in nodes if node is not None and (
                ignore_cont is False or (ignore_cont is True and "[No support / Contradictory]" not in prediction_tree[node]["pred"]))]
            ctxs[path_i] = [prediction_tree[node]["ctx"] for node in nodes if node is not None and (ignore_cont is False or (
                ignore_cont is True and "[No support / Contradictory]" not in prediction_tree[node]["pred"]))]

        print("final_prediction => \n",final_prediction)

        fresult = {"final_prediction": final_prediction,
                "splitted_sentences": splitted_sentences,
                "original_splitted_sentences": original_splitted_sentences,
                "best_selections": best_selections,
                "ctxs": ctxs,
                "prediction_tree": prediction_tree}
        
        postprocessed_result = self.fix_spacing(final_prediction[0])
        finalResult = {
            "prompt":input_data["input"],
            "result": postprocessed_result,
            "meta": [fresult],
        }   
        return finalResult


    def run_one_step(self,prompt,ctxs):
        
        input_node = {
            "prompt": prompt,
            "pred": "[Retrieval]",
            "processed_pred": "",
            "score": None,
            "ctxs": ctxs,
            "parent": None
        }
        
        ret_tokens, rel_tokens, grd_tokens, ut_tokens = self.load_special_tokens(self.tokenizer)

        default_ctxs = input_node["ctxs"]
        default_ctxs = default_ctxs[0:5]
        
        do_retrieve_type = "always_retrieve"
        
        prompt = input_node["prompt"]
        processed_prompt = self.PROMPT_DICT["prompt_no_input"].format_map({"instruction": prompt})
        
        # Get token ids for reflection tokens.

        
        prompt = processed_prompt
        model = self.model
        max_new_tokens=300
        ctxs = default_ctxs
        max_depth=7
        threshold=0.2
        # 
        beam_width=2
        w_rel=1.0
        w_sup=1.0
        w_use=0.5
        
        sampling_params = SamplingParams(
            temperature=0.0, top_p=1, max_tokens=25, logprobs=32000)
        preds = model.generate([prompt], sampling_params)
        pred_log_probs = preds[0].outputs[0].logprobs
        preds = [pred.outputs[0].text.split("\n\n")[0] for pred in preds]
        if "[Retrieval]" not in preds[0]:
            do_retrieve = False
        else:
            if threshold is None:
                do_retrieve = False
            else:
                ret_token_score_dict = {}
                for tok, tok_id in ret_tokens.items():
                    prob = pred_log_probs[0][tok_id]
                    ret_token_score_dict[tok] = np.exp(prob)
                retrieve_prob = ret_token_score_dict["[Retrieval]"] / (
                    ret_token_score_dict["[Retrieval]"] + ret_token_score_dict["[No Retrieval]"])
                do_retrieve = True if retrieve_prob > threshold else False

        if do_retrieve_type == "always_retrieve":
            do_retrieve = True

        # do_retrieve = False
        if do_retrieve is False:
            print("--------------------------------do_retrieve is False-----------------------------------------")
            sampling_params = SamplingParams(
                temperature=0.0, top_p=1, max_tokens=max_new_tokens)
            prompt += "[No Retrieval]"
            preds = model.generate([prompt], sampling_params)
            
            print("-"*30+"preds  !!!"+"-"*30)
            print(preds)
            
            preds = [pred.outputs[0].text.split("\n\n")[0] for pred in preds]
            prediction_tree = {}
            finalResult = {
                "prompt":input_node["prompt"],
                "result": preds[0],
                "node":None,
                "meta": None,
            }   
            return finalResult
        
        elif do_retrieve is True:
            prompt = input_node["prompt"]
            pred = input_node["pred"]
            prev_generation = input_node["processed_pred"]
            score = input_node["score"]
            print("--------------------------------do_retrieve is True-----------------------------------------")
            curr_depth = 1
            node_id = 0
            prediction_tree = {}
            tempNodeList = []
            top_1_node = None
            if "[Retrieval]" in pred:
                retrieval_results = {}
                ctxs = default_ctxs
                
                preds, scores, overall_score_dict = self.run_step_generation_batch(
                    model, prompt + prev_generation, ctxs, max_new_tokens,
                    rel_tokens, ret_tokens=ret_tokens, grd_tokens=grd_tokens, ut_tokens=ut_tokens,
                    threshold=threshold, w_rel=w_rel, w_sup=w_sup, w_use=w_use)
                
                for i, (pred, p_score) in enumerate(zip(preds, scores)):
                    retrieval_results[i] = {
                        "pred": pred, "score": p_score}
                    
                for i, result in retrieval_results.items():
                    node_id += 1
                    node_score = result["score"] * \
                        score if score is not None else result["score"]
                    pred = result["pred"]
                    prediction_tree[node_id] = {"prompt": prompt + prev_generation, "pred": pred,
                                                "score": node_score, "ctx": ctxs[i], "parent": 0,
                                                "overall_score_dict": overall_score_dict}
                    if "[Retrieval]" in pred:
                        gen_result_index = pred.index("[Retrieval]")
                        prev_generation = pred[:gen_result_index]
                    else:
                        prev_generation = pred
                    prediction_tree[node_id]["processed_pred"] = prev_generation
                    tempNodeList.append(node_id)
                current_rank = tempNodeList
                print("-"*30+"current_rank"+"-"*30+"\n",current_rank)
                node2score = {node_id: prediction_tree[node_id]["score"] for node_id in current_rank}
                print("-"*30+"node2score"+"-"*30+"\n",node2score)
                top_nodes = sorted(node2score.items(), key=lambda x: x[1], reverse=True)[:beam_width]
                print("-"*30+"top_nodes"+"-"*30+"\n",top_nodes)
                top1_nodeId = top_nodes[0][0]
                top_1_node = prediction_tree[top1_nodeId]
                curr_depth += 1

        
        print("-"*30+"top_1_node"+"-"*30+"\n",top_1_node)
        
        print("-"*30+"prediction_tree"+"-"*30+"\n",prediction_tree)
        postprocessed_result = self.fix_spacing(top_1_node["pred"])
        
        finalResult = {
            "prompt":input_node['prompt'],
            "result": postprocessed_result,
            "node":top_1_node,
            "meta": prediction_tree,
        }   
        return finalResult


app = Flask(__name__, static_url_path='')
app.config['JSON_AS_ASCII'] = False
CORS(app, resources={r"/*": {"origins": "*"}}, send_wildcard=True)

input_data = {
    "input": "Question: Tell me a bio of Jessie Mae Brown Beavers.",
    "output": "Jessie Mae Brown Beavers was an African American educator, civil rights activist, and community leader. She was born in 1910 in rural Mississippi and moved to Memphis, Tennessee in 1929. She was a teacher for 40 years and was active in the civil rights movement, working with the NAACP and the Memphis chapter of the National Council of Negro Women. She was also a founding member of the Memphis chapter of the Southern Christian Leadership Conference and was a leader in the fight for desegregation in Memphis. She was also a leader in the fight for economic justice and was a founding member of the Memphis chapter of the Southern Christian Leadership Conference. She was also a leader in the fight for economic justice and was a founding member of the Memphis chapter of the Southern Christian Leadership Conference. She was also a leader in the fight for economic justice and was a founding member of the Memphis chapter of the Southern Christian Leadership Conference. She was also a leader in the fight for economic justice and was a founding member of the Memphis chapter of the Southern Christian Leadership Conference. She was also a leader in the fight for economic justice and was a founding member of the Memphis chapter of the Southern Christian Leadership Conference. She was also a leader in the fight for economic justice and was a founding member of the Memphis chapter of the Southern Christian Leadership Conference. She was also a leader in the fight for economic justice and was a founding member of the Memphis chapter of the Southern Christian Leadership Conference. She was also a leader in the fight for economic justice and was a founding member of the Memphis chapter of the Southern Christian Leadership Conference. She was also a leader in the fight for economic justice and was a founding member of the Memphis chapter of the Southern Christian Leadership Conference. She was also a leader in the fight for economic justice and was a founding member of the Memphis chapter of the Southern Christian Leadership Conference. She was also a leader in the fight for economic justice and was a founding member of the Memphis chapter of the Southern Christian Leadership Conference. She was also a leader in the fight for economic justice and was a founding member of the Memphis chapter of the Southern Christian Leadership",
    "topic": "Jessie Mae Brown Beavers",
    "cat": [
        "very rare",
        "North America"
    ],
    "question": "Who is Jessie Mae Brown Beavers?",
    "answers": [
        "Jessie Mae Brown Beavers"
    ],
    "ctxs": [
        {
            "id": "7711613",
            "title": "Mae Beavers",
            "text": "Mae Beavers Mae Beavers (born December 11, 1947 in Millport, Alabama) is an American former politician. A Republican, she was a member of the Tennessee Senate for the 17th district from 2003 until she resigned to run for governor in August 2017. The 17th District is composed of Cannon, Clay, DeKalb, Macon, Smith, and Wilson counties. Prior to becoming a state senator, Beavers was a state representative in the 99th through the 102nd General Assemblies. She was an unsuccessful candidate for Governor of Tennessee in the 2018 Tennessee gubernatorial election. Mae Beavers was born on December 11, 1947, in Millport,",
            "score": "1.6195295"
        },
        {
            "id": "20393405",
            "title": "Mae Brown",
            "text": "Mae Brown Mae Brown (1935-1973) was the second deaf-blind woman and the first deaf-blind Canadian to earn a university degree. She graduated from the University of Toronto Scarborough in 1972. Brown was born in Thunder Bay in 1935. Her sight and hearing deteriorated throughout her childhood; by high school her vision had deteriorated to the point where she could not read a blackboard, and she dropped out. An operation performed on Brown later in her teens to remove a brain tumor led to the complete loss of her hearing as well. Brown registered with the Canadian National Institute for the",
            "score": "1.5686135"
        },
        {
            "id": "4113706",
            "title": "Rita Mae Brown",
            "text": "Rita Mae Brown Rita Mae Brown (born November 28, 1944) is an American writer, activist, and feminist. She is best known for her first novel \"Rubyfruit Jungle\". Brown is also a mystery writer and screenwriter. Brown was born in 1944 in Hanover, Pennsylvania to an unmarried, teenage mother and her mother's married boyfriend. Brown's birth mother left the newborn Brown at an orphanage. Brown's mother's cousin Julia \"Juts\" Brown and her husband Ralph retrieved her from the orphanage, and raised her as their own in York, Pennsylvania, and later in Ft. Lauderdale, Florida. Julia and Ralph Brown were active Republicans",
            "score": "1.48156"
        },
        {
            "id": "20142345",
            "title": "Jessie Trout",
            "text": "Owen Sound in the summer. Jessie Trout Jessie Trout (July 26, 1895 – 1990) was a Canadian missionary to Japan for nearly 20 years until she left Japan during World War II. She was a leader in the Disciples of Christ (Campbell Movement) and the first woman to serve as vice president of the United Christian Missionary Society. She was a member of the Disciples of Christ, an author, translator, and co-founder of the Christian Women's Fellowship (1950) and the International Christian Women's Fellowship (1953). She received an honorary doctor of divinity degree from Bethany College in 1955. Jessie Mary",
            "score": "1.4498292"
        },
        {
            "id": "4563484",
            "title": "Louise Beavers",
            "text": "Filmmakers Hall of Fame in 1976. Features: Short subjects: Louise Beavers Louise Beavers (March 8, 1902 – October 26, 1962) was an American film and television actress. Beavers appeared in dozens of films and two hit television shows from the 1920s until 1960, most often cast in the role of a maid, servant, or slave. She was a native of Cincinnati, Ohio, and a member of Sigma Gamma Rho sorority, one of the four African-American sororities. Beavers was a breakthrough actress for black women and became known as a symbol of a \"mammy\" on the screen. A mammy archetype \"is",
            "score": "1.4398854"
        },
        {
            "id": "5034991",
            "title": "Jessie Mae Hemphill",
            "text": "In 2003, Hemphill's protégé and collaborator, Olga Wilhelmine Munding, established the Jessie Mae Hemphill Foundation to preserve and archive the African-American music of northern Mississippi and to provide assistance for regional musicians in need who could not survive on meager publishing royalties. One of Hemphill's songs was featured in the dance \"Tales from the Creek,\" by Reggie Wilson's Fist and Heel Performance Group, in a series of events celebrating black culture in Union Square Park, Manhattan in 1998. Jessie Mae Hemphill Jessie Mae Hemphill (October 18, 1923 – July 22, 2006) was an American electric guitarist, songwriter, and vocalist specializing",
            "score": "1.4380236"
        },
        {
            "id": "17716831",
            "title": "Virginia Mae Brown",
            "text": "Virginia Mae Brown Virginia Mae Brown (November 13, 1923 – February 15, 1991) was an American civil servant, government official, and lawyer. Brown (1923–91) was born on November 13, 1923, in Pliny, West Virginia. She had a sister (Anna) that was a year older and a brother (Winston) that was two years younger than her in the Brown family. The U.S. Census shows the Brown family to be living in Buffalo, West Virginia, in 1930 and 1940 - just across the Kanawha River from Pliny (where she was born). The 1940 U.S. Census shows Brown to be in her third",
            "score": "1.4338607"
        },
        {
            "id": "8063346",
            "title": "Jessie Willcox Smith",
            "text": "Jessie Willcox Smith Jessie Willcox Smith (September 6, 1863 – May 3, 1935) was an American female illustrator during the Golden Age of American illustration. She was considered \"one of the greatest pure illustrators\". She was a contributor to books and magazines during the late 19th and early 20th centuries. Smith illustrated stories and articles for clients such as \"Century\", \"Collier's\", \"Leslie's Weekly\", \"Harper's\", \"McClure's\", \"Scribners\", and the \"Ladies' Home Journal\". She had an ongoing relationship with \"Good Housekeeping\", which included the long-running Mother Goose series of illustrations and also the creation of all of the \"Good Housekeeping\" covers from",
            "score": "1.4252641"
        },
        {
            "id": "4563474",
            "title": "Louise Beavers",
            "text": "Louise Beavers Louise Beavers (March 8, 1902 – October 26, 1962) was an American film and television actress. Beavers appeared in dozens of films and two hit television shows from the 1920s until 1960, most often cast in the role of a maid, servant, or slave. She was a native of Cincinnati, Ohio, and a member of Sigma Gamma Rho sorority, one of the four African-American sororities. Beavers was a breakthrough actress for black women and became known as a symbol of a \"mammy\" on the screen. A mammy archetype \"is the portrayal within a narrative framework or other imagery",
            "score": "1.4207637"
        },
        {
            "id": "5034986",
            "title": "Jessie Mae Hemphill",
            "text": "bars a few times in the 1950s, most of her playing was done in family and informal settings, such as picnics with fife-and-drum music, until she was recorded in 1979. Her first recordings were field recordings made by the blues researcher George Mitchell in 1967 and the ethnomusicologist David Evans in 1973, but they were not released. She was then known as Jessie Mae Brooks, using the surname from a brief early marriage. In 1978, Evans came to Memphis, Tennessee, to teach at Memphis State University (now the University of Memphis). The school founded the High Water Recording Company in",
            "score": "1.402215"
        },
        {
            "id": "19181844",
            "title": "Jessie Rosser",
            "text": "Jessie Rosser Jessie Rosser \"(born 1921;died 2013)\" was a Missionary of the Canadian Baptist Ministries who served in India for over 40 years and was Principal of the Eva Rose York Bible Training and Technical School for Women in Tuni, Andhra Pradesh. Jessie worked as a school teacher at St. Thomas, Ontario for some time and then studied social sciences at the McMaster University from where she graduated in 1947 with a B.A. and decided to serve the cause of people in difficult circumstances overseas. She came to India in 1947 and served as a Missionary in Kakinada, Vuyyuru, and",
            "score": "1.3973815"
        },
        {
            "id": "3090588",
            "title": "Jessie Redmon Fauset",
            "text": "and 1930s, exploring the lives of the black middle class. She also was the editor and co-author of the African-American children's magazine \"The Brownies' Book\". She is known for discovering and mentoring other African-American writers, including Langston Hughes, Jean Toomer, Countee Cullen, and Claude McKay. She was born Jessie Redmona Fauset (later known as Jessie Redmon Fauset) on April 27, 1882, in Fredericksville, Camden County, Snow Hill Center Township, New Jersey. The town is now known as Lawnside, New Jersey. She was the seventh child of Redmon Fauset, an African Methodist Episcopal minister, and Annie (née Seamon) Fauset. Jessie's mother",
            "score": "1.392673"
        },
        {
            "id": "19181846",
            "title": "Jessie Rosser",
            "text": "in Tuni. Jessie Rosser Jessie Rosser \"(born 1921;died 2013)\" was a Missionary of the Canadian Baptist Ministries who served in India for over 40 years and was Principal of the Eva Rose York Bible Training and Technical School for Women in Tuni, Andhra Pradesh. Jessie worked as a school teacher at St. Thomas, Ontario for some time and then studied social sciences at the McMaster University from where she graduated in 1947 with a B.A. and decided to serve the cause of people in difficult circumstances overseas. She came to India in 1947 and served as a Missionary in Kakinada,",
            "score": "1.3898633"
        },
        {
            "id": "17536723",
            "title": "Jessie Brown Pounds",
            "text": "Jessie Brown Pounds Jessie Hunter Brown Pounds (August 31, 1861 – 1921) was an American lyricist of gospel songs. Jessie Hunter Brown was born into a farm family in the village of Hiram, Portage County. A staff writer for \"Christian Standard\", she often collaborated with composer Frederick A. Fillmore (1856–1925). In 1897 she married John E. Pounds, minister of the Central Christian Church in Indianapolis, IN. As a college-educated, frontier woman, she's considered by some to be part of the \"first generation\" of \"New Women.\" Her parents were Holland Brown and Jane Abel Brown. Holland Brown was baptized after hearing",
            "score": "1.3853728"
        },
        {
            "id": "8634517",
            "title": "Jessie Hill",
            "text": "infection while on tour in Tokyo on May 4, 2015. Jessie Hill Jessie Hill (December 9, 1932 – September 17, 1996) was an American R&B and Louisiana blues singer and songwriter, best remembered for the classic song \"Ooh Poo Pah Doo\". Hill was born in New Orleans, Louisiana, United States. By his teens he was playing drums in local bands, and in 1951 he formed his own group, the House Rockers. After periods performing as drummer with Professor Longhair and then Huey \"Piano\" Smith, Hill formed a new version of the House Rockers in 1958, which enabled him to focus",
            "score": "1.3831363"
        },
        {
            "id": "20142338",
            "title": "Jessie Trout",
            "text": "Jessie Trout Jessie Trout (July 26, 1895 – 1990) was a Canadian missionary to Japan for nearly 20 years until she left Japan during World War II. She was a leader in the Disciples of Christ (Campbell Movement) and the first woman to serve as vice president of the United Christian Missionary Society. She was a member of the Disciples of Christ, an author, translator, and co-founder of the Christian Women's Fellowship (1950) and the International Christian Women's Fellowship (1953). She received an honorary doctor of divinity degree from Bethany College in 1955. Jessie Mary Trout was born to Archibald",
            "score": "1.3827628"
        },
        {
            "id": "5034985",
            "title": "Jessie Mae Hemphill",
            "text": "Jessie Mae Hemphill Jessie Mae Hemphill (October 18, 1923 – July 22, 2006) was an American electric guitarist, songwriter, and vocalist specializing in the North Mississippi hill country blues traditions of her family and regional heritage. Hemphill was born near Como and Senatobia, Mississippi, in the northern Mississippi hill country, just east of the Mississippi Delta. She began playing the guitar at the age of seven. She also played drums in local fife-and-drum bands, beginning with the band led by her paternal grandfather, Sid Hemphill, in which she played snare drum and bass drum. Aside from sitting in at Memphis",
            "score": "1.3797479"
        },
        {
            "id": "7711619",
            "title": "Mae Beavers",
            "text": "resign her spot in the state senate to focus fully on her campaign. Mark Pody won a special election to assume Beavers' senate seat. On January 30, 2018, Beavers announced that she would be stepping out of the 2018 Tennessee gubernatorial race. In March 2018, Beavers announced her candidacy in the Wilson County mayoral election. Beavers is married to Jerry Beavers, with whom she has two children. They attend Trevecca Community Church. Mae Beavers Mae Beavers (born December 11, 1947 in Millport, Alabama) is an American former politician. A Republican, she was a member of the Tennessee Senate for the",
            "score": "1.3781004"
        },
        {
            "id": "13237049",
            "title": "Irene Bennett Brown",
            "text": "of Women Writing the West. She continues to live in Oregon with her husband, Bob. Irene Bennett Brown Irene Bennett Brown is an American author of children's, young adult and adult fiction. Brown was born in Topeka, Kansas and when she was nine years old, moved with her family from Kansas to the Willamette Valley in Oregon. Brown's fourth book, \"To Rainbow Valley\", became the first one to sell and be published in 1969. It was re-released as an Easy Reader book in 2001. Brown has her own publishing company, Riveredge Books, which has published and re-issued several of her",
            "score": "1.3779411"
        },
        {
            "id": "20865962",
            "title": "Jessie Saulteaux",
            "text": "Jessie Saulteaux Jessie Prettyshield Saulteaux (1912 - 1995) was a Canadian Assiniboine elder and theological leader. Early in life, Saulteaux desired to become a nurse, but she was unable to do so due to reasons of race. Instead she turned her talents towards helping her community, the Carry-the-Kettle First Nation, and her church. She was among the first women in Saskatchewan to be elected tribal chief; she also supported the development of ministers and church leaders from the First Nations community. She participated in the founding of the All Native Circle Conference in the United Church of Canada, and the",
            "score": "1.3749332"
        },
        {
            "id": "4563475",
            "title": "Louise Beavers",
            "text": "of a black domestic servant, generally good-natured, often overweight, and loud\". Beavers was born in Cincinnati, Ohio, to school teacher Ernestine Monroe Beavers and William M. Beavers, who was originally from Georgia. Due to her mother's illness, Louise and her parents moved to Pasadena, California. In Pasadena, she attended school and engaged in several after-school activities, such as basketball and church choir. Her mother also worked as a voice teacher and taught Louise how to sing for concerts. In June 1920, she graduated from Pasadena High School. She worked as a dressing room attendant for a photographer and served as",
            "score": "1.3708005"
        },
        {
            "id": "17536724",
            "title": "Jessie Brown Pounds",
            "text": "Walter Scott preach; and the couple were abolitionists. Her parents hosted pioneers and luminaries including James A. Garfield. \"Her pen produced upwards of eight hundred hymns, eighty short stories, seven novels, lyrics, and scripts for cantatas, and numerous brief essays and non-fiction articles.\" \"Anywhere with Jesus\" is possibly the most well-known of her poems. Some of her poems have been set to a number of musical scores, the most familiar being the tune \"Serenity\" by Daniel B. Towner (1850–1919). Her 1896 poem \"Beautiful Isle\" became the song \"Beautiful Isle of Somewhere\", which was sung at President McKinley's funeral and criticized",
            "score": "1.3702077"
        },
        {
            "id": "2216611",
            "title": "Benny Beaver",
            "text": "of regents, Bell became hugely popular among the students for his ritual of marching to the Marys River after each of Oregon State's Civil War victories. He was said to have tossed his top hat into the water as a token of celebration. Earlier mascots include \"Jimmie\" the Coyote (1892–1893) and \"Bulldog\" (1906–1910, unofficial and for specific teams only, such as the Wrestling squad). The beaver mascot's name, \"Benny,\" was officially adopted in 1945. Two failed attempts to maintain a live beaver mascot include Bevo Beaver (rescued from Mary's River in 1921 and later stolen ) and Billy Beaver (made",
            "score": "1.3672736"
        },
        {
            "id": "3980421",
            "title": "June Brown",
            "text": "career, she played the roles of Hedda Gabler and Lady Macbeth. In 2009, Brown played Jessie in the West End production of \"Calendar Girls\" at the Noël Coward Theatre. Also in the play were former \"EastEnders\" stars Anita Dobson (Angie Watts), Jack Ryder (Jamie Mitchell) and Jill Halfpenny (Kate Mitchell). June Brown June Muriel Brown, (born 16 February 1927) is an English actress, known for her role as Dot Cotton in the BBC soap opera \"EastEnders\" from 1985 onwards. In 2005, she won Best Actress at the Inside Soap Awards, and in the same year, also received the Lifetime Achievement",
            "score": "1.3651499"
        },
        {
            "id": "19998679",
            "title": "Jessie Isabelle Price",
            "text": "she began working on vaccine development for \"Pasteurella anatipestifer\" for white pekin ducks, which she would continue in avian cholera and TB for various species through her career. Some of the vaccines were commercially developed. She worked with national and international colleagues, publishing on \"Pasteurella anatipestifer\" in pheasants, medication for bacterial infections in ducklings, \"Pasteurella multocida\" in Nebraska wetlands and in snow geese. There is an extensive photoessay publicly available in \"Ebony\" magazine. See also a photo of her in later life in an obituary. Jessie Isabelle Price Jessie Isabelle Price (1930-2015) was a veterinary microbiologist. She isolated and reproduced",
            "score": "1.3603673"
        }
    ]
}

model_run = LLM(model="selfrag_llama2_7b")

# post
@app.route('/chat', methods=["POST"])
def chat():
    jsondata = request.get_json()
    aug_prompts = jsondata['aug_prompts']
    temperature = jsondata['temperature']
    top_p = jsondata['top_p']
    max_tokens = jsondata['max_tokens']
    if "logprobs" in jsondata:
        logprobs = jsondata['logprobs']
        result = model_run.generate(aug_prompts,SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, logprobs=logprobs))
    else:
        result = model_run.generate(aug_prompts,SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens))
    
    print("-"*30+"type result"+"-"*30+"\n",type(result))
    print("-"*30+"result"+"-"*30+"\n",result)
    
    return jsonify({"data": result, "code": 200})

@app.route('/chat_all', methods=["POST"])
def chat_all():
    jsondata = request.get_json()

    result = model_run.run(input_data=input_data)
    return jsonify({"data": result, "code": 200})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10030, threaded=True)
