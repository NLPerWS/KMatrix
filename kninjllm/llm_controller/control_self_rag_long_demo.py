import argparse
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
from vllm import SamplingParams

from kninjllm.llm_generator.base_generator.self_rag.retrieval_lm.utils import TASK_INST, PROMPT_DICT, load_special_tokens, load_jsonlines, postprocess, fix_spacing
from kninjllm.llm_common.component import component
from kninjllm.llm_generator.base_generator.self_rag.self_rag_generator import RagGenerator
from kninjllm.llm_retriever.contriever.Contriever_retriever import Contriever_Retriever
from root_config import RootConfig


@component
class SelfRagLongDemoController:
    def __init__(self,variableDataPath=""):
        self.top_k = 3
        generator=RagGenerator(model_path=RootConfig.selfRAG_model_path,executeType="infer")
        retriever=Contriever_Retriever(top_k=self.top_k,executeType="infer",model_path=RootConfig.contriever_model_path)
        self.contriever = retriever
        self.generator = generator
        self.tokenizer = self.generator.tokenizer
        self.variableDataPath = variableDataPath
        self.logSaver = RootConfig.logSaver
        self.step_count = 0
        self.prediction_tree = {}
        self.node_id = 0
        self.terminated = False
        self.levels = {0:[0],1:[0],2:[0],3:[0],4:[0]}
        self.curr_depth = 5

    @component.output_types(final_result=Dict[str,Any])
    def run(self,knowledge_info:Dict[str,Any],query_obj:Dict[str,Any] = {}):
        if knowledge_info != {}:
            self.contriever.run(knowledge_info = knowledge_info)
        if query_obj == {}:
            return {"final_result":{}} 
        parser = argparse.ArgumentParser()
        args = parser.parse_args()
        args.device = "cuda"
        args.ndocs = 10
        args.world_size = 1
        args.dtype = "half"
        args.use_grounding = True
        args.use_utility = True
        args.use_seqscore = True
        args.max_new_tokens=300
        args.max_depth=7
        args.beam_width=1
        args.w_rel=1.0
        args.w_sup=1.0
        args.w_use=0.5
        args.ignore_cont = False
        args.threshold=0.2
        # args.mode = "always_retrieve"
        args.mode = "adaptive_retrieval"
        args.model = self.generator
        args.retriever = self.contriever
        args.tokenizer = self.tokenizer
        args.logSaver = self.logSaver
        args.input_data = [query_obj]
        # res = self.main(args)[0]
        model = args.model
        tokenizer = args.tokenizer
        input_data = args.input_data
        retriever = args.retriever
        logSaver = args.logSaver
        # Get token ids for reflection tokens.
        ret_tokens, rel_tokens, grd_tokens, ut_tokens = load_special_tokens(
            tokenizer, use_grounding=args.use_grounding, use_utility=args.use_utility)

        new_results = []
        for idx, item in enumerate(input_data):
            prompt = item["question"]
            processed_prompt = PROMPT_DICT["prompt_no_input"].format_map(
                {"instruction": prompt})
            result, intermediate = self.call_model_beam_batch(prompt=processed_prompt, model=model, max_new_tokens=args.max_new_tokens, retriever=retriever, query=prompt,
                                        rel_tokens=rel_tokens, ret_tokens=ret_tokens, grd_tokens=grd_tokens, ut_tokens=ut_tokens,
                                        use_seqscore=args.use_seqscore, threshold=args.threshold, 
                                        beam_width=args.beam_width, max_depth=args.max_depth,
                                        w_rel=1.0, w_sup=1.0, w_use=0.5, mode=args.mode, ignore_cont=args.ignore_cont,logSaver=logSaver )  
            
            postprocessed_result = result[0]
            new_results.append({"question": prompt, "content": postprocessed_result})
        return {"final_result":new_results[0]}
    
    def call_model_beam_batch(self,prompt, model, max_new_tokens=15, retriever=None, query=None, max_depth=5, rel_tokens=None,
                            grd_tokens=None, ret_tokens=None, threshold=None, beam_width=2, ut_tokens=None, use_seqscore=False,
                            w_rel=1.0, w_sup=1.0, w_use=0.5, ignore_cont=False, mode="adaptive_retrieval",logSaver=None):
        special_tokens = []
        if "## Input:\n\n" in query:
            query = query.split("## Input:\n\n")[1]
        if rel_tokens is not None:
            special_tokens = list(rel_tokens.keys())
        if ret_tokens is not None:
            special_tokens += list(ret_tokens.keys())

        if mode == "no_retrieval":
            sampling_params = SamplingParams(
                temperature=0.0, top_p=1, max_tokens=max_new_tokens,skip_special_tokens=False)
            prompt += "[No Retrieval]"
            # preds = model.generate([prompt], sampling_params)
            preds = []
            for one_prompt in [prompt]:
                pred = model.run(query_obj={"question":one_prompt}, sampling_params=sampling_params)['final_result']['meta']['pred']
                preds.append(pred)
            
            
            preds = [pred['outputs'][0]['text'].split("\n\n")[0] for pred in preds]
            return {0:preds[0]}, prediction_tree

        do_retrieve = False
        if mode == "always_retrieve":
            do_retrieve = True

        else:
            sampling_params = SamplingParams(
                temperature=0.0, top_p=1, max_tokens=25, logprobs=32000,skip_special_tokens=False)
            # preds = model.generate([prompt], sampling_params)
            preds = []
            for one_prompt in [prompt]:
                # pred = model.run(query_obj={"question":one_prompt}, sampling_params=sampling_params)['final_result']['meta']['pred']
                pred = model.run(query_obj={"question":one_prompt}, sampling_params=sampling_params,saveLogFlag=False)['final_result']['meta']['pred']
                preds.append(pred)
            
            pred_log_probs = preds[0]['outputs'][0]['logprobs']
            preds = [pred['outputs'][0]['text'].split("\n\n")[0] for pred in preds]
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


        if do_retrieve is False:
            
            sampling_params = SamplingParams(
                temperature=0.0, top_p=1, max_tokens=max_new_tokens,skip_special_tokens=False)
            prompt += "[No Retrieval]"
            # preds = model.generate([prompt], sampling_params)
            preds = []
            for one_prompt in [prompt]:
                pred = model.run(query_obj={"question":one_prompt}, sampling_params=sampling_params)['final_result']['meta']['pred']
                preds.append(pred)
            
            preds = [pred['outputs'][0]['text'].split("\n\n")[0] for pred in preds]
            prediction_tree = {}
            return {0:preds[0]}, prediction_tree
        
        
        elif do_retrieve is True:
            curr_depth = 1
            terminated = False
            node_id = 0
            prediction_tree = {}
            levels = {}
            prediction_tree[node_id] = {"prompt": prompt, "pred": "[Retrieval]",
                                        "processed_pred": "", "score": None, "ctx": None, "parent": None}
            levels[0] = [0]
            while curr_depth < max_depth:
                levels[curr_depth] = []
                if curr_depth-1 in levels and terminated is False:
                    for node in levels[curr_depth-1]:
                        pred = prediction_tree[node]["pred"]
                        if pred == "</s>":
                            terminated = True
                            continue
                        prompt = prediction_tree[node]["prompt"]
                        prev_generation = prediction_tree[node]["processed_pred"]
                        score = prediction_tree[node]["score"]
                        if "[Retrieval]" in pred:
                            retrieval_results = {}
                            
                            ctxs = retriever.run(query_obj={"question": prompt + fix_spacing(postprocess(prev_generation))})['final_result'][0]
                            
                            preds, scores, overall_score_dict = self.run_step_generation_batch(
                                model, prompt + prev_generation, ctxs, max_new_tokens,
                                rel_tokens, ret_tokens=ret_tokens, grd_tokens=grd_tokens, ut_tokens=ut_tokens,
                                threshold=threshold, w_rel=w_rel, w_sup=w_sup, w_use=w_use,logSaver=logSaver)
                            
                            
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
                    node2score = {
                        node_id: prediction_tree[node_id]["score"] for node_id in current_rank}
                    top_nodes = sorted(node2score.items(), key=lambda x: x[1], reverse=True)[
                        :beam_width]
                    levels[curr_depth] = [node[0] for node in top_nodes]
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
            
            splitted_sentences[path_i] = [prediction_tree[node]["processed_pred"] for node in nodes if node is not None and (
                ignore_cont is False or (ignore_cont is True and "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]))]
            original_splitted_sentences[path_i] = [prediction_tree[node]["pred"] for node in nodes if node is not None and (
                ignore_cont is False or (ignore_cont is True and "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]))]
            ctxs[path_i] = [prediction_tree[node]["ctx"] for node in nodes if node is not None and (ignore_cont is False or (
                ignore_cont is True and "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]))]

        result = {"final_prediction": final_prediction,
                "splitted_sentences": splitted_sentences,
                "original_splitted_sentences": original_splitted_sentences,
                "best_selections": best_selections,
                "ctxs": ctxs,
                "prediction_tree": prediction_tree}

        return final_prediction, result

    def run_step_generation_batch(self,model, prompt, paragraphs,  max_new_tokens,
                                rel_tokens=None, grd_tokens=None, ret_tokens=None, ut_tokens=None,
                                threshold=None, w_rel=1.0, w_sup=1.0, w_use=0.5, use_seqscore=False,logSaver=None):
        if paragraphs is not None:
            aug_prompts = [prompt + "[Retrieval]" + "<paragraph>{}</paragraph>".format(
                paragraph["title"] + "\n" + paragraph["content"]) for paragraph in paragraphs]
        else:
            aug_prompts = [prompt]

        sampling_params = SamplingParams(
            temperature=0.0, top_p=1.0, max_tokens=max_new_tokens, logprobs=32000,skip_special_tokens=False )
        # preds = model.generate(aug_prompts, sampling_params)
        
        preds = []
        for one_prompt in aug_prompts:
            # pred = model.run(query_obj={"question":one_prompt}, sampling_params=sampling_params)['final_result']['meta']['pred']
            pred = model.run(query_obj={"question":one_prompt}, sampling_params=sampling_params,saveLogFlag=False)['final_result']['meta']['pred']
            preds.append(pred)
        

        # compute the scores for each generation
        relevance_score_dict = {}
        grd_score_dict = {}
        ut_score_dict = {}
        overall_scores = {}
        final_preds = []
        for p_idx, pred in enumerate(preds):
            pred_token_ids = pred['outputs'][0]['token_ids']
            pred_text = pred['outputs'][0]['text']
            pred_log_probs = pred['outputs'][0]['logprobs']
            seq_score = pred['outputs'][0]['cumulative_logprob'] / \
                max(len(pred['outputs'][0]['token_ids']), 1)
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

            # modify and add do retrieve tokens
            if "[No Retrieval]" in pred_text:
                ret_token_appear_indices = []
                substrings = pred_text.split("[No Retrieval]")

                for tok_idx, tok in enumerate(pred_token_ids):
                    if tok == ret_tokens["[No Retrieval]"]:
                        ret_token_appear_indices.append(tok_idx)
                        substrings
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
            else:
                final_preds.append(pred_text)

        preds = final_preds
        scores = [overall_scores[p_idx]["final_score"] for p_idx in overall_scores]
        
        maxScore = max(scores)
        for score,prompt,pred in zip(scores,aug_prompts,preds):
            if score == maxScore:
                if logSaver is not None:
                    logSaver.writeStrToLog("Function -> RagGenerator -> run")
                    logSaver.writeStrToLog("Given generator prompt -> : " + prompt)
                    logSaver.writeStrToLog("Returns generator reply : " + pred)
        
        return preds, scores, overall_scores


