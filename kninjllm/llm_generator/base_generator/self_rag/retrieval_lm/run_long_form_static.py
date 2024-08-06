import os

import argparse
import jsonlines
from transformers import AutoTokenizer
import numpy as np
import json
import argparse
from vllm import LLM, SamplingParams
from utils import TASK_INST, PROMPT_DICT, load_special_tokens, load_jsonlines, postprocess, fix_spacing


def run_step_generation_batch(model, prompt, paragraphs,  max_new_tokens,
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
    
    print("-"*30+"preds"+"-"*30+"\n",preds)
    print("-"*30+"scores"+"-"*30+"\n",scores)
    print("-"*30+"overall_scores"+"-"*30+"\n",overall_scores)
    
    
    
    return preds, scores, overall_scores

def call_model_beam_batch(prompt, model, max_new_tokens=15, ctxs=None, query=None, max_depth=5, rel_tokens=None,
                          grd_tokens=None, ret_tokens=None, threshold=None, beam_width=2, ut_tokens=None, use_seqscore=False,
                          w_rel=1.0, w_sup=1.0, w_use=0.5, ignore_cont=False, mode="adaptive_retrieval"):
    special_tokens = []
    if "## Input:\n\n" in query:
        query = query.split("## Input:\n\n")[1]
    if rel_tokens is not None:
        special_tokens = list(rel_tokens.keys())
    if ret_tokens is not None:
        special_tokens += list(ret_tokens.keys())

    if mode == "no_retrieval":
        print("--------------------------------no_retrieval-----------------------------------------")
        sampling_params = SamplingParams(
            temperature=0.0, top_p=1, max_tokens=max_new_tokens)
        prompt += "[No Retrieval]"
        preds = model.generate([prompt], sampling_params)
        preds = [pred.outputs[0].text.split("\n\n")[0] for pred in preds]
        return preds[0], prediction_tree

    do_retrieve = False
    if mode == "always_retrieve":
        do_retrieve = True

    else:
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

    if do_retrieve is False:
        print("--------------------------------do_retrieve is False-----------------------------------------")
        sampling_params = SamplingParams(
            temperature=0.0, top_p=1, max_tokens=max_new_tokens)
        prompt += "[No Retrieval]"
        preds = model.generate([prompt], sampling_params)
        preds = [pred.outputs[0].text.split("\n\n")[0] for pred in preds]
        prediction_tree = {}
        return preds[0], prediction_tree
    
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
                        preds, scores, overall_score_dict = run_step_generation_batch(
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
                node2score = {node_id: prediction_tree[node_id]["score"] for node_id in current_rank}
                top_nodes = sorted(node2score.items(), key=lambda x: x[1], reverse=True)[:beam_width]
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
        final_prediction[path_i] = " ".join([prediction_tree[node]["processed_pred"] for node in nodes if node is not None and (
            ignore_cont is False or (ignore_cont is True and "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]))])
        splitted_sentences[path_i] = [prediction_tree[node]["processed_pred"] for node in nodes if node is not None and (
            ignore_cont is False or (ignore_cont is True and "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]))]
        original_splitted_sentences[path_i] = [prediction_tree[node]["pred"] for node in nodes if node is not None and (
            ignore_cont is False or (ignore_cont is True and "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]))]
        ctxs[path_i] = [prediction_tree[node]["ctx"] for node in nodes if node is not None and (ignore_cont is False or (
            ignore_cont is True and "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]))]

    fresult = {"final_prediction": final_prediction,
              "splitted_sentences": splitted_sentences,
              "original_splitted_sentences": original_splitted_sentences,
              "best_selections": best_selections,
              "ctxs": ctxs,
              "prediction_tree": prediction_tree}

   
    return final_prediction, fresult

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--task', type=str)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--max_new_tokens', type=int, default=15)
    parser.add_argument('--tokenizer_path', type=str)
    parser.add_argument('--download_dir', type=str, help="specify vllm model download dir",
                        default=".cache")
    parser.add_argument("--ndocs", type=int, default=10,
                        help="Number of documents to retrieve per questions")
    parser.add_argument("--world_size",  type=int, default=1,
                        help="world size to use multiple GPUs.")
    parser.add_argument("--dtype",  type=str, default="half",
                        help="We use bfloat16 for training. If you run inference on GPUs that do not support BF16, please set this to be `half`.")
    # Decoding hyperparams
    parser.add_argument('--threshold', type=float,
                        default=None, help="Adaptive threshold.")
    parser.add_argument("--use_grounding", action="store_true",
                        help="use ground score")
    parser.add_argument("--use_seqscore", action="store_true", help="use sequence scores.")
    parser.add_argument(
        "--use_utility", action="store_true", help="tree search")
    parser.add_argument("--beam_width",  type=int,
                        default=2, help="beam search width")
    parser.add_argument("--max_depth",  type=int,
                        default=2, help="tree depth width")
    parser.add_argument("--w_rel",  type=float, default=1.0,
                        help="reward weight for document relevance")
    parser.add_argument("--w_sup",  type=float, default=1.0,
                        help="reward weight for generation support (attribution)")
    parser.add_argument("--w_use",  type=float, default=1.0,
                        help="reward weight for overall completeness / utility.")
    parser.add_argument("--ignore_cont", action="store_true",
                        help="filter out sentences that include [No support / Contradictory] ")
    parser.add_argument('--mode', type=str, help="mode to control retrieval.",
                        default="default", choices=['adaptive_retrieval', 'no_retrieval', 'always_retrieve'],)

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, padding_side="left")

    # Get token ids for reflection tokens.
    ret_tokens, rel_tokens, grd_tokens, ut_tokens = load_special_tokens(
        tokenizer, use_grounding=args.use_grounding, use_utility=args.use_utility)

    if args.world_size is not None:
        model = LLM(model=args.model_name, download_dir=args.download_dir,
                    dtype=args.dtype, tensor_parallel_size=args.world_size,)

    else:
        model = LLM(model=args.model_name,
                    download_dir=args.download_dir, dtype=args.dtype)

    def generate(prompt, ctxs, max_new_tokens):
        processed_prompt = PROMPT_DICT["prompt_no_input"].format_map({"instruction": prompt})
        final_prediction, result = call_model_beam_batch(processed_prompt, model=model, max_new_tokens=max_new_tokens, ctxs=ctxs, query=prompt,
                                    rel_tokens=rel_tokens, ret_tokens=ret_tokens, grd_tokens=grd_tokens, ut_tokens=ut_tokens,
                                    use_seqscore=args.use_seqscore, threshold=args.threshold, 
                                    beam_width=args.beam_width, max_depth=args.max_depth,
                                    w_rel=1.0, w_sup=1.0, w_use=0.5, mode=args.mode, ignore_cont=args.ignore_cont, )
        return final_prediction, result

    input_path = args.input_file
    if input_path.endswith(".json"):
        input_data = json.load(open(input_path))
    else:
        input_data = load_jsonlines(input_path)

    # input_data = input_data[0:1]

    # input_data = [{
    #     "input": "Question: Tell me a bio of Jessie Mae Brown Beavers.",
    #     "output": "Jessie Mae Brown Beavers was an African American educator, civil rights activist, and community leader. She was born in 1910 in rural Mississippi and moved to Memphis, Tennessee in 1929. She was a teacher for 40 years and was active in the civil rights movement, working with the NAACP and the Memphis chapter of the National Council of Negro Women. She was also a founding member of the Memphis chapter of the Southern Christian Leadership Conference and was a leader in the fight for desegregation in Memphis. She was also a leader in the fight for economic justice and was a founding member of the Memphis chapter of the Southern Christian Leadership Conference. She was also a leader in the fight for economic justice and was a founding member of the Memphis chapter of the Southern Christian Leadership Conference. She was also a leader in the fight for economic justice and was a founding member of the Memphis chapter of the Southern Christian Leadership Conference. She was also a leader in the fight for economic justice and was a founding member of the Memphis chapter of the Southern Christian Leadership Conference. She was also a leader in the fight for economic justice and was a founding member of the Memphis chapter of the Southern Christian Leadership Conference. She was also a leader in the fight for economic justice and was a founding member of the Memphis chapter of the Southern Christian Leadership Conference. She was also a leader in the fight for economic justice and was a founding member of the Memphis chapter of the Southern Christian Leadership Conference. She was also a leader in the fight for economic justice and was a founding member of the Memphis chapter of the Southern Christian Leadership Conference. She was also a leader in the fight for economic justice and was a founding member of the Memphis chapter of the Southern Christian Leadership Conference. She was also a leader in the fight for economic justice and was a founding member of the Memphis chapter of the Southern Christian Leadership Conference. She was also a leader in the fight for economic justice and was a founding member of the Memphis chapter of the Southern Christian Leadership Conference. She was also a leader in the fight for economic justice and was a founding member of the Memphis chapter of the Southern Christian Leadership",
    #     "topic": "Jessie Mae Brown Beavers",
    #     "cat": [
    #         "very rare",
    #         "North America"
    #     ],
    #     "question": "Who is Jessie Mae Brown Beavers?",
    #     "answers": [
    #         "Jessie Mae Brown Beavers"
    #     ],
    #     "ctxs": [
    #         {
    #             "id": "7711613",
    #             "title": "Mae Beavers",
    #             "text": "Mae Beavers Mae Beavers (born December 11, 1947 in Millport, Alabama) is an American former politician. A Republican, she was a member of the Tennessee Senate for the 17th district from 2003 until she resigned to run for governor in August 2017. The 17th District is composed of Cannon, Clay, DeKalb, Macon, Smith, and Wilson counties. Prior to becoming a state senator, Beavers was a state representative in the 99th through the 102nd General Assemblies. She was an unsuccessful candidate for Governor of Tennessee in the 2018 Tennessee gubernatorial election. Mae Beavers was born on December 11, 1947, in Millport,",
    #             "score": "1.6195295"
    #         },
    #         {
    #             "id": "20393405",
    #             "title": "Mae Brown",
    #             "text": "Mae Brown Mae Brown (1935-1973) was the second deaf-blind woman and the first deaf-blind Canadian to earn a university degree. She graduated from the University of Toronto Scarborough in 1972. Brown was born in Thunder Bay in 1935. Her sight and hearing deteriorated throughout her childhood; by high school her vision had deteriorated to the point where she could not read a blackboard, and she dropped out. An operation performed on Brown later in her teens to remove a brain tumor led to the complete loss of her hearing as well. Brown registered with the Canadian National Institute for the",
    #             "score": "1.5686135"
    #         },
    #         {
    #             "id": "4113706",
    #             "title": "Rita Mae Brown",
    #             "text": "Rita Mae Brown Rita Mae Brown (born November 28, 1944) is an American writer, activist, and feminist. She is best known for her first novel \"Rubyfruit Jungle\". Brown is also a mystery writer and screenwriter. Brown was born in 1944 in Hanover, Pennsylvania to an unmarried, teenage mother and her mother's married boyfriend. Brown's birth mother left the newborn Brown at an orphanage. Brown's mother's cousin Julia \"Juts\" Brown and her husband Ralph retrieved her from the orphanage, and raised her as their own in York, Pennsylvania, and later in Ft. Lauderdale, Florida. Julia and Ralph Brown were active Republicans",
    #             "score": "1.48156"
    #         },
    #         {
    #             "id": "20142345",
    #             "title": "Jessie Trout",
    #             "text": "Owen Sound in the summer. Jessie Trout Jessie Trout (July 26, 1895 – 1990) was a Canadian missionary to Japan for nearly 20 years until she left Japan during World War II. She was a leader in the Disciples of Christ (Campbell Movement) and the first woman to serve as vice president of the United Christian Missionary Society. She was a member of the Disciples of Christ, an author, translator, and co-founder of the Christian Women's Fellowship (1950) and the International Christian Women's Fellowship (1953). She received an honorary doctor of divinity degree from Bethany College in 1955. Jessie Mary",
    #             "score": "1.4498292"
    #         },
    #         {
    #             "id": "4563484",
    #             "title": "Louise Beavers",
    #             "text": "Filmmakers Hall of Fame in 1976. Features: Short subjects: Louise Beavers Louise Beavers (March 8, 1902 – October 26, 1962) was an American film and television actress. Beavers appeared in dozens of films and two hit television shows from the 1920s until 1960, most often cast in the role of a maid, servant, or slave. She was a native of Cincinnati, Ohio, and a member of Sigma Gamma Rho sorority, one of the four African-American sororities. Beavers was a breakthrough actress for black women and became known as a symbol of a \"mammy\" on the screen. A mammy archetype \"is",
    #             "score": "1.4398854"
    #         },
    #         {
    #             "id": "5034991",
    #             "title": "Jessie Mae Hemphill",
    #             "text": "In 2003, Hemphill's protégé and collaborator, Olga Wilhelmine Munding, established the Jessie Mae Hemphill Foundation to preserve and archive the African-American music of northern Mississippi and to provide assistance for regional musicians in need who could not survive on meager publishing royalties. One of Hemphill's songs was featured in the dance \"Tales from the Creek,\" by Reggie Wilson's Fist and Heel Performance Group, in a series of events celebrating black culture in Union Square Park, Manhattan in 1998. Jessie Mae Hemphill Jessie Mae Hemphill (October 18, 1923 – July 22, 2006) was an American electric guitarist, songwriter, and vocalist specializing",
    #             "score": "1.4380236"
    #         },
    #         {
    #             "id": "17716831",
    #             "title": "Virginia Mae Brown",
    #             "text": "Virginia Mae Brown Virginia Mae Brown (November 13, 1923 – February 15, 1991) was an American civil servant, government official, and lawyer. Brown (1923–91) was born on November 13, 1923, in Pliny, West Virginia. She had a sister (Anna) that was a year older and a brother (Winston) that was two years younger than her in the Brown family. The U.S. Census shows the Brown family to be living in Buffalo, West Virginia, in 1930 and 1940 - just across the Kanawha River from Pliny (where she was born). The 1940 U.S. Census shows Brown to be in her third",
    #             "score": "1.4338607"
    #         },
    #         {
    #             "id": "8063346",
    #             "title": "Jessie Willcox Smith",
    #             "text": "Jessie Willcox Smith Jessie Willcox Smith (September 6, 1863 – May 3, 1935) was an American female illustrator during the Golden Age of American illustration. She was considered \"one of the greatest pure illustrators\". She was a contributor to books and magazines during the late 19th and early 20th centuries. Smith illustrated stories and articles for clients such as \"Century\", \"Collier's\", \"Leslie's Weekly\", \"Harper's\", \"McClure's\", \"Scribners\", and the \"Ladies' Home Journal\". She had an ongoing relationship with \"Good Housekeeping\", which included the long-running Mother Goose series of illustrations and also the creation of all of the \"Good Housekeeping\" covers from",
    #             "score": "1.4252641"
    #         },
    #         {
    #             "id": "4563474",
    #             "title": "Louise Beavers",
    #             "text": "Louise Beavers Louise Beavers (March 8, 1902 – October 26, 1962) was an American film and television actress. Beavers appeared in dozens of films and two hit television shows from the 1920s until 1960, most often cast in the role of a maid, servant, or slave. She was a native of Cincinnati, Ohio, and a member of Sigma Gamma Rho sorority, one of the four African-American sororities. Beavers was a breakthrough actress for black women and became known as a symbol of a \"mammy\" on the screen. A mammy archetype \"is the portrayal within a narrative framework or other imagery",
    #             "score": "1.4207637"
    #         },
    #         {
    #             "id": "5034986",
    #             "title": "Jessie Mae Hemphill",
    #             "text": "bars a few times in the 1950s, most of her playing was done in family and informal settings, such as picnics with fife-and-drum music, until she was recorded in 1979. Her first recordings were field recordings made by the blues researcher George Mitchell in 1967 and the ethnomusicologist David Evans in 1973, but they were not released. She was then known as Jessie Mae Brooks, using the surname from a brief early marriage. In 1978, Evans came to Memphis, Tennessee, to teach at Memphis State University (now the University of Memphis). The school founded the High Water Recording Company in",
    #             "score": "1.402215"
    #         },
    #         {
    #             "id": "19181844",
    #             "title": "Jessie Rosser",
    #             "text": "Jessie Rosser Jessie Rosser \"(born 1921;died 2013)\" was a Missionary of the Canadian Baptist Ministries who served in India for over 40 years and was Principal of the Eva Rose York Bible Training and Technical School for Women in Tuni, Andhra Pradesh. Jessie worked as a school teacher at St. Thomas, Ontario for some time and then studied social sciences at the McMaster University from where she graduated in 1947 with a B.A. and decided to serve the cause of people in difficult circumstances overseas. She came to India in 1947 and served as a Missionary in Kakinada, Vuyyuru, and",
    #             "score": "1.3973815"
    #         },
    #         {
    #             "id": "3090588",
    #             "title": "Jessie Redmon Fauset",
    #             "text": "and 1930s, exploring the lives of the black middle class. She also was the editor and co-author of the African-American children's magazine \"The Brownies' Book\". She is known for discovering and mentoring other African-American writers, including Langston Hughes, Jean Toomer, Countee Cullen, and Claude McKay. She was born Jessie Redmona Fauset (later known as Jessie Redmon Fauset) on April 27, 1882, in Fredericksville, Camden County, Snow Hill Center Township, New Jersey. The town is now known as Lawnside, New Jersey. She was the seventh child of Redmon Fauset, an African Methodist Episcopal minister, and Annie (née Seamon) Fauset. Jessie's mother",
    #             "score": "1.392673"
    #         },
    #         {
    #             "id": "19181846",
    #             "title": "Jessie Rosser",
    #             "text": "in Tuni. Jessie Rosser Jessie Rosser \"(born 1921;died 2013)\" was a Missionary of the Canadian Baptist Ministries who served in India for over 40 years and was Principal of the Eva Rose York Bible Training and Technical School for Women in Tuni, Andhra Pradesh. Jessie worked as a school teacher at St. Thomas, Ontario for some time and then studied social sciences at the McMaster University from where she graduated in 1947 with a B.A. and decided to serve the cause of people in difficult circumstances overseas. She came to India in 1947 and served as a Missionary in Kakinada,",
    #             "score": "1.3898633"
    #         },
    #         {
    #             "id": "17536723",
    #             "title": "Jessie Brown Pounds",
    #             "text": "Jessie Brown Pounds Jessie Hunter Brown Pounds (August 31, 1861 – 1921) was an American lyricist of gospel songs. Jessie Hunter Brown was born into a farm family in the village of Hiram, Portage County. A staff writer for \"Christian Standard\", she often collaborated with composer Frederick A. Fillmore (1856–1925). In 1897 she married John E. Pounds, minister of the Central Christian Church in Indianapolis, IN. As a college-educated, frontier woman, she's considered by some to be part of the \"first generation\" of \"New Women.\" Her parents were Holland Brown and Jane Abel Brown. Holland Brown was baptized after hearing",
    #             "score": "1.3853728"
    #         },
    #         {
    #             "id": "8634517",
    #             "title": "Jessie Hill",
    #             "text": "infection while on tour in Tokyo on May 4, 2015. Jessie Hill Jessie Hill (December 9, 1932 – September 17, 1996) was an American R&B and Louisiana blues singer and songwriter, best remembered for the classic song \"Ooh Poo Pah Doo\". Hill was born in New Orleans, Louisiana, United States. By his teens he was playing drums in local bands, and in 1951 he formed his own group, the House Rockers. After periods performing as drummer with Professor Longhair and then Huey \"Piano\" Smith, Hill formed a new version of the House Rockers in 1958, which enabled him to focus",
    #             "score": "1.3831363"
    #         },
    #         {
    #             "id": "20142338",
    #             "title": "Jessie Trout",
    #             "text": "Jessie Trout Jessie Trout (July 26, 1895 – 1990) was a Canadian missionary to Japan for nearly 20 years until she left Japan during World War II. She was a leader in the Disciples of Christ (Campbell Movement) and the first woman to serve as vice president of the United Christian Missionary Society. She was a member of the Disciples of Christ, an author, translator, and co-founder of the Christian Women's Fellowship (1950) and the International Christian Women's Fellowship (1953). She received an honorary doctor of divinity degree from Bethany College in 1955. Jessie Mary Trout was born to Archibald",
    #             "score": "1.3827628"
    #         },
    #         {
    #             "id": "5034985",
    #             "title": "Jessie Mae Hemphill",
    #             "text": "Jessie Mae Hemphill Jessie Mae Hemphill (October 18, 1923 – July 22, 2006) was an American electric guitarist, songwriter, and vocalist specializing in the North Mississippi hill country blues traditions of her family and regional heritage. Hemphill was born near Como and Senatobia, Mississippi, in the northern Mississippi hill country, just east of the Mississippi Delta. She began playing the guitar at the age of seven. She also played drums in local fife-and-drum bands, beginning with the band led by her paternal grandfather, Sid Hemphill, in which she played snare drum and bass drum. Aside from sitting in at Memphis",
    #             "score": "1.3797479"
    #         },
    #         {
    #             "id": "7711619",
    #             "title": "Mae Beavers",
    #             "text": "resign her spot in the state senate to focus fully on her campaign. Mark Pody won a special election to assume Beavers' senate seat. On January 30, 2018, Beavers announced that she would be stepping out of the 2018 Tennessee gubernatorial race. In March 2018, Beavers announced her candidacy in the Wilson County mayoral election. Beavers is married to Jerry Beavers, with whom she has two children. They attend Trevecca Community Church. Mae Beavers Mae Beavers (born December 11, 1947 in Millport, Alabama) is an American former politician. A Republican, she was a member of the Tennessee Senate for the",
    #             "score": "1.3781004"
    #         },
    #         {
    #             "id": "13237049",
    #             "title": "Irene Bennett Brown",
    #             "text": "of Women Writing the West. She continues to live in Oregon with her husband, Bob. Irene Bennett Brown Irene Bennett Brown is an American author of children's, young adult and adult fiction. Brown was born in Topeka, Kansas and when she was nine years old, moved with her family from Kansas to the Willamette Valley in Oregon. Brown's fourth book, \"To Rainbow Valley\", became the first one to sell and be published in 1969. It was re-released as an Easy Reader book in 2001. Brown has her own publishing company, Riveredge Books, which has published and re-issued several of her",
    #             "score": "1.3779411"
    #         },
    #         {
    #             "id": "20865962",
    #             "title": "Jessie Saulteaux",
    #             "text": "Jessie Saulteaux Jessie Prettyshield Saulteaux (1912 - 1995) was a Canadian Assiniboine elder and theological leader. Early in life, Saulteaux desired to become a nurse, but she was unable to do so due to reasons of race. Instead she turned her talents towards helping her community, the Carry-the-Kettle First Nation, and her church. She was among the first women in Saskatchewan to be elected tribal chief; she also supported the development of ministers and church leaders from the First Nations community. She participated in the founding of the All Native Circle Conference in the United Church of Canada, and the",
    #             "score": "1.3749332"
    #         },
    #         {
    #             "id": "4563475",
    #             "title": "Louise Beavers",
    #             "text": "of a black domestic servant, generally good-natured, often overweight, and loud\". Beavers was born in Cincinnati, Ohio, to school teacher Ernestine Monroe Beavers and William M. Beavers, who was originally from Georgia. Due to her mother's illness, Louise and her parents moved to Pasadena, California. In Pasadena, she attended school and engaged in several after-school activities, such as basketball and church choir. Her mother also worked as a voice teacher and taught Louise how to sing for concerts. In June 1920, she graduated from Pasadena High School. She worked as a dressing room attendant for a photographer and served as",
    #             "score": "1.3708005"
    #         },
    #         {
    #             "id": "17536724",
    #             "title": "Jessie Brown Pounds",
    #             "text": "Walter Scott preach; and the couple were abolitionists. Her parents hosted pioneers and luminaries including James A. Garfield. \"Her pen produced upwards of eight hundred hymns, eighty short stories, seven novels, lyrics, and scripts for cantatas, and numerous brief essays and non-fiction articles.\" \"Anywhere with Jesus\" is possibly the most well-known of her poems. Some of her poems have been set to a number of musical scores, the most familiar being the tune \"Serenity\" by Daniel B. Towner (1850–1919). Her 1896 poem \"Beautiful Isle\" became the song \"Beautiful Isle of Somewhere\", which was sung at President McKinley's funeral and criticized",
    #             "score": "1.3702077"
    #         },
    #         {
    #             "id": "2216611",
    #             "title": "Benny Beaver",
    #             "text": "of regents, Bell became hugely popular among the students for his ritual of marching to the Marys River after each of Oregon State's Civil War victories. He was said to have tossed his top hat into the water as a token of celebration. Earlier mascots include \"Jimmie\" the Coyote (1892–1893) and \"Bulldog\" (1906–1910, unofficial and for specific teams only, such as the Wrestling squad). The beaver mascot's name, \"Benny,\" was officially adopted in 1945. Two failed attempts to maintain a live beaver mascot include Bevo Beaver (rescued from Mary's River in 1921 and later stolen ) and Billy Beaver (made",
    #             "score": "1.3672736"
    #         },
    #         {
    #             "id": "3980421",
    #             "title": "June Brown",
    #             "text": "career, she played the roles of Hedda Gabler and Lady Macbeth. In 2009, Brown played Jessie in the West End production of \"Calendar Girls\" at the Noël Coward Theatre. Also in the play were former \"EastEnders\" stars Anita Dobson (Angie Watts), Jack Ryder (Jamie Mitchell) and Jill Halfpenny (Kate Mitchell). June Brown June Muriel Brown, (born 16 February 1927) is an English actress, known for her role as Dot Cotton in the BBC soap opera \"EastEnders\" from 1985 onwards. In 2005, she won Best Actress at the Inside Soap Awards, and in the same year, also received the Lifetime Achievement",
    #             "score": "1.3651499"
    #         },
    #         {
    #             "id": "19998679",
    #             "title": "Jessie Isabelle Price",
    #             "text": "she began working on vaccine development for \"Pasteurella anatipestifer\" for white pekin ducks, which she would continue in avian cholera and TB for various species through her career. Some of the vaccines were commercially developed. She worked with national and international colleagues, publishing on \"Pasteurella anatipestifer\" in pheasants, medication for bacterial infections in ducklings, \"Pasteurella multocida\" in Nebraska wetlands and in snow geese. There is an extensive photoessay publicly available in \"Ebony\" magazine. See also a photo of her in later life in an obituary. Jessie Isabelle Price Jessie Isabelle Price (1930-2015) was a veterinary microbiologist. She isolated and reproduced",
    #             "score": "1.3603673"
    #         }
    #     ]
    # }]


    if args.task is not None and args.task == "factscore":
        new_results = []
        for idx, item in enumerate(input_data):
            prompt = item["input"]
            ctxs = item["ctxs"][:args.ndocs]
            processed_prompt = PROMPT_DICT["prompt_no_input"].format_map({"instruction": prompt})
            result, intermediate = call_model_beam_batch(processed_prompt, model=model, max_new_tokens=args.max_new_tokens, ctxs=ctxs, query=prompt,
                                     rel_tokens=rel_tokens, ret_tokens=ret_tokens, grd_tokens=grd_tokens, ut_tokens=ut_tokens,
                                     use_seqscore=args.use_seqscore, threshold=args.threshold, 
                                     beam_width=args.beam_width, max_depth=args.max_depth,
                                     w_rel=1.0, w_sup=1.0, w_use=0.5, mode=args.mode, ignore_cont=args.ignore_cont, )
            print("result => \n",result)
            
            postprocessed_result = fix_spacing(postprocess(result[0]))
            new_results.append({"input": item["input"], "output": postprocessed_result, "topic": item["topic"],
                                "cat": item["cat"], "intermediate": intermediate["original_splitted_sentences"][0]})
            if idx % 10 == 0:
                with jsonlines.open(args.output_file + "_tmp", 'w') as writer:
                    writer.write_all(new_results)
        with jsonlines.open(args.output_file, 'w') as writer:
            writer.write_all(new_results)

    elif args.task is not None and (args.task in ["asqa", "eli5"]):
        new_results = {"data": [], "args": [],
                       "total_cost": 0.0, "azure_filter_fail": ""}
        for instance_idx, item in enumerate(input_data):
            prompt = item["question"]
            ctxs = item["docs"][:args.ndocs]
            instructions = TASK_INST[args.task]
            prev_gen = []
            prompt = instructions + "## Input:\n\n" + prompt
            final_pred, intermediate = call_model_beam_batch(prompt, model=model, max_new_tokens=args.max_new_tokens, ctxs=ctxs, query=prompt,
                                     rel_tokens=rel_tokens, ret_tokens=ret_tokens, grd_tokens=grd_tokens, ut_tokens=ut_tokens,
                                     use_seqscore=args.use_seqscore, threshold=args.threshold, 
                                     beam_width=args.beam_width, max_depth=args.max_depth,
                                     w_rel=1.0, w_sup=1.0, w_use=0.5, mode=args.mode, ignore_cont=args.ignore_cont, )
            
            final_output = ""
            docs = []
            prev_gen = []
            if "splitted_sentences" not in intermediate:
                item["output"] = postprocess(final_pred)
            else:
                if len(postprocess(final_pred[0])) == 0:
                    intermediate["splitted_sentences"][0], intermediate["ctxs"][
                        0] = intermediate["splitted_sentences"][1], intermediate["ctxs"][1]
                for idx, (sent, doc) in enumerate(zip(intermediate["splitted_sentences"][0], intermediate["ctxs"][0])):
                    if len(sent) == 0:
                        continue
                    postprocessed_result = postprocess(sent)
                    if postprocessed_result in prev_gen:
                        continue
                    else:
                        prev_gen.append(postprocessed_result)
                    final_output += postprocessed_result[:-
                                                         1] + " [{}]".format(idx) + ". "
                    docs.append(doc)
                if len(final_output) == 0:
                    item["output"] = fix_spacing(final_output)
                if len(final_output) > 0 and final_output[-1] == " ":
                    final_output = final_output[:-1]
                item["output"] = fix_spacing(final_output)
                item["output"] = item["output"].replace(
                    ".[Continue to Use Evidence]", " [1]. ")
                item["output"] = item["output"].replace(". [1] ", " [1]. ")
            item["docs"] = docs
            if "original_splitted_sentences" in intermediate:
                item["intermediate"] = intermediate["original_splitted_sentences"][0]
            new_results["data"].append(item)

            if instance_idx % 10 == 0:
                with open(args.output_file + "_tmp", 'w') as writer:
                    json.dump(new_results, writer)
        with open(args.output_file, 'w') as writer:
            json.dump(new_results, writer)
    else:
        raise NotImplementedError





if __name__ == "__main__":
    main()
