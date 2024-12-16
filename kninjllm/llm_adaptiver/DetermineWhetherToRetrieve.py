from vllm import SamplingParams
from kninjllm.llm_generator.base_generator.self_rag.self_rag_generator import RagGenerator
from root_config import RootConfig
import numpy as np
from kninjllm.llm_generator.base_generator.self_rag.retrieval_lm.utils import load_special_tokens 

class DetermineWhetherToRetrieve:
    
    def __init__():
        pass
    
    def determine(one_prompt):
        threshold = 0.2
        model = RagGenerator(model_path=RootConfig.selfRAG_model_path,executeType="infer")
        sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=25, logprobs=32000,skip_special_tokens=False)
        pred = model.run(query_obj={"question":one_prompt}, sampling_params=sampling_params,saveLogFlag=False)['final_result']['meta']['pred']
        pred_log_probs = pred['outputs'][0]['logprobs']
        ret_tokens, rel_tokens, grd_tokens, ut_tokens = load_special_tokens(model.tokenizer, use_grounding=True, use_utility=True)

        if "[Retrieval]" not in pred:
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
                
        return do_retrieve