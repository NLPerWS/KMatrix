python retrieval_lm/run_long_form_static.py \
  --model_name selfrag_llama2_7b \
  --ndocs 5 --max_new_tokens 300 --threshold 0.2 --beam_width 1 \
  --use_grounding --use_utility --use_seqscore \
  --task factscore --input_file eval_data/factscore_unlabeled_alpaca_13b_retrieval.jsonl \
  --output_file factscore_output --max_depth 7  --mode always_retrieve\
