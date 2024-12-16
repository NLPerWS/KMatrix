python retrieval_lm/run_short_form.py \
  --model_name selfrag_llama2_7b \
  --input_file eval_data/arc_challenge_processed.jsonl \
  --max_new_tokens 50 --threshold 0.2 \
  --output_file arc_output \
  --metric match --ndocs 5 --use_groundness --use_utility --use_seqscore \
  --task arc_c --mode adaptive_retrieval