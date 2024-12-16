python retrieval_lm/run_short_form.py \
--model_name selfrag_llama2_7b \
--input_file eval_data/popqa_longtail_w_gs.jsonl \
--max_new_tokens 100 \
--threshold 0.2 \
--output_file popqa_OUTPUT \
--metric match --ndocs 10 --use_groundness --use_utility --use_seqscore \
--dtype half


# Final result: 0.5511079342387419
# Retrieval Frequencies: 174.875