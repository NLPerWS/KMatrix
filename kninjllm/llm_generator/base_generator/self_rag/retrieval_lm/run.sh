# python run_long_form_static.py \
#   --model_name selfrag_llama2_7b \
#   --ndocs 5 --max_new_tokens 300 --threshold 0.2 \
#   --use_grounding --use_utility --use_seqscore \
#   --task asqa --input_file eval_data/asqa_eval_gtr_top100.json \
#   --output_file YOUR_OUTPUT_FILE_NAME --max_depth 7 --mode always_retrieve \


python run_short_from_origin.py \
  --model_name selfrag_llama2_7b \
  --ndocs 5 --max_new_tokens 300 --threshold 0.2 \
  --use_grounding --use_utility --use_seqscore \
  --task asqa --input_file eval_data/asqa_eval_gtr_top100.json \
  --output_file YOUR_OUTPUT_FILE_NAME --max_depth 7 --mode always_retrieve \