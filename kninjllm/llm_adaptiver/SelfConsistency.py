
from openai import OpenAI
from root_config import RootConfig

class SelfConsistency:
    
	def __init__():
		pass
    
	def self_consisitency_timing_judge(self,input_query,model_name=RootConfig.openai_model_version,api_key=None,max_tokens=256, temperature=0.7, n=10, threshold=0.7):
			
		cot_sc_responses = self.call_openai_api(model_name, input_query, max_tokens=max_tokens, temperature=temperature, n=n)

		if cot_sc_responses is not None:
			all_cot_text_response = [x.message.content.strip() for x in cot_sc_responses[0].choices]
			most_common_answer = max(set(all_cot_text_response), key = all_cot_text_response.count)
			most_common_answer_indices = [i for i, x in enumerate(all_cot_text_response) if x == most_common_answer]
			sc_score = float(len(most_common_answer_indices)) / len(all_cot_text_response)
			if sc_score<threshold:
				return False
			else:
				return True
		else:
			raise Exception("Stage 1: OpenAI API call failed")

	
	def call_openai_api(self,model, input_text, max_tokens=256, temperature=0, n=1):
		api_key = "sk-xxx"
		error_times = 0
		client = OpenAI(api_key=api_key)
		while error_times < 5:
			try:
				if "text-davinci" in model:
					
					model="gpt-3.5-turbo-instruct"
					response=client.completions.create(
						model=model,
						prompt=input_text,
						max_tokens=max_tokens,
						seed=0,
						temperature=temperature,
						n=n,
					)

					return [response, response.choices[0].text]
				elif "gpt-" in model:
					
					response = client.chat.completions.create(
						model=model,
						messages = [
							{"role": "system", "content": "You are a helpful assistant."},
							{"role": "user", "content": input_text}
						],
						seed=0,
						max_tokens=max_tokens,
						temperature=temperature,
						n=n,
						timeout=60,
						
					)
					
					return [response, response.choices[0].message.content]
				else:
					raise Exception("Invalid model name")
			except Exception as e:
				print('Retry due to:', e)
				error_times += 1
				continue
			
		return None

