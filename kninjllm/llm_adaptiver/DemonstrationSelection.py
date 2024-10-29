from openai import OpenAI
from root_config import RootConfig

class DemonstrationSelection:
    
	def __init__():
		pass
    
	domain_selection_demonstration = """
		Follow the below example, select relevant knowledge domains from Available Domains to the Q.
		Available Domains: factual, medical, physical, biology

		Q: This British racing driver came in third at the 2014 Bahrain GP2 Series round and was born in what year
		Relevant domains: factual

		Q: Which of the following drugs can be given in renal failure safely?
		Relevant domains: medical

		Q: Which object has the most thermal energy? 
		Relevant domains: factual, physical

		Q: Is the following trait inherited or acquired? Barry has a scar on his left ankle. 
		Relevant domains: biology

	"""


	def source_selection(self,domain_selection_demonstration,input_query,model_name,api_key=None):

		prompt_question_input=domain_selection_demonstration+"Q: "+input_query+"\nRelevant domains: "

		s1_domains=self.call_openai_api(RootConfig.openai_model_version, prompt_question_input, max_tokens=1024, temperature=0, n=1)
		s1_domains=[x.message.content.strip() for x in s1_domains[0].choices][0]
		s1_domains=s1_domains.strip().split(", ")

		
		return s1_domains

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