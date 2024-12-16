from src.contriever import Contriever
from transformers import AutoTokenizer



model = Contriever.from_pretrained("contriever_model") 
tokenizer = AutoTokenizer.from_pretrained("contriever_model") 


sentences = [
    "Where was Marie Curie born?",
    "Maria Sklodowska, later known as Marie Curie, was born on November 7, 1867.",
    "Born in Paris on 15 May 1859, Pierre Curie was the son of Eugène Curie, a doctor of French Catholic origin from Alsace."
]


inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
embeddings = model(**inputs)


score01 = embeddings[0] @ embeddings[1] 
score02 = embeddings[0] @ embeddings[2] 



print("------------------------------")
print("embeddings=>\n",embeddings)
print("------------------------------")

print("score01=>\n",score01.item())
print("score02=>\n",score02.item())