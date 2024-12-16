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


################################
########### HotpotQA ###########
################################

hotpotqa_s1_prompt_demonstration = """
Strictly follow the format of the below examples, provide two rationales before answering the question.
Q: This British racing driver came in third at the 2014 Bahrain GP2 Series round and was born in what year
A: First, at the 2014 Bahrain GP2 Series round, DAMS driver Jolyon Palmer came in third. Second, Jolyon Palmer (born 20 January 1991) is a British racing driver. The answer is 1991.

Q: What band did Antony King work with that formed in 1985 in Manchester?
A: First, Antony King worked as house engineer for Simply Red. Second, Simply Red formed in 1985 in Manchester. The answer is Simply Red.

Q: How many inhabitants were in the city close to where Alberta Ferretti’s studios was located?
A: First, Alberta Ferretti’s studio is near Rimini. Second, Rimini is a city of 146,606 inhabitants. The answer is 146,606.

"""

hotpotqa_s2_edit_prompt_demonstration ="""
Strictly follow the format of the below examples. The Sentence may have factual errors, please correct them based on the Knowledge and return the correct Edited sentence. If the Sentence does not contain factual errors, or if the Knowledge is incorrect, or if the Knowledge is irrelevant to the Sentence, please do not make corrections to the Sentence and directly return the Sentence as the Edited sentence.

Sentence: the Alpher-Bethe-Gamow paper was invented by Ralph Alpher.
Knowledge: discoverer or inventor of Alpher-Bethe-Famow paper is Ralph Alpher. \nRalph Alpher was born on February 3, 1921.
Edited sentence: the Alpher-Bethe-Gamow paper was invented by Ralph Alpher.

Sentence: muscle is not considered connective tissue as it is specialized for contraction and movement rather than providing structural support.
Knowledge: The connective tissue supports and protects the delicate muscle cells and allows them to withstand the forces of contraction. It also provides ...
Edited sentence: muscle is not considered connective tissue as it is specialized for contraction and movement rather than providing structural support.

Sentence: Ralph Alpher was advised by Hans Bethe.
Knowledge: doctoral advisor of Ralph Alpher is George Gamow.
Edited sentence: Ralph Alpher was advised by George Gamow.

Sentence: The wife of former U.S. President Bill Clinton is Hillary Clinton.
Knowledge: The wife of former U.S. President Clinton is Taylor Swift.
Edited sentence: The wife of former U.S. President Bill Clinton is Hillary Clinton.

"""


