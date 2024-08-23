agent_prompt_0 = '''
Given the following image and text, please judge whether there is sarcasm based on the superficial expression. This includes detect underlying critiques in contexts through image-text discrepancies. Without considering conclusions drawn solely from images or text, both must be considered together. Then, you should output the corresponding chain-of-thought to support your answer.
text: <text>
'''
agent_prompt_1 = '''
Given the following image and text, please judge whether there is sarcasm based on the semantic information. This includes detect extreme portrayals and metaphorical in contexts through image-text semantic. Without considering conclusions drawn solely from images or text, both must be considered together. Then, you should output the corresponding chain-of-thought to support your answer.
text: <text>
'''
agent_prompt_2 = '''
Given the following image and text, please judge whether there is sarcasm based on the sentiment expression. This includes detect criticize emotion on specific subjects or behaviors in the content. Without considering conclusions drawn solely from images or text, both must be considered together. Then, you should output the corresponding chain-of-thought to support your answer.
text: <text>
'''
agent_prompt_3 = '''
Unmask the hidden intent!  Given a text comment, delve into its layers of meaning.  Analyze the surface - the literal words used.  Pierce deeper to uncover the semantic information - the intended meaning behind those words.  Finally, gauge the sentiment - the emotional undercurrent.  By weaving these insights together, can you crack the code of sarcasm and determine if the comment is meant to be sincere or laced with sarcastic?
Follow these rules:
1. If any perspective cannot determine sarcasm due to lack of information, disregard that perspective.
2. If after one answer is disregard the remaining two views conflict, choose the answer with the most well-founded reasoning.
3. Finally, give the answer in JSON format: {"Answer": "yes" or "no"}

Perspectives:
Superficial Expression: <output_0>
Semantic Information: <output_1>
Sentiment Expression: <output_2>
'''