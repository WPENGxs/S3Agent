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
Given a text comment, you need to analyze it from three perspectives: Superficial expression, Semantic information, and Sentiment expression. Combine these perspectives to determine if the comment is sarcastic or not.
Follow these rules:
1. If any perspective cannot determine sarcasm due to lack of information, disregard that perspective.
2. If the remaining perspectives conflict, choose the answer with the most well-reasoned justification.
3. Finally, give the answer in JSON format: {"Answer": "<<<yes>>>"} or {"Answer": "<<<no>>>"}

Perspectives:
Superficial Expression: <output_0>
Semantic Information: <output_1>
Sentiment Expression: <output_2>
'''