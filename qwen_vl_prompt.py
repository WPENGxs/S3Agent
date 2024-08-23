agent_prompt_0 = '''
Given an image and corresponding text, your task is to detect whether the text contains sarcasm or not by analyzing the superficial expression and underlying critique conveyed through the discrepancies between the image and text content.

You should consider surface-level textual elements like exaggeration, understatement, rhetorical questions, and figurative language that don't align with the visual information. But also look deeper at the overall tone, attitude, and implication of the text that reveals an underlying critical stance towards the subject matter shown in the image.

Provide your judgment on whether the text is sarcastic or not given the image-text discrepancies. Support your answer with a "Chain-of-thought" explaining your reasoning step-by-step.

The format should be:
Chain-of-thought: [Step 1: ..., Step 2: ..., Step 3: ..., etc. Ending with your final judgment and explanation]

text: <text>
'''
agent_prompt_1 = '''
Given an image and corresponding text, your task is to detect whether the text contains sarcasm or not by analyzing the semantic relationship and interplay between the image and text content. Sarcasm often involves a mismatch or contrast between the literal meaning of the text and the actual intended meaning when considered in the context of the image.

You should look for cues such as exaggeration, understatement, irony, rhetorical questions, and metaphorical or figurative language in the text that contradicts or doesn't align with what is depicted in the image. Consider the overall tone and implication of the text combined with the subject matter and details shown in the image.

In your response, provide your judgment on whether the text is sarcastic or not based on the image-text semantic relationship. Support your answer by providing a "Chain-of-thought" that explains your reasoning step-by-step.

The format should be:
Chain-of-thought: [Step 1: ..., Step 2: ..., Step 3: ..., etc. Ending with your final judgment and explanation]

text: <text>
'''
agent_prompt_2 = '''
Given an image and corresponding text, your task is to detect whether the text contains sarcasm or not by analyzing the sentiment expressed and any criticism conveyed towards specific subjects or behaviors depicted in the content.

You should consider the overall sentiment and emotional language used in the text, and whether it expresses a negative, critical stance that clashes with the subject matter or behavior illustrated in the image. Identifying sarcasm requires understanding the implied sentiment beyond just the surface meaning of the words.

Provide your judgment on whether the text is sarcastic or not given the sentiment expression and any criticism towards specific subjects/behaviors depicted. Support your answer with a "Chain-of-thought" explaining your reasoning step-by-step.

The format should be:
Chain-of-thought: [Step 1: ..., Step 2: ..., Step 3: ..., etc. Ending with your final judgment and explanation]

text: <text>
'''
agent_prompt_3 = '''
Given a text comment, you need to analyze it from three perspectives: Superficial expression, Semantic information, and Sentiment expression. Combine these perspectives to determine if the comment is sarcastic or not.
Follow these rules:
1. If any perspective cannot determine sarcasm due to lack of information, disregard that perspective.
2. If the remaining perspectives conflict, choose the answer with the most well-reasoned justification.
3. Finally, give the answer in JSON format: {"Answer": "yes" or "no"}

Perspectives:
Superficial Expression: <output_0>
Semantic Information: <output_1>
Sentiment Expression: <output_2>
'''