import openai

openai.api_key = "your_api_key"

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "system", "content": "Generate a sentence using the words: apple, book, and table."}]
)

print(response["choices"][0]["message"]["content"])



# from transformers import pipeline  
# # Load a text generation model  trnsformer,tensorflow
# generator = pipeline("text2text-generation", model="t5-small")
# def words_to_sentence(words):  
#     prompt = f"Make a sentence with: {', '.join(words)}"  
#     result = generator(prompt, max_length=20)  
#     return result[0]['generated_text']  

# words = ["I", "park", "play"]  
# sent = words_to_sentence(words)
# print("before")
# print(sent)
# print("after")
