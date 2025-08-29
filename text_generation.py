# --------------------------------using pipeline--------------------------------
print("///////////////////////////using pipeline///////////////////////////////")

from transformers import pipeline

pipeline = pipeline("text-generation", model="google/gemma-3-270m-it",device="cpu") # device="cuda" for GPU
max_new_tokens=50

# text completion
result=pipeline("The secret to baking a good cake is ",max_new_tokens=max_new_tokens)
print("----------------------Text completion----------------------")
print(result)

# chat completion
chat=[
    {"role": "system", "content": "You are a sassy, wise-cracking robot as imagined by Hollywood circa 1986."},
    {"role": "user", "content": "Hey, can you tell me any fun things to do in New York?"}
    ]
result=pipeline(chat,max_new_tokens=max_new_tokens)
print("----------------------Chat completion----------------------")
print(result)


# --------------------------------using model directly--------------------------------
print("///////////////////////////using model directly///////////////////////////////")

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m-it")
max_new_tokens=50

# text completion
inputs = tokenizer(["The secret to baking a good cake is "], return_tensors="pt").to("cpu") # "cuda" for GPU
generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
result=tokenizer.batch_decode(generated_ids)[0]
print("----------------------Text completion----------------------")
print(result)

# chat completion
chat=[
    {"role": "system", "content": "You are a sassy, wise-cracking robot as imagined by Hollywood circa 1986."},
    {"role": "user", "content": "Hey, can you tell me any fun things to do in New York?"}
    ]
inputs = tokenizer.apply_chat_template(
    chat,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    ).to("cpu") # "cuda" for GPU
generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
result=tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[-1]:])
print("----------------------Chat completion----------------------")
print(result)