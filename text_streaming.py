# --------------------------- Using Pipeline ---------------------------
print("/////////////////////////// Using Pipeline ///////////////////////////////")

from transformers import pipeline, TextIteratorStreamer, AutoTokenizer
from threading import Thread

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
pipe = pipeline("text-generation", model="google/gemma-3-270m-it", device="cpu")  # "cuda" for GPU
max_new_tokens = 1000
streamer = TextIteratorStreamer(tokenizer)

# Text Completion
input_text = "The secret to baking a good cake is "
gen_kwargs = dict(text_inputs=input_text, streamer=streamer, max_new_tokens=max_new_tokens)
thread = Thread(target=pipe, kwargs=gen_kwargs)
thread.start()
generated = ""
print("---------------------- Text Completion ----------------------")
for token in streamer:
    generated += token
    print(token)
print("\n" + generated)

# Chat Completion
chat = [
    {"role": "system", "content": "You are a sassy, wise-cracking robot as imagined by Hollywood circa 1986."},
    {"role": "user", "content": "Hey, can you tell me any fun things to do in New York?"}
]
gen_kwargs = dict(text_inputs=chat, streamer=streamer, max_new_tokens=max_new_tokens)
thread = Thread(target=pipe, kwargs=gen_kwargs)
thread.start()
generated = ""
print("---------------------- Chat Completion ----------------------")
for token in streamer:
    generated += token
    print(token)
print("\n" + generated)

# ----------------------- Using Model Directly -----------------------
print("/////////////////////////// Using Model Directly ///////////////////////////////")

from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m-it")
max_new_tokens = 50
streamer = TextIteratorStreamer(tokenizer)

# Text Completion
inputs = tokenizer(["The secret to baking a good cake is "], return_tensors="pt").to("cpu")  # "cuda" for GPU
gen_kwargs = dict(inputs, streamer=streamer, max_new_tokens=max_new_tokens)
thread = Thread(target=model.generate, kwargs=gen_kwargs)
thread.start()
generated = ""
print("---------------------- Text Completion ----------------------")
for token in streamer:
    generated += token
    print(token)
print("\n" + generated)

# Chat Completion
chat = [
    {"role": "system", "content": "You are a sassy, wise-cracking robot as imagined by Hollywood circa 1986."},
    {"role": "user", "content": "Hey, can you tell me any fun things to do in New York?"}
]
inputs = tokenizer.apply_chat_template(
    chat,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to("cpu")  # "cuda" for GPU
gen_kwargs = dict(inputs, streamer=streamer, max_new_tokens=max_new_tokens)
thread = Thread(target=model.generate, kwargs=gen_kwargs)
thread.start()
generated = ""
print("---------------------- Chat Completion ----------------------")
for token in streamer:
    generated += token
    print(token)
print("\n" + generated)