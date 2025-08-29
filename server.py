import litserve as ls
from transformers import AutoModelForCausalLM, pipeline, AutoTokenizer, TextIteratorStreamer
from threading import Thread

class Gemma3API(ls.LitAPI):
    def setup(self, device):
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
        self.pipeline = pipeline("text-generation", model="google/gemma-3-270m-it",device="cpu") # device="cuda" for GPU
        self.streamer = TextIteratorStreamer(self.tokenizer)

    def decode_request(self, request):
        messages = [{"role": "system", "content": "You are a sassy, wise-cracking robot as imagined by Hollywood circa 1986."}]
        messages.append({"role": "user", "content": request["prompt"]})
        return messages   

    def predict(self, request):
        # Run the generation in a separate thread, so that we can fetch the generated text in a non-blocking way.
        generation_kwargs = dict(text_inputs=request,streamer=self.streamer, max_new_tokens=1000)
        thread = Thread(target=self.pipeline, kwargs=generation_kwargs)
        thread.start()
        yield from self.streamer

    def encode_response(self, output):
        for out in output:
            yield {"output": out}

if __name__ == "__main__":
    server = ls.LitServer(Gemma3API(stream=True), accelerator="auto")
    server.run(port=8000)


# --------------------------------using model directly--------------------------------
# class Gemma3API(ls.LitAPI):
#     def setup(self, device):
#         self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
#         self.model = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m-it",device="cpu")
#         self.streamer = TextIteratorStreamer(self.tokenizer)

#     def decode_request(self, request):
#         messages = [{"role": "system", "content": "You are a sassy, wise-cracking robot as imagined by Hollywood circa 1986."}]
#         messages.append({"role": "user", "content": request["prompt"]})
#         inputs = self.tokenizer.apply_chat_template(
#             messages,
#             add_generation_prompt=True,
#             tokenize=True,
#             return_dict=True,
#             return_tensors="pt",
#         ).to(self.model.device)
#         return inputs 

#     def predict(self, request):
#         # Run the generation in a separate thread, so that we can fetch the generated text in a non-blocking way.
#         generation_kwargs = dict(request, streamer=self.streamer, max_new_tokens=1000)
#         thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
#         thread.start()
#         yield from self.streamer

#     def encode_response(self, output):
#         for out in output:
#             yield {"output": out}

