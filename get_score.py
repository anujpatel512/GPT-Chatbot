import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig

model_name = "/baldig/deeplearning/mirana_llm/llama/models_hf/7B"

n_gpus = torch.cuda.device_count()

max_memory = {i: '20GB' for i in range(n_gpus)}


model = LlamaForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_name,
    load_in_4bit=True,
    device_map='sequential',
    max_memory=max_memory,
    torch_dtype=torch.bfloat16,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4'
    ),
)
tokenizer = LlamaTokenizer.from_pretrained(model_name, legacy = False)
prompt = """
Answer the following multiple choice question:
C. B. is a 25-year-old white female who was recently diagnosed with generalized anxiety disorder. She has been taking
escitalopram 10 mg po once daily for 1 week now but complains of increased anxiety. What would be the best course of
action for C. B.’s physician?
A. Increase the dose of escitalopram to 10 mg bid
B. Decrease the escitalopram to 5 mg po once daily, and add lorazepam 0.5 mg tid prn anxiety for 2 weeks
C. Add alprazolam 1 mg po bid prn anxiety for 2 weeks
D. Discontinue escitalopram, and initiate citalopram 20 mg po qd.
E. No medication therapy change is necessary. The anxiety will resolve on its ow
---
Answer:
"""


input_ids = tokenizer.encode(prompt, return_tensors="pt").to('cuda:1') 
with torch.no_grad():
    generated_ids = model.generate(
        input_ids,
        do_sample=True,
        min_length=0,
        max_length=400,
        top_p=0.9,
        temperature=0.7,
    )

generated_text = tokenizer.decode(
    [el.item() for el in generated_ids[0]], skip_special_tokens=True)

print(generated_text)