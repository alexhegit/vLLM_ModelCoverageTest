import os
import torch
gpus = [0, 1]
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ','.join(map(str, gpus)))
print(f"PyTorch detected number of available devices: {torch.cuda.device_count()}")
tp = torch.cuda.device_count()

from vllm import LLM, SamplingParams
prompts = [
    "The capital of France is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

model_id = "facebook/opt-125m"
llm = LLM(model=model_id, tensor_parallel_size = tp)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
