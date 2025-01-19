import os
import torch
import logging
from argparse import ArgumentParser

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from vllm import LLM, SamplingParams

def infer_with_model(model_id, gpus):
    # Log the start of inference for each model
    logging.info(f"Starting inference for model: {model_id}, TP: {gpus}")
    try:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", ','.join(map(str, gpus)))
        tp = len(gpus)

        llm = LLM(model=model_id, tensor_parallel_size=tp)

        # Fixed list of prompts
        prompts = ["The capital of France is"]

        outputs = llm.generate(prompts, SamplingParams(temperature=0.8, top_p=0.9))
        if not outputs or len(outputs) == 0:
            raise ValueError("No outputs received from the model.")

        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            # Log the prompt and generated text
            logging.info(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        return {"model_id": model_id, "status": "PASS"}
    except Exception as e:
        # Log any errors that occur during inference for each model
        logging.error(f"Error occurred while inferring model {model_id}: {e}")
        return {"model_id": model_id, "status": "FAILED"}

def main():
    parser = ArgumentParser(description="Test a single model with specified prompts.")
    # New argument for model ID with default value 'facebook/opt-125m'
    parser.add_argument("--model", type=str, default="facebook/opt-125m",
                        help="Model ID to test")
    # New argument for GPU list with default value '0'
    parser.add_argument("--gpus", type=str, default="0",
                        help="GPU indices separated by commas (e.g., 0,1)")
    args = parser.parse_args()
    model_id = args.model
    gpus = [int(gpu) for gpu in args.gpus.split(',')]
    result = infer_with_model(model_id, gpus)
    # Log completion of all models inference
    logging.info("Inference for the model completed")
    # Print the results list
    print(result)

if __name__ == "__main__":
    main()
