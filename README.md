# vLLM_ModelCoverageTest
A tool of model coverage test for vLLM

# Steps

## Start the docker container

Here we use `rocm/vllm-dev:20250112` as example. You should refer the commands bellow to start the vLLM container.

```bash
docker run rocm/vllm-dev:20250112

docker run -it --rm --network=host --device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 32G --hostname=vLLM-CT -v $PWD:/ws -w /ws rocm/vllm-dev:20250112 /bin/bash
```

## Run in the container

Login with your HF account for model downloading
```bash
huggingface-cli login
```

### Clone this repo

```bash
git clone https://github.com/alexhegit/vLLM_ModelCoverageTest.git
cd vLLM_ModelCoverageTest
```

### Test

Quick Test

```bash
python3 ModelCoverageTest.py --csv model_list.csv
```

The test finished with two output files.
- Add the new column 'status' with 'PASS' or 'FAILED' output from the compatiable test
- mct-[yyyymmdd].log for checking the detail of the test expecial for the error.


Get help for the tool usage,

```bash
python ModelCoverageTest.py --help

usage: ModelCoverageTest.py [-h] [--csv CSV]

Test models with specified prompts from a CSV file.

options:
  -h, --help  show this help message and exit
  --csv CSV   Path to the CSV file containing model_id and gpus
```

Here a example csv file defined the model list with model id(of Huggingface) and the GPU ID used for test.

```bash
# cat model_list.csv
model_id,gpus
facebook/opt-125m,"0"
tiiuae/falcon-7b,"0"
google/flan-t5-small,"0"
openai-community/gpt2,"0"
```
Example CSV of MTP
```
model_id,gpus
facebook/opt-125m,"0,1"
```

There are to output file for checking the test results.
- [modle_list]_results.csv
```
# cat model_list_results.csv
model_id,gpus,status
facebook/opt-125m,"0",PASS
...
```
- mt-yyyymmdd.log
There is a prefix <vLLM-CMT> for quick filter out the log print by vLLM MCT. You should check the what happend in detail for the model run FAILED.

# NOTES & FAQ
1. Some model like LLama need to request access at first. You may check the error from the log if not have.
2. You should try multiple tensor parallel if LLM is OOM with single GPU.
3. Some model may run PASS with tp=1 but may failed with mulitplel tp. You clould use `vllm serve` test it for double confirm.
4. The vLMM CMT will follow the process [download LLM | inference LLM | delete LLM] to save disk space and avoid run failed cuased by the continue enlarged .cache/huggingface/hub directory.
