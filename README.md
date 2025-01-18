# vLLM_ModelCoverageTest
A tool to do the model coverage test for vLLM

# Steps

## Start the docker container

Here we use `rocm/vllm-dev:20250112` as example. You should refer the commands bellow to start the vLLM container.

```bash
docker run rocm/vllm-dev:20250112

docker run -it --rm --network=host --device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 32G --hostname=vLLM-CT -v $PWD:/ws -w /ws rocm/vllm-dev:20250112 /bin/bash
```

## Run in the container

Export your HF Token for downloading models from Huggingface

```bash
export HF_TOKEN=hf_***
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
- model_list-[yyyymmdd].log for checking the detail of the test expecial for the error.


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
