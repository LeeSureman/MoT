# MoT: Memory-of-Thought Enables ChatGPT to Self-Improve (EMNLP 2023)

The official repository for MoT. 
Check out our [paper](https://arxiv.org/pdf/2305.05181.pdf) for more information.




## Requirements

Python>=3.8
```
pip install torch==1.8.2+cu111 torchtext==0.9.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
pip install -r requirements.txt
```

## Datasets

Download the datasets from [Google Drive](https://drive.google.com/file/d/1UksTfFms_GyeLpFIFkXFnkCVv6lSrcxp/view?usp=sharing).

## Run Our Method
First, fill your account in to a text file, as follows:
```
[Email_1]----[Password_1]----[Openai_API_Key_1]
[Email_2]----[Password_2]----[Openai_API_Key_2]
...
```
Our code supports using multiple openai accounts simultaneously and call openai api in parallel. 

If you want to run the entire method:
```
# dataset:[aqua, drop, anli_a1, anli_a2, anli_a3, obqa, com_v, boolq, fact_checker, qa_wikidata]
dataset=[dataset] bash commands/run_mot_full.sh
```
Or you can directly download the memory of thoughts in pre-thinking, from [Google Drive](https://drive.google.com/file/d/1Rwm3PqGxL6x19oZoXFGso0mpdu7unie6/view?usp=sharing), and run the subsequent memory filtering and recalling:
```
# dataset:[aqua, drop, anli_a1, anli_a2, anli_a3, obqa, com_v, boolq, fact_checker, qa_wikidata]
dataset=[dataset] bash commands/run_mot_with_existing_memory.sh
```


## Citing MoT
```
@article{memory_of_thought,
  author       = {Xiaonan Li and
                  Xipeng Qiu},
  title        = {MoT: Memory-of-Thought Enables ChatGPT to Self-Improve},
  journal      = {CoRR},
  volume       = {abs/2305.05181},
  year         = {2023},
  url          = {https://doi.org/10.48550/arXiv.2305.05181},
  doi          = {10.48550/ARXIV.2305.05181},
  eprinttype    = {arXiv},
  eprint       = {2305.05181},
  timestamp    = {Fri, 12 May 2023 16:06:58 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2305-05181.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
