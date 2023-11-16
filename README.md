# MoT: Memory-of-Thought Enables ChatGPT to Self-Improve (EMNLP 2023)

[//]: # ([![Open Auto-CoT in Colab]&#40;https://colab.research.google.com/assets/colab-badge.svg&#41;]&#40;https://colab.research.google.com/github/amazon-science/auto-cot/blob/main/try_cot_colab.ipynb&#41;)

[//]: # (Cheer AI up with the "let's think step by step" prompt? More plz. *Let’s think not just step by step, but also one by one.*)

[//]: # ()
[//]: # (Auto-CoT uses more cheers & diversity to SAVE huge manual efforts in chain of thought prompt design, matching or even exceeding performance of manual design on GPT-3.)

The official repository for MoT
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

If you want run the entire method:
```
# dataset:[aqua, drop, anli_a1, anli_a2, anli_a3, obqa, com_v, boolq, fact_checker, qa_wikidata]
dataset=[dataset] bash commands/run_mot_full.sh
```

Or you can download the memory of thoughts in pre-thinking, from [Google Drive](https://drive.google.com/file/d/1Rwm3PqGxL6x19oZoXFGso0mpdu7unie6/view?usp=sharing), and run
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
