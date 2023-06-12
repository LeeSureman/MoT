# ChatGPT can Self-Improve via Memory-of-Thought
This repository contains the code and relevant resources, e.g., the generated memory-of-thought, for our paper: [MoT: Pre-thinking and Recalling Enable ChatGPT to Self-Improve with Memory-of-Thoughts](https://arxiv.org/pdf/2305.05181)

For now, we provide the generated memory-of-thought pool. The full code will come in the later months of the year.

## Generated Memory-of-Thought Download
You can download the files of the generated thoughts in [google drive](https://drive.google.com/file/d/1Rwm3PqGxL6x19oZoXFGso0mpdu7unie6/view?usp=sharing)

The structure of one line of the memory-of-thought files is as:
- question: *the original question of the example*
- gold_ans: *the ground truth of the example*
- pred_ans: *the final answer from majority voting*
- wrap_question: *the full question (including few-shot demonstration and template) sent to Openai API*
- response: *the original response from Openai API*
    - choices:
        - tmp_pred: *the prediction of this sampled reasoning path*
        - 
- response_set_size: *the number of unique Openai API response*
- pred_count: *the counting of prediction*
- pred_set_size: *the number of unique prediction*

### How to read the file in your code
```python
import jsonlines
js_s = list(jsonlines.open('mot_generated_thoughts/[DATASET_NAME]','r'))
# the first example's first response
print(js_s[0]['response']['choices'][0]['message']['content'])
```