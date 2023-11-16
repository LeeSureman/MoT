import argparse
import jsonlines
import torch
from InstructorEmbedding import INSTRUCTOR
import numpy as np
import re
from utils import generate_one_demo_prompt
from openai_account_manager import call_openai_multi_thread


def answer_cleansing(dataset, pred, must_choice=False, verbose=True):
    direct_answer_trigger_for_fewshot = "The answer is"
    if verbose:
        print("pred_before : " + pred)

    preds = pred.split(direct_answer_trigger_for_fewshot)
    answer_flag = True if len(preds) > 1 else False
    pred = preds[-1]

    if dataset in ("aqua", "commonsensqa", 'openbookqa'):
        pred = re.findall(r'A|B|C|D|E', pred)
    elif dataset == "bigbench_date":
        pred = re.findall(r'A|B|C|D|E|F', pred)
    elif dataset in ("object_tracking"):
        pred = re.findall(r'A|B|C', pred)
    elif dataset in ("gsm8k", "addsub", "multiarith", "svamp", "singleeq"):
        if must_choice:
            pred = re.findall(r'A|B|C|D', pred)
        else:
            pred = pred.replace(",", "")
            pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]
    elif dataset in ("strategyqa", "coin_flip", "strategyqa_small"):
        pred = pred.lower()
        pred = re.sub("\"|\'|\n|\.|\s|\:|\,", " ", pred)
        pred = pred.split(" ")
        pred = [i for i in pred if i in ("yes", "no")]
    elif dataset == "last_letters":
        pred = re.sub("\"|\'|\n|\.|\s", "", pred)
        pred = [pred]
    elif 'nli' in dataset:
        pred = re.findall(r'yes|no|it is not possible to tell', pred)
    else:
        raise ValueError("dataset is not properly defined ...")

    # If there is no candidate in list, null is set.
    if len(pred) == 0:
        pred = ""
    else:
        if answer_flag:
            # choose the first element in list ...
            pred = pred[0]
        else:
            # choose the last element in list ...
            pred = pred[-1]

    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred != "":
        if pred[-1] == ".":
            pred = pred[:-1]
    if verbose:
        print("pred_after : " + pred)

    if len(pred) == 0:
        pred = 'answer_parsing_error'

    return pred


# def make_result_candidate_pair()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lm_inference_result_fp', required=True)
    parser.add_argument('--retriever_name', default='hkunlp/instructor-large')

    args = parser.parse_args()

    dataset = re.findall(r'strategyqa_small|commonsensqa|openbookqa|aqua|gsm8k|anli_a3|anli_a2',
                         args.lm_inference_result_fp)[0]

    js_s = list(jsonlines.open(args.lm_inference_result_fp))

    question_pool = list(map(lambda x: x['question'], js_s))

    q_retrieval_instruction = 'Represent the question for retrieving duplicate questions: '

    instruction_plus_test_questions = list(
        map(lambda x: [q_retrieval_instruction, x], question_pool))

    print('start sencoding demo questions')
    retriever = INSTRUCTOR(args.retriever_name)
    pool = retriever.start_multi_process_pool()
    demo_embeddings = retriever.encode_multi_process(instruction_plus_test_questions, pool)
    retriever.stop_multi_process_pool(pool)
    scores = np.matmul(demo_embeddings, demo_embeddings.T)

    print('end sencoding demo questions')

    result = []

    num_candidates = 15

    for i, js in enumerate(js_s):
        # q = js['question']
        if js['pred_ans'] == 'answer_parsing_error':
            continue
        if dataset == 'gsm8k':
            if js['pred_ans'] == js['gold_ans'] or float(js['gold_ans'].replace(',', '')) == float(
                    js['pred_ans'].replace(',', '')):
                continue

        else:
            if js['pred_ans'] == js['gold_ans']:
                continue

        _, demo_idxs = torch.topk(torch.from_numpy(scores[i]), k=80)
        demo_idxs = demo_idxs.tolist()
        demo_idxs = list(filter(lambda x: x != i, demo_idxs))

        correct_demos = []

        for tmp_idx in demo_idxs:
            for j, r in enumerate(js_s[tmp_idx]['response']['choices']):
                if r['tmp_pred'] == js_s[tmp_idx]['gold_ans']:
                    correct_demos.append({'demonstration': js_s[tmp_idx]['question'] + ' ' + r['message']['content'],
                                          'question': js_s[tmp_idx]['question'][:-3]})
                    break
            if len(correct_demos) >= num_candidates:
                break
        tmp_candidate_pair = {}
        tmp_candidate_pair['question'] = js['question']
        tmp_candidate_pair['pred_ans'] = js['pred_ans']
        tmp_candidate_pair['gold_ans'] = js['gold_ans']
        tmp_candidate_pair['demos'] = correct_demos

        result.append(tmp_candidate_pair)

    print(len(result))
    result = list(filter(lambda x: len(x['demos']) == num_candidates, result))
    print(len(result))
    out_js_f = jsonlines.open('tmp_candidate_pairs_for_retrieval_prompting/{}.jsonl'.format(dataset), 'w')
    for tmp in result:
        out_js_f.write(tmp)
    # if js_s[tmp_idx]['pred_ans'] == js_s[tmp_idx]['gold_ans']:

    # exit()

    # result = result[:50]
    tmp_for_gpt_to_decode = []

    for i, tmp_candidate_pair in enumerate(result):
        q = tmp_candidate_pair['question']
        demos = tmp_candidate_pair['demos']
        for j, demo_dict in enumerate(demos):
            tmp_candidate_pair_prompt = generate_one_demo_prompt(demo_dict, q)
            tmp_for_gpt_to_decode.append(tmp_candidate_pair_prompt)

    gpt_hyper_parameter = {'model': 'gpt-3.5-turbo-0301', 'max_length': 128, 'n': 1, 'temperature': 0, 'top_p': 1}
    responses = call_openai_multi_thread(tmp_for_gpt_to_decode, [gpt_hyper_parameter], 100, 1)

    responses_reshaped = []

    for i in range(len(result)):
        responses_reshaped.append(responses[i * num_candidates: (i + 1) * num_candidates])

    for i, tmp_candidate_pair in enumerate(result):
        assert len(tmp_candidate_pair['demos']) == len(responses_reshaped[i])
        # tmp_candidate_pair['demos'] = list(zip(tmp_candidate_pair['demos'], responses_reshaped))

        for j, demo in enumerate(tmp_candidate_pair['demos']):
            pred = answer_cleansing(dataset, responses_reshaped[i][j]['choices'][0]['message']['content'])
            tmp_candidate_pair['demos'][j]['pred_using_this_demo'] = pred
            tmp_candidate_pair['demos'][j]['helpful'] = int((pred == tmp_candidate_pair['gold_ans']))

    out_js_f = jsonlines.open('tmp_candidate_pairs_for_retrieval_prompting/{}_labeled.jsonl'.format(dataset), 'w')
    for tmp in result:
        out_js_f.write(tmp)

    # for j, (demo, response) in enumerate(tmp_candidate_pair['demos']):
    #     pred = answer_cleansing(dataset, response['choice'][0]['message']['content'])
    #     tmp_candidate_pair['demos']
