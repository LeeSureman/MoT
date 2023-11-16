import sys
sys.path.append('.')
import argparse
import os

import jsonlines
import functools
import tqdm
from collections import Counter
import re
import string
from evaluations.drop_f1 import pred_to_many_f1_metrics, pred_to_one_answer_f1_metrics


def remove_line_break(s):
    s = s.strip().split('\n')
    s = list(map(lambda x:x.strip(),s))
    s = list(filter(lambda x: len(x) > 0, s))
    s = list(map(lambda x: x + '.' if (x[-1] not in string.punctuation) else x, s))
    s = ' '.join(s)
    return s

def make_a_demo_dict(js, correct_r):
    question = js['question']
    gold_ans = js['gold_ans']
    pred_ans = correct_r['tmp_pred']
    content_from_lm = remove_line_break(correct_r['message']['content'])

    result = {}
    result['question'] = question
    result['gold_ans'] = gold_ans
    result['pred_ans'] = pred_ans
    result['demo'] = question + ' ' + content_from_lm

    return result



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp_fp', required=True)
    parser.add_argument('--filtering_criteria', choices=['gt', 'confidence', 'none'], required=True)
    parser.add_argument('--out_fp', )
    parser.add_argument('--confidence_threshold', type=float)
    # here confidence is the number of the choosed argmax path

    args = parser.parse_args()

    dataset = re.findall(r'strategyqa_small|commonsensqa|openbookqa|aqua|gsm8k|anli_a3|anli_a2|anli_a1|hotpot_qa|qa_wikidata|drop|com_v|com_e|fact_checker|boolq|elementary_math_qa',args.inp_fp)
    assert len(dataset) == 1
    dataset = dataset[0]

    os.makedirs('./demos/filter_by_{}'.format(args.filtering_criteria),exist_ok=True)
    args.out_fp = './demos/filter_by_{}/{}.jsonl'.format(args.filtering_criteria,dataset)


    js_s = list(jsonlines.open(args.inp_fp))

    print('input_js_num:{}'.format(len(js_s)))
    # print()

    # js_s = list(js_f)
    results_demos = []
    if args.filtering_criteria == 'gt':
        for js in tqdm.tqdm(js_s):
            gt = js['gold_ans']
            correct_r = None
            for r in js['response']['choices']:
                if r['tmp_pred'] == gt:
                    correct_r = r
                    break
            if correct_r != None:
                tmp_demo = make_a_demo_dict(js, correct_r)
                results_demos.append(tmp_demo)

    elif args.filtering_criteria == 'confidence':
        for i,js in enumerate(tqdm.tqdm(js_s)):
            confident_paths_num = int(args.confidence_threshold * len(js['response']['choices']))
            # print('confident_paths_num: {}'.format(confident_paths_num))
            if i == 0:
                print('confident_paths_num:{}'.format(confident_paths_num))
            gt = js['gold_ans']
            pred_counter = Counter()
            js['response']['choices'] = list(filter(lambda x:'The answer is ' in x['message']['content'], js['response']['choices']))
            for r in js['response']['choices']:
                # print(repr(r['tmp_pred']))
                if type(r['tmp_pred']) is list:
                    r['tmp_pred'] = tuple(r['tmp_pred'])
                pred_counter[r['tmp_pred']] += 1

            pred_counter = list(sorted(pred_counter.items(),key=lambda x:x[-1],reverse=True))
            if len(pred_counter) == 0:
                continue
            if i<20:
                print('{}:{}'.format(i,pred_counter[0][-1]))
            if pred_counter[0][-1] >= confident_paths_num:
                argmax_pred = pred_counter[0][0]
                correct_r = None
                for r in js['response']['choices']:
                    if r['tmp_pred'] == argmax_pred:
                        correct_r = r
                        break

                tmp_demo = make_a_demo_dict(js, correct_r)
                results_demos.append(tmp_demo)
                if i<20:
                    print('add {} th training example'.format(i))
            else:
                pass




    else:
        raise NotImplementedError

    print('total : {}'.format(len(js_s)))
    print('after filtering : {}'.format(len(results_demos)))
    print('filter / total : {}'.format(len(results_demos) / len(js_s)))

    if args.filtering_criteria != 'gt':
        print('*'*50)
        print('start calculating the correctness of resulting demos')
        if dataset == 'drop':
            em_list_single_gold = []
            f1_list_single_gold = []
            em_list_multiple_gold = []
            f1_list_multiple_gold = []
            for d in tqdm.tqdm(results_demos):
                y = d['gold_ans']
                pred = d['pred_ans']
                tmp_exact_match_single_gold, tmp_f1_single_gold = pred_to_one_answer_f1_metrics(pred, y[0],
                                                                                                numerically_strict=1)
                tmp_exact_match_multiple_gold, tmp_f1_multiple_gold = pred_to_many_f1_metrics(pred, y, numerically_strict=1)
                em_list_single_gold.append(tmp_exact_match_single_gold)
                f1_list_single_gold.append(tmp_f1_single_gold)
                em_list_multiple_gold.append(tmp_exact_match_multiple_gold)
                f1_list_multiple_gold.append(tmp_f1_multiple_gold)

            total = len(results_demos)
            total_example_number = total
            em_single_gold = (sum(em_list_single_gold) * 1.0 / total) * 100
            f1_single_gold = (sum(f1_list_single_gold) * 1.0 / total) * 100

            em_multiple_gold = (sum(em_list_multiple_gold) * 1.0 / total) * 100
            f1_multiple_gold = (sum(f1_list_multiple_gold) * 1.0 / total) * 100

            print('{}/{} exact match single gold: {}'.format(total, total_example_number, em_single_gold))
            print('{}/{} f1 single gold: {}'.format(total, total_example_number, f1_single_gold))
            print('')
            print('{}/{} exact match multiple gold: {}'.format(total, total_example_number, em_multiple_gold))
            print('{}/{} f1 multiple gold: {}'.format(total, total_example_number, f1_multiple_gold))

        elif dataset in ['hotpot_qa', 'qa_wikidata']:
            em_list_single_gold = []
            f1_list_single_gold = []
            for d in tqdm.tqdm(results_demos):
                y = d['gold_ans']
                pred = d['pred_ans']
                tmp_exact_match_single_gold, tmp_f1_single_gold = pred_to_one_answer_f1_metrics(pred, y,
                                                                                                numerically_strict=0)
                em_list_single_gold.append(tmp_exact_match_single_gold)
                f1_list_single_gold.append(tmp_f1_single_gold)


            total = len(results_demos)
            total_example_number = total
            em_single_gold = (sum(em_list_single_gold) * 1.0 / total) * 100
            f1_single_gold = (sum(f1_list_single_gold) * 1.0 / total) * 100

            print('{}/{} exact match single gold: {}'.format(total, total_example_number, em_single_gold))
            print('{}/{} f1 single gold: {}'.format(total, total_example_number, f1_single_gold))

        else:
            correct_num = 0
            for d in tqdm.tqdm(results_demos):
                if d['pred_ans'] == d['gold_ans']:
                    correct_num += 1
            print('total demos : {}'.format(len(results_demos)))
            print('correct demos : {}'.format(correct_num))
            print('acc : {}'.format(correct_num / len(results_demos)))

    print('args.out_fp:{}'.format(args.out_fp))

    if args.out_fp != None:
        with jsonlines.open(args.out_fp,mode='w') as writer:
            for d in results_demos:
                writer.write(d)
    # print(len(results_demos))
