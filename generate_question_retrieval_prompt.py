import argparse
import copy

import jsonlines
import random
from openai_account_manager import call_openai_multi_thread, get_account_manager
from collections import Counter

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate_pair_fp', required=True)
    parser.add_argument('--num_retrieval_demos', required=True, type=int)
    parser.add_argument('--need_explanation', required=True, type=int)
    parser.add_argument('--use_retrieval_demos', required=True, type=int)
    parser.add_argument('--just_random_select', required=True, type=int)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--num_candidate_q', type=int, required=True)

    args = parser.parse_args()

    manager = get_account_manager(1)

    random.seed(args.seed)

    js_s = list(jsonlines.open(args.candidate_pair_fp))

    useful_js_s = []

    for i, js in enumerate(js_s):
        helpful_d = []
        unhelpful_d = []
        for j, d_dict in enumerate(js['demos']):
            if d_dict['helpful'] == 1:
                helpful_d.append(d_dict)
            else:
                unhelpful_d.append(d_dict)
        js['helpful_d'] = helpful_d
        js['unhelpful_d'] = unhelpful_d
        if len(helpful_d) > 0:
            useful_js_s.append(js)

    print('useful_js_s:{}'.format(len(useful_js_s)))

    # for js in useful_js_s:
    random.Random(42).shuffle(useful_js_s)

    js_s_for_prompting = random.Random(args.seed).sample(useful_js_s[:50], args.num_retrieval_demos)
    js_s_for_testing = useful_js_s[50:]

    js_s_for_testing = js_s_for_testing[:500]

    print('js_s_for_testing:{}'.format(len(js_s_for_testing)))

    if args.just_random_select:
        acc_list = []
        for seed in range(100):
            random.seed(seed)
            correct = 0
            for js in js_s_for_testing:
                tmp_selected_demo = random.choice(js['demos'][:args.num_candidate_q])
                tmp_selected_demo_correct = tmp_selected_demo['helpful']
                correct += tmp_selected_demo_correct

            acc_list.append(correct / len(js_s_for_testing))
            print('random accuracy:{}'.format(acc_list[-1]))

        print(sum(acc_list) / len(acc_list))

        exit()

    # 生成用来prompt retrieval的数据
    # for js in useful_js_s:

    retrieval_description = 'I will provide you with a target question and a list of reference questions. ' \
                            'I need you to choose a reference question from "Reference Questions",' \
                            'whose question, train of thought or answer would be most helpful for you to answer the target question. ' \
                            'Please note that the following reference questions are in random order without prioritization. '

    # retrieval_description = 'I need you to choose a reference question from "Reference Questions", ' \
                            # 'whose question, train of thought or answer would be most helpful for you to answer the target question.'
    # 'The following are some examples.'

    prompt_string = retrieval_description

    retrieval_demon_question_nums = [4, 5, 6, 8, 7]
    retrieval_demon_question_nums = random.sample(retrieval_demon_question_nums, args.num_retrieval_demos)

    retrieval_demonstration_s = []
    helpful_q_idx_s = []
    for i, js in enumerate(js_s_for_prompting):
        helpful_question = random.choice(js['helpful_d'])['question']
        print('{} helpful question:\n{}'.format(i, helpful_question))
        unhelpful_d = random.sample(js['unhelpful_d'], min(retrieval_demon_question_nums[i], len(js['unhelpful_d'])))
        unhelpful_q_s = []
        for d in unhelpful_d:
            unhelpful_q_s.append(d['question'])

        candidate_questions = copy.deepcopy(unhelpful_q_s)
        candidate_questions.append(helpful_question)
        random.shuffle(candidate_questions)
        candidate_questions = list(enumerate(candidate_questions))
        candidate_questions = list(map(list, candidate_questions))
        for tmp in candidate_questions:
            tmp[0] += 1

        for tmp in candidate_questions:
            if tmp[1] == helpful_question:
                helpful_q_idx = tmp[0]
                break

        # tmp_for_gpt_to_decode = 'I need you to choose a reference question from "Reference Questions", ' \
        # 'whose question, train of thought or answer would be most helpful in answering the target question.\n'
        tmp_retrieval_demonstration = ''

        tmp_retrieval_demonstration += 'Target Question:\n{}\n'.format(js['question'][3:-3])

        tmp_retrieval_demonstration += 'Reference Questions:'
        for idx, q in candidate_questions:
            tmp_retrieval_demonstration += '\n{}. {}'.format(idx, q[3:])

        tmp_retrieval_demonstration += '\n'
        tmp_retrieval_demonstration += 'What is the desirable reference question whose question, train of thought or answer ' \
                                       'would be most helpful for you to answer the target question? '
        tmp_retrieval_demonstration += 'The most helpful reference question would be question {}.'.format(helpful_q_idx)

        helpful_q_idx_s.append(helpful_q_idx)

        retrieval_demonstration_s.append(tmp_retrieval_demonstration)

    if args.need_explanation:
        print('start ask gpt for expalination of each seleceted question in retrieval demonstrations.')

        explanation_for_gpt_to_decode = []

        for retrieval_d in retrieval_demonstration_s:
            tmp_explanation_for_gpt_to_decode = retrieval_description + '\n' + retrieval_d + ' why?'

            explanation_for_gpt_to_decode.append(tmp_explanation_for_gpt_to_decode)

        gpt_hyper_parameter = {'model': 'gpt-3.5-turbo-0301', 'max_length': 256, 'n': 1, 'temperature': 0, 'top_p': 1}
        responses = call_openai_multi_thread(explanation_for_gpt_to_decode, [gpt_hyper_parameter], 10, 1)

        explanations = list(map(lambda x: x['choices'][0]['message']['content'], responses))

        for i in range(len(retrieval_demonstration_s)):
            retrieval_demonstration_s[i] = retrieval_demonstration_s[i] + ' ' + explanations[
                i] + ' So the desirable question is {}.'.format(helpful_q_idx_s[i])

        # retrieval_demonstration_s






    if 1:
        if args.num_retrieval_demos > 0:
            retrieval_prompt_with_demonstration = retrieval_description + ' The following are some examples.\n'
        else:
            retrieval_prompt_with_demonstration = retrieval_description + '\n'
            retrieval_prompt_with_demonstration = retrieval_prompt_with_demonstration + 'At the end of your response, '

        for retrieval_d in retrieval_demonstration_s:
            retrieval_prompt_with_demonstration = retrieval_prompt_with_demonstration + '\n\n' + retrieval_d

        print('*' * 100)
        print(retrieval_prompt_with_demonstration)
        print('*' * 100)

        for_gpt_to_decode = []

        for test_js in js_s_for_testing:
            tmp_test_inp = ''
            tmp_test_inp += 'Target Question:\n{}\n'.format(test_js['question'][3:-3])
            tmp_test_inp += 'Reference Questions:'
            # random.shuffle(test_js['demos'])
            for i, d_dict in enumerate(test_js['demos'][:args.num_candidate_q]):
                tmp_test_inp += '\n{}. {}'.format(i + 1, d_dict['question'][3:])
                # tmp_test_inp +=
            tmp_test_inp += '\n'
            # tmp_test_inp += 'What is the desirable reference question whose question, train of thought or answer ' \
            #                            'would be most helpful for you to answer the target question?'

            tmp_test_inp += '\nWhich one of the above reference questions is most relevant and similar for the target question? ' \
                            'You must choose exactly one reference question for helping you to answer the target question. ' \
                            'Your response must end in this format: "So, the most helpful question is question [index].". ' \
                            'For example, if question 5 is your answer, you must end in "So, the most helpful question is question 5."'
                            # 'You can analyze first then end in this format.'

            tmp_test_inp = retrieval_prompt_with_demonstration + '\n\n' + tmp_test_inp
            for_gpt_to_decode.append(tmp_test_inp)

        print('*' * 100)
        print(for_gpt_to_decode[0])
        print('*' * 100)

        # for tmp_test_inp in

        # for_gpt_to_decode=for_gpt_to_decode
        for_gpt_to_decode = for_gpt_to_decode
        gpt_hyper_parameter = {'model': 'gpt-3.5-turbo-0301', 'max_length': 256, 'n': 1, 'temperature': 0, 'top_p': 1}
        responses = call_openai_multi_thread(for_gpt_to_decode, [gpt_hyper_parameter], 100, 1)

        correct = 0
        parsing_error_num = 0
        correct_list = []
        pred_counter = Counter()
        for i, r in enumerate(responses):
            pred = r['choices'][0]['message']['content']
            print('the {} th retrieval test'.format(i))
            print('pred:\n{}'.format(pred))
            if args.need_explanation:
                pred = pred.split('desirable question is ')[-1]
                pred = pred[:-1]
                pass
            else:
                # pred = pred.split('The most helpful reference question would be question ')[-1]
                pred = pred.split('helpful question is question ')[-1]
                pred = pred[:-1]
            print('before parse:\n{}'.format(pred))

            try:
                pred = int(pred)
            except:
                print('pred parse error')
                parsing_error_num += 1
                pred = random.choice(list(range(10))) + 1

            pred_counter[pred] +=1
            tmp_correct = js_s_for_testing[i]['demos'][pred - 1]['helpful']
            correct_list.append(tmp_correct)
            print('after parse:')
            print(pred)

            correct += tmp_correct

        print('parsing_error_num:{}'.format(parsing_error_num))
        print('parsing_error_p:{}'.format(parsing_error_num / len(for_gpt_to_decode)))
        print('retrieval acc:{}'.format(correct / len(for_gpt_to_decode)))
        print('pred_counter:\n{}'.format(sorted(pred_counter.items(),key=lambda x:x[-1],reverse=True)))