import tqdm

from openai_account_manager import call_openai_multi_thread
import random
import logging
from collections import Counter
from transformers import AutoTokenizer
import jsonlines

logger = logging.getLogger(__name__)


def retrieve_demos_by_lm(demos_group_s_for_gpt_to_decode, hyper_parameter, num_threads, use_tqdm,
                         demos_for_retrieval_using_purely_question, shuffle_demos_in_query, format_requirement_at_last):
    '''

    :param demos_group_for_gpt_to_decode:
    :param hyper_parameter:
    :param num_threads:
    :param use_tqdm:
    :param retrieval_answer_trigger:
    :param retrieval_description: 在开头对所需要LM检索的任务的描述
    :param retrieval_demos: 检索的示例，可以没有
    :param demos_for_retrieval_using_purely_question: 每个检索示例中（包括要LM去找的）里的参考示例是要用只有问题，还是【问题，思路，答案】对
    :param shuffle_demos_in_query: 要不要打乱待检索的demo的顺序
    :return:
    '''
    tokenizer = AutoTokenizer.from_pretrained('gpt2-large')
    logger.info('shuffle_demos_in_query:{}'.format(shuffle_demos_in_query))
    demos_group_for_gpt_to_decode_flat = []
    for demos_group_and_target_q in demos_group_s_for_gpt_to_decode:
        demos_groups, target_q = demos_group_and_target_q
        for demos in demos_groups:
            demos_group_for_gpt_to_decode_flat.append([demos, target_q])
        # demos_group_for_gpt_to_decode_flat.extend(demos_group)

    retrieval_candidate_size = len(demos_group_for_gpt_to_decode_flat[0][0])
    num_demo_for_every_target_q = len(demos_group_s_for_gpt_to_decode[0][0])

    def transform_demos_into_retrieval_list_prompt(demos, helpful_demo_idx=None):
        if helpful_demo_idx is not None:
            helpful_demo = demos[helpful_demo_idx]['demonstration']

        if shuffle_demos_in_query:
            random.shuffle(demos)
            if helpful_demo_idx is not None:
                for i, tmp_js in enumerate(demos):
                    if helpful_demo == tmp_js['demonstration']:
                        helpful_demo_idx_now = i
                        break
            else:
                helpful_demo_idx_now = helpful_demo_idx
        else:
            helpful_demo_idx_now = helpful_demo_idx

        result_retrieval_list_prompt = ""
        for i, d_dict in enumerate(demos):
            if demos_for_retrieval_using_purely_question:
                result_retrieval_list_prompt = result_retrieval_list_prompt + '{}.\n'.format(i + 1) + d_dict[
                    'question'] + '\n\n'
            else:
                result_retrieval_list_prompt = result_retrieval_list_prompt + '{}.\n'.format(i + 1) + d_dict[
                    'demonstration'] + '\n\n'

        return result_retrieval_list_prompt, helpful_demo_idx_now

    if demos_for_retrieval_using_purely_question:

        task_description = 'I will provide you with a target question and {} reference questions. ' \
                           'I need you to choose a reference question from "Reference Questions", whose question, train of thought or answer ' \
                           'would be most helpful for you to answer the target question. ' \
                           'Please note that the following reference QA pairs are presented in a random order without any prioritization.'
    else:
        task_description = 'I will provide you with a target question and {} reference QA pairs. ' \
                           'I need you to choose a reference QA pair from "Reference QA pairs", ' \
                           'which would be most helpful for you to answer the target question. ' \
                           'Please note that the following reference QA pairs are presented in a random order without any prioritization.'

    task_description = task_description.format(retrieval_candidate_size)

    final_for_gpt_to_decode_flat = []

    actual_num_demos_for_retrieval_list = []

    for demos_and_target_q in tqdm.tqdm(demos_group_for_gpt_to_decode_flat):

        demos, target_q = demos_and_target_q

        tmp_result_retrieval_prompt = ''

        target_q = 'Target Question:\n{}'.format(target_q)
        while 1:
            demo_list_for_retrieval, helpful_demo_idx_now = transform_demos_into_retrieval_list_prompt(demos, )
            # logger.info('tokenizer.tokenize(demo_list_for_retrieval):{}'.format(len(tokenizer.tokenize(demo_list_for_retrieval))))
            # actual_num_demos_for_retrieval_list.append(len(demos))
            # logger.info('len(tokenizer.tokenize(demo_list_for_retrieval +target_q)) :{}'.format(
            #     len(tokenizer.tokenize(demo_list_for_retrieval + target_q))))
            if len(tokenizer.tokenize(demo_list_for_retrieval + target_q)) < 3300:
                actual_num_demos_for_retrieval_list.append(len(demos))
                break
            else:
                if len(demos) > 0:
                    demos = demos[:-1]
                else:
                    break

        if demos_for_retrieval_using_purely_question:
            demo_list_for_retrieval = 'Reference Questions:\n{}'.format(demo_list_for_retrieval)
        else:
            demo_list_for_retrieval = 'Reference QA pairs:\n{}'.format(demo_list_for_retrieval)

        if demos_for_retrieval_using_purely_question:
            final_require_for_retrieval = 'Which one of the above reference questions is the most helpful question for you to answer the target question? ' \
                                          'You must choose exactly one reference question to you answer the target question.'
        else:
            final_require_for_retrieval = 'Which one of the above reference QA pairs is the most helpful QA pair for you to answer the target question? ' \
                                          'You must choose exactly one reference QA pair to you answer the target question.'

        if demos_for_retrieval_using_purely_question:
            format_requirement = 'Your response must end in this format: "The most helpful question is question [index].". ' \
                                 'For example, if question 5 is your answer, you must end in "The most helpful question is question 5."'
        else:
            format_requirement = 'Your response must end in this format: "The most helpful QA pair is QA [index].". ' \
                                 'For example, if QA 5 is your answer, you must end in "The most helpful QA pair is QA 5."'

        if not format_requirement_at_last:
            task_description = task_description + ' ' + format_requirement
        else:
            final_require_for_retrieval = final_require_for_retrieval + ' ' + format_requirement

        tmp_result_retrieval_prompt = tmp_result_retrieval_prompt + task_description

        tmp_result_retrieval_prompt = tmp_result_retrieval_prompt + '\n\n' + target_q

        tmp_result_retrieval_prompt = tmp_result_retrieval_prompt + '\n\n' + demo_list_for_retrieval

        tmp_result_retrieval_prompt = tmp_result_retrieval_prompt + '\n\n' + final_require_for_retrieval

        final_for_gpt_to_decode_flat.append(tmp_result_retrieval_prompt)

    retrieved_demos_flat = []

    assert hyper_parameter['n'] == 1

    logger.info('actual_num_demos_for_retrieval list_len : {}'.format(len(actual_num_demos_for_retrieval_list)))
    logger.info('actual_num_demos_for_retrieval_avg : {}'.format(
        sum(actual_num_demos_for_retrieval_list) / len(actual_num_demos_for_retrieval_list)))

    tmp_example_out_f = jsonlines.open('tmp_lm_retrieval_example.jsonl','w')

    for tmp in final_for_gpt_to_decode_flat:
        tmp_example_out_f.write(tmp)

    # exit()

    responses = call_openai_multi_thread(final_for_gpt_to_decode_flat, [hyper_parameter], num_threads, use_tqdm)

    parsing_error_num = 0
    retrieved_demo_idx_counter = Counter()
    for i, r_dict in enumerate(responses):
        demos = demos_group_for_gpt_to_decode_flat[i][0]
        r_content = r_dict['choices'][0]['message']['content']
        if i < 5:
            logger.info('retrieval response content {} : {}'.format(i, r_content))

        if demos_for_retrieval_using_purely_question:
            tmp_demo_idx = r_content.split('helpful question is question ')[-1]
        else:
            tmp_demo_idx = r_content.split('helpful QA pair is QA ')[-1]
        tmp_demo_idx = tmp_demo_idx[:-1]

        try:
            tmp_demo_idx = int(tmp_demo_idx)
            retrieved_demo_idx_counter[tmp_demo_idx] += 1
            tmp_demo_idx -= 1
            tmp_demo = demos[tmp_demo_idx]
        except:
            parsing_error_num += 1
            tmp_demo_idx = 1
            tmp_demo = demos[tmp_demo_idx]


        retrieved_demos_flat.append(tmp_demo)

    retrieved_demos = []

    for i in range(len(demos_group_s_for_gpt_to_decode)):
        retrieved_demos.append(
            retrieved_demos_flat[i * num_demo_for_every_target_q:(i + 1) * num_demo_for_every_target_q])
    logger.info('parsing_error_num : {}'.format(parsing_error_num))
    logger.info('parsing_error_p : {}'.format(parsing_error_num / len(responses)))
    logger.info('actual_num_demos_for_retrieval_avg : {}'.format(
        sum(actual_num_demos_for_retrieval_list) / len(actual_num_demos_for_retrieval_list)))
    logger.info('retrieved_demo_idx_distribution:{}'.format(
        sorted(retrieved_demo_idx_counter.items(), key=lambda x: x[-1], reverse=True)))
    return {'retrieved_demos': retrieved_demos, 'parsing_error_p': parsing_error_num / len(responses),
            'actual_num_demos_for_retrieval_avg':sum(actual_num_demos_for_retrieval_list) / len(actual_num_demos_for_retrieval_list)}

    # return final_for_gpt_to_decode_flat
