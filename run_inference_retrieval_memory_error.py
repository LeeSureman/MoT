'''
Adapted from https://github.com/kojima-takeshi188/zero_shot_cot
'''

import argparse
import os
import random

import torch

from utils import *
from sentence_transformers import SentenceTransformer
from InstructorEmbedding import INSTRUCTOR
import fitlog
from collections import Counter
import tqdm

os.makedirs('fitlog_dir', exist_ok=True)
fitlog.set_log_dir('fitlog_dir')
from utils import get_kmeans_clustered_idx
import time
from multi_thread_openai_api_call import MyThread
from openai_account_manager import get_account_manager, OpenAI_API_inp_Manager_MultiThread, call_openai_multi_thread
from transformers import AutoTokenizer
from data_process_utils import extract_premise_and_hypothesis
from lm_retrieval import retrieve_demos_by_lm
from fastNLP import cache_results
from evaluations.drop_f1 import pred_to_many_f1_metrics, pred_to_one_answer_f1_metrics

nli_dataset = ['anli_a2', 'anli_a3', 'anli_a1']

answer_format_prompt = "\nYour response must end with the format \"The answer is ...\". " \
                       "If my question is a multi-choice question and the answer is A, your response must end with \"The answer is A.\"" \
                       " If the answer is Bob, your response must end with \"The answer is Bob.\""
plan_prompt = "\nLet's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan to solve the problem step by step."
zero_shot_plan_cot_prompt = answer_format_prompt + plan_prompt

def main():
    args = parse_arguments()
    print('*****************************')
    print(args)
    print('*****************************')

    task = args.dataset
    if task == "aqua" or task == "last_letters":
        num_demo = 4
    elif task == "commonsensqa":
        num_demo = 7
    elif task in ["strategyqa", "strategyqa_small"]:
        num_demo = 6
    else:
        num_demo = 8

    if args.num_demo > 0:
        num_demo = args.num_demo

    # for k,v in args.__dict__.items():
    #     print('{}:{}'.format(k,type(v)))
    # exit()
    logger.info('fitlog.add_hyper start')
    logger.info('args:\n{}'.format(args))
    fitlog.add_hyper(args)
    logger.info('fitlog.add_hyper end')
    fitlog.add_best_metric({'tmp': 1})

    logger.info('num_demo:{}'.format(num_demo))
    logger.info('num_cluster:{}'.format(args.clustered_retrieval))

    fix_seed(args.random_seed)

    # print("OPENAI_API_KEY:")
    # manager = get_manager()
    # print(manager.[0:15] + '**********')

    # Initialize decoder class (load model and tokenizer) ...
    decoder = Decoder()

    print("setup data loader ...")
    dataloader = setup_data_loader(args)
    dataset_in_loader = dataloader.dataset
    print_now()

    if args.method == "few_shot":
        demo = create_demo_text(args, cot_flag=False)
    elif args.method == "few_shot_cot" or args.method == "auto_cot":
        demo = create_demo_text(args, cot_flag=True)
    elif args.method in ['my_random_sample_few_shot_cot', 'my_random_sample_few_shot']:
        if args.demo_pool_from == 'gt':
            demo_pool = load_gt_demo_pool(args.dataset, args.direct_answer_trigger_for_fewshot, 'cot' in args.method)
        else:
            demo_pool = load_lm_inference_demo_pool(args.demo_pool_path, 'cot' in args.method)

        sampled_demos = random.Random(args.demo_sampling_seed).sample(demo_pool, num_demo)
        sampled_demos = list(map(lambda x: x['demonstration'], sampled_demos))
        demo = concat_demos(sampled_demos)

    else:
        pass

    if 'retrieval' in args.method:
        if args.demo_pool_from == 'gt':
            demo_pool = load_gt_demo_pool(args.dataset, args.direct_answer_trigger_for_fewshot, 'cot' in args.method)
        else:
            demo_pool = load_lm_inference_demo_pool(args.demo_pool_path, 'cot' in args.method)

        # logger.info('demo_pool[0]')
        # logger.info(demo_pool[0])
        #
        # logger.info('demo_pool[1]')
        # logger.info(demo_pool[1])
        #
        # logger.info('demo_pool[2]')
        # logger.info(demo_pool[2])
        # exit()

        if args.retrieval_hybrid_with_task_demos == 'random_sample_from_demo_pool':
            sampled_demos = random.Random(args.demo_sampling_seed).sample(demo_pool, num_demo)
            sampled_demos = list(map(lambda x: x['demonstration'], sampled_demos))
            task_level_demo = concat_demos(sampled_demos)
        elif args.retrieval_hybrid_with_task_demos == 'manual':
            task_level_demo = create_demo_text(args, cot_flag=True)
        elif args.retrieval_hybrid_with_task_demos is None or args.retrieval_hybrid_with_task_demos.lower() == 'none':
            pass
        else:
            raise NotImplementedError

        premise_to_demo_idxs_dict = {}
        premise_list = []
        if args.dataset in nli_dataset:
            for idx, d in enumerate(tqdm.tqdm(demo_pool, desc='building premise index')):
                premise = extract_premise_and_hypothesis(d['question'])['premise']
                premise_list.append(premise)
                if premise in premise_to_demo_idxs_dict:
                    premise_to_demo_idxs_dict[premise].append(idx)
                else:
                    premise_to_demo_idxs_dict[premise] = [idx]
        logger.info('nli demo pool\'s premises: {}'.format(len(premise_to_demo_idxs_dict)))

        if 'instructor' not in args.retriever_name:
            retriever = SentenceTransformer(args.retriever_name)
        else:
            retriever = INSTRUCTOR(args.retriever_name)

        if args.query_encoding == 'x' and args.demo_encoding == 'x':
            q_retrieval_instruction = 'Represent the question for retrieving duplicate questions: '
            d_retrieval_instruction = 'Represent the question for retrieving duplicate questions: '
        elif args.query_encoding == 'z' and args.demo_encoding == 'z':
            q_retrieval_instruction = 'Represent the rationale for retrieving duplicate rationales: '
            d_retrieval_instruction = 'Represent the rationale for retrieving duplicate rationales: '
        elif args.query_encoding == 'x' and args.demo_encoding == 'z':
            q_retrieval_instruction = 'Represent the question for retrieving relevant rationales: '
            d_retrieval_instruction = 'Represent the rationale for retrieval: '
        else:
            logger.info(
                'Invalid [query_encoding:{} and demo_encoding:{}]'.format(args.query_encoding, args.demo_encoding))
            raise NotImplementedError

        logger.info('q_retrieval_instruction:{}'.format(q_retrieval_instruction))
        logger.info('d_retrieval_instruction:{}'.format(d_retrieval_instruction))

        # assert args.query_encoding == 'x'
        # assert args.demo_encoding == 'x'

        if args.demo_encoding == 'x':
            demo_text_to_encode_pool = list(map(lambda x: x['question'], demo_pool))
        elif args.demo_encoding == 'z':
            demo_text_to_encode_pool = list(map(lambda x: x['rationale'], demo_pool))
        else:
            raise NotImplementedError

        if 'instructor' in args.retriever_name:
            instruction_plus_demo_text = list(
                map(lambda x: [d_retrieval_instruction, x], demo_text_to_encode_pool))
        else:
            instruction_plus_demo_text = demo_text_to_encode_pool

        pool = retriever.start_multi_process_pool()
        os.makedirs('embeddings_caches', exist_ok=True)

        @cache_results(_cache_fp='./embeddings_caches/demo_embedding_{}_{}_{}_{}_{}_{}'.format(args.dataset,
                                                                                               args.retriever_name.replace(
                                                                                                   '/', '_'),
                                                                                               args.demo_encoding,
                                                                                               args.query_encoding,
                                                                                               args.demo_pool_path.replace(
                                                                                                   '/', '_'),
                                                                                               args.demo_pool_from),
                       _refresh=False)
        def get_demo_embeddings():
            demo_embeddings = retriever.encode_multi_process(instruction_plus_demo_text, pool, batch_size=128)
            return demo_embeddings

        demo_embeddings = get_demo_embeddings()

        logger.info('demo_embeddings:{}'.format(demo_embeddings.shape))

        if args.clustered_retrieval > 0:
            assert args.clustered_retrieval >= num_demo
            demo_cluster_idxs = get_kmeans_clustered_idx(demo_embeddings, args.clustered_retrieval).tolist()
        else:
            demo_cluster_idxs = None

        logger.info('demo_cluster_idxs:\n{}'.format(demo_cluster_idxs))
        counter = Counter(demo_cluster_idxs)
        logger.info('clusters size:{}'.format(counter))
        # import counter

        # demo_embeddings =
        fitlog.add_hyper({'demo_pool_size': len(demo_pool)})
    else:
        demo_pool = None
        demo_embeddings = None
        fitlog.add_hyper({'demo_pool_size': 0})
        pass

    gpt2_tokenizer = AutoTokenizer.from_pretrained('gpt2-large')

    total = 0
    correct_list = []

    em_list_single_gold = []
    f1_list_single_gold = []

    em_list_multiple_gold = []
    f1_list_multiple_gold = []

    total_example_number = len(dataset_in_loader)

    if 'retrieval' in args.method:
        if args.query_encoding == 'x':
            questions_text_to_encode = dataset_in_loader.questions
        elif args.query_encoding == 'z':
            assert args.dataset in ['gsm8k', 'aqua', 'strategyqa_small']
            questions_text_to_encode = dataset_in_loader.rationales
        else:
            raise NotImplementedError

        if 'instructor' in args.retriever_name:
            instruction_plus_test_questions = list(
                map(lambda x: [q_retrieval_instruction, x], questions_text_to_encode))
        else:
            instruction_plus_test_questions = questions_text_to_encode

        @cache_results(_cache_fp='./embeddings_caches/query_embedding_{}_{}_{}_{}_{}_{}'.format(args.dataset,
                                                                                                args.retriever_name.replace(
                                                                                                    '/', '_'),
                                                                                                args.demo_encoding,
                                                                                                args.query_encoding,
                                                                                                args.demo_pool_path.replace(
                                                                                                    '/', '_'),
                                                                                                args.demo_pool_from),
                       _refresh=False)
        def get_query_embeddings():
            query_embeddings = retriever.encode_multi_process(instruction_plus_test_questions, pool)
            return query_embeddings

        query_embeddings = get_query_embeddings()
        logger.info('query_embeddings:{}'.format(query_embeddings.shape))

        retriever.stop_multi_process_pool(pool)
    # second_start = time.time()

    demos_num_clip_times = 0
    datas = list(dataset_in_loader)
    if args.method in ['lm_retrieval_few_shot_cot', 'lm_retrieval_few_shot',
                       'lm_retrieval_few_shot_cot_but_no_thinking']:
        logger.info('method is lm_retrieval_few_shot_cot, so start retrieve demos by lm')
        demos_group_s_for_gpt_to_decode = []
        for i, data in enumerate(list(dataset_in_loader)):
            x, y = data
            if args.dataset in nli_dataset:
                x = x + "\n" + "A:"
            else:
                x = "Q: " + x + "\n" + "A:"
            y = y
            if type(y) is str:
                y = y.strip()
            demo_scores = np.matmul(query_embeddings[i:i + 1], demo_embeddings.T)[0]
            demo_scores = torch.from_numpy(demo_scores)
            if args.dataset in nli_dataset:
                tmp_premise = extract_premise_and_hypothesis(x)['premise']
                if args.do_not_retrieve_same_premise_demo_with_test:
                    if tmp_premise in premise_to_demo_idxs_dict:
                        demo_scores[premise_to_demo_idxs_dict[tmp_premise]] = demo_scores[premise_to_demo_idxs_dict[
                            tmp_premise]] - 999
            if args.clustered_retrieval == 0:
                if args.how_to_divide_demos_for_retrieval == 'score_division':
                    # 1-10，11-20，21-30，31-40各一组去检索
                    _, demo_idxs = torch.topk(demo_scores, k=num_demo * 10)
                    demo_idxs_group = []
                    for j in range(num_demo):
                        demo_idxs_group.append(demo_idxs[j * 10: (j + 1) * 10])
                elif args.how_to_divide_demos_for_retrieval == 'score_mod':
                    # 1，5，9，13……一组，2，6，10，14……一组
                    _, demo_idxs = torch.topk(demo_scores, k=num_demo * 10)
                    demo_idxs_group = []
                    for j in range(num_demo):
                        demo_idxs_group.append([])
                        for k in range(10):
                            demo_idxs_group[-1].append(demo_idxs[j + k * num_demo])
                else:
                    raise NotImplementedError

            elif args.clustered_retrieval > 0:
                sorted_demo_idxs = torch.argsort(demo_scores, dim=0, descending=True).tolist()
                demo_idxs_group = []
                for j in range(args.num_demo):
                    demo_idxs_group.append([])

                for j in range(len(sorted_demo_idxs)):
                    tmp_demo_idx = sorted_demo_idxs[j]
                    if len(demo_idxs_group[demo_cluster_idxs[tmp_demo_idx]]) < 10:
                        demo_idxs_group[demo_cluster_idxs[tmp_demo_idx]].append(tmp_demo_idx)

                    if all(list(map(lambda x: len(x) == 10, demo_idxs_group))):
                        break

            else:
                raise NotImplementedError

            # demo_idxs_group
            demos_group = []
            for demo_idxs in demo_idxs_group:
                demos_group.append([])
                for tmp_demo_idx in demo_idxs:
                    demos_group[-1].append(demo_pool[tmp_demo_idx])

            demos_group_s_for_gpt_to_decode.append([demos_group, x[3:-3]])

            if (args.limit_dataset_size != 0) and ((i + 1) >= args.limit_dataset_size):
                break

        lm_retrieval_hyper_parameter = dict(model=args.model, n=1, top_p=1, temperature=0,
                                            max_length=64)

        logger.info('start retrieve_demos_by_lm')
        os.makedirs('lm_r_cache', exist_ok=True)
        os.makedirs('lm_r_cache/{}'.format(args.dataset), exist_ok=True)

        if args.num_demo == 4:
            @cache_results(
                _cache_fp='lm_r_cache/{}/{}_{}_{}_{}'.format(args.dataset, args.retriever_name,
                                                             args.clustered_retrieval, args.model,
                                                             args.demo_pool_path.replace('/', '_')), _hash_param=True,_refresh=True)
            def tmp():
                lm_retrieval_result = retrieve_demos_by_lm(demos_group_s_for_gpt_to_decode,
                                                           lm_retrieval_hyper_parameter,
                                                           args.multi_thread, 1,
                                                           args.demos_for_retrieval_using_purely_question,
                                                           args.shuffle_demos_for_lm_retrieval,
                                                           args.lm_format_requirement_at_last)
                return lm_retrieval_result

            lm_retrieval_result = tmp()
        else:
            @cache_results(
                _cache_fp='lm_r_cache/{}/{}_{}_{}_{}_{}'.format(args.dataset, args.retriever_name,
                                                                args.clustered_retrieval, args.model,
                                                                args.demo_pool_path.replace('/', '_'), args.num_demo),
                _hash_param=True)
            def tmp():
                lm_retrieval_result = retrieve_demos_by_lm(demos_group_s_for_gpt_to_decode,
                                                           lm_retrieval_hyper_parameter,
                                                           args.multi_thread, 1,
                                                           args.demos_for_retrieval_using_purely_question,
                                                           args.shuffle_demos_for_lm_retrieval,
                                                           args.lm_format_requirement_at_last)
                return lm_retrieval_result

            lm_retrieval_result = tmp()

        demos_for_every_x = lm_retrieval_result['retrieved_demos']

        fitlog.add_best_metric({'parsing_error_p': lm_retrieval_result['parsing_error_p']}, name='lm_retrieval')
        fitlog.add_best_metric(
            {'actual_num_demos_for_retrieval_avg': lm_retrieval_result['actual_num_demos_for_retrieval_avg']},
            name='lm_retrieval')

        logger.info('method is lm_retrieval_few_shot_cot, retrieve demos by lm finish')

        logger.info('demos_for_every_x:\n')
        logger.info(demos_for_every_x[0])

        demo_for_every_x_final = []

        token_num_list_save_for_debug = []

        demos_correct_p_for_every_x = []

        for i, d_dict_s in enumerate(demos_for_every_x):
            x, y = datas[i]
            if args.dataset in nli_dataset:
                x = x + "\n" + "A:"
            else:
                x = "Q: " + x + "\n" + "A:"
            tmp_demos = []
            # tmp_demos_correct_p = []
            demos_correct_p_for_every_x.append(0)
            for d_dict in d_dict_s:
                if 'cot' in args.method:
                    tmp_demos.append(d_dict['demonstration'])
                else:
                    tmp_demos.append(d_dict['demostration_without_rationale'])

                demos_correct_p_for_every_x[-1]+=int(d_dict['gold_ans'] == d_dict['pred_ans'].replace('.',''))

            while 1:
                tmp_demo = concat_demos(tmp_demos)
                if len(gpt2_tokenizer.tokenize(tmp_demo + x)) > 3600:
                    tmp_demos = tmp_demos[:-1]
                else:
                    break
                if len(tmp_demos) == 0:
                    break

            if len(tmp_demos) < num_demo:
                demos_num_clip_times += 1

            tmp_demo = concat_demos(tmp_demos)

            token_num_list_save_for_debug.append(tmp_demo + x)

            demo_for_every_x_final.append(tmp_demo)

        # with jsonlines.open('tmp_debug_1.jsonl','w') as out_f:
        #     for tmp_js in token_num_list_save_for_debug:
        #         out_f.write(tmp_js)

        logger.info('lm_retrieval_few_shot_cot: demos_num_clip_times:{}'.format(demos_num_clip_times))
        # exit()

    logger.info('demos_correct_p_for_every_x count:{}'.format(Counter(demos_correct_p_for_every_x)))

    x_list_to_decode = []

    demos_num_clip_times = 0

    with open(args.output_dir, "a") as wp:
        logger.info('args.limit_dataset_size:{}'.format(args.limit_dataset_size))
        for i, data in enumerate(dataset_in_loader):
            if i < args.resume_id - 1:
                # if i < 297:
                continue
            output_line = {}

            # print('*************************')
            # print("{}st data".format(i + 1))

            # Prepare question template ...
            x, y = data
            if args.dataset in nli_dataset:
                x = x
            else:
                x = "Q: " + x


            x = x + "\n" + "A:"

            # logger.info(i)
            # logger.info('y:{}'.format(y))

            y = y
            if type(y) is str:
                y = y.strip()
            # logger.info('y[0]:{}'.format(y))

            # print(x, y)

            output_line["question"] = x
            output_line["gold_ans"] = y

            if args.method == "zero_shot":
                x = x + " " + args.direct_answer_trigger_for_zeroshot
            elif args.method == "zero_shot_cot":
                x = x + " " + args.cot_trigger
            elif args.method == "few_shot":
                x = demo + x
            elif args.method == "few_shot_cot":
                x = demo + x
            elif args.method == 'my_random_sample_few_shot':
                x = demo + x
            elif args.method == 'my_random_sample_few_shot_cot':
                x = demo + x
            elif args.method == "auto_cot":
                x = demo + x + " " + args.cot_trigger
            elif args.method == 'retrieval_few_shot_cot' or args.method == 'retrieval_few_shot':
                if args.clustered_retrieval == 0:
                    demo_scores = np.matmul(query_embeddings[i:i + 1], demo_embeddings.T)[0]
                    demo_scores = torch.from_numpy(demo_scores)
                    if args.dataset in nli_dataset:
                        tmp_premise = extract_premise_and_hypothesis(x)['premise']
                        if args.do_not_retrieve_same_premise_demo_with_test:
                            if tmp_premise in premise_to_demo_idxs_dict:
                                demo_scores[premise_to_demo_idxs_dict[tmp_premise]] = demo_scores[
                                                                                          premise_to_demo_idxs_dict[
                                                                                              tmp_premise]] - 999
                        # demo_scores
                        pass
                    # print(demo_scores.size())
                    demos = []
                    if args.dataset in nli_dataset and args.do_not_retrieve_same_premise_demos:

                        sorted_demo_idxs = torch.argsort(demo_scores, dim=0, descending=True).tolist()
                        tmp_demo_premises = set()
                        for j, idx in enumerate(sorted_demo_idxs):
                            if len(demos) >= num_demo:
                                break
                            if premise_list[idx] not in tmp_demo_premises:
                                demos.append(demo_pool[idx]['demonstration'])
                                tmp_demo_premises.add(premise_list[idx])
                            else:
                                continue
                    else:

                        _, demo_idxs = torch.topk(demo_scores, k=num_demo)
                        # print(demo_idxs)
                        for idx in demo_idxs:
                            if 'cot' in args.method:
                                demos.append(demo_pool[idx]['demonstration'])
                            else:
                                demos.append(demo_pool[idx]['demostration_without_rationale'])

                    # print('demos:{}'.format(demos))
                    while 1:
                        tmp_demo = concat_demos(demos)
                        if len(gpt2_tokenizer.tokenize(tmp_demo + x)) > 3600:
                            demos = demos[:-1]
                        else:
                            break
                        if len(demos) == 0:
                            break
                    if len(demos) < num_demo:
                        demos_num_clip_times += 1

                    demo = concat_demos(demos)

                    if args.retrieval_hybrid_with_task_demos in ['manual', 'random_sample_from_demo_pool']:
                        #warn: if retireving the demos from zero_shot_plan_prompt, the task demo may not have the zero_shot_plan_prompt
                        demo = task_level_demo + demo

                    if zero_shot_plan_cot_prompt in demo:
                        x = x.replace('\nA:','')
                        x = x + zero_shot_plan_cot_prompt
                        x = x+ '\nA:'

                    x = demo + x
                elif args.clustered_retrieval > 0:
                    demo_scores = np.matmul(query_embeddings[i:i + 1], demo_embeddings.T)[0]
                    demo_scores = torch.from_numpy(demo_scores)
                    if args.dataset in nli_dataset:
                        tmp_premise = extract_premise_and_hypothesis(x)['premise']
                        if args.do_not_retrieve_same_premise_demo_with_test:
                            if tmp_premise in premise_to_demo_idxs_dict:
                                demo_scores[premise_to_demo_idxs_dict[tmp_premise]] = demo_scores[
                                                                                          premise_to_demo_idxs_dict[
                                                                                              tmp_premise]] - 999
                    # print(demo_scores.size())
                    # _, demo_idxs = torch.topk(demo_scores, k=)
                    sorted_demo_idxs = torch.argsort(demo_scores, dim=0, descending=True).tolist()
                    top_cluster_idxs = []
                    for j in range(100):
                        top_cluster_idxs.append(demo_cluster_idxs[sorted_demo_idxs[j]])
                    # logger.info('top_cluster_idxs:{}'.format(top_cluster_idxs))
                    tmp_demo_idxs = []
                    demos = []
                    tmp_demo_cluster_idxs = []
                    for j, demo_idx in enumerate(sorted_demo_idxs):
                        if len(demos) >= num_demo:
                            break
                        if demo_cluster_idxs[demo_idx] not in tmp_demo_cluster_idxs:
                            tmp_demo_idxs.append(demo_idx)
                            demos.append(demo_pool[demo_idx]['demonstration'])
                            tmp_demo_cluster_idxs.append(demo_cluster_idxs[demo_idx])
                        else:
                            continue

                    while 1:
                        tmp_demo = concat_demos(demos)
                        if len(gpt2_tokenizer.tokenize(tmp_demo + x)) > 3600:
                            demos = demos[:-1]
                        else:
                            break
                        if len(demos) == 0:
                            break
                    if len(demos) < num_demo:
                        demos_num_clip_times += 1

                    demo = concat_demos(demos)

                    if zero_shot_plan_cot_prompt in demo:
                        x = x.replace('\nA:','')
                        x = x + zero_shot_plan_cot_prompt
                        x = x+ '\nA:'
                    x = demo + x

                    # print('demo_idxs:{}'.format(tmp_demo_idxs))
                    # print('demo_cluster_idxs:{}'.format(tmp_demo_cluster_idxs))

                    # while len(demos)<num_demo:

                pass
            elif args.method in ['lm_retrieval_few_shot_cot', 'lm_retrieval_few_shot',
                                 'lm_retrieval_few_shot_cot_but_no_thinking']:
                demo = demo_for_every_x_final[i]

                if args.retrieval_hybrid_with_task_demos in ['manual', 'random_sample_from_demo_pool']:
                    demo = task_level_demo + demo

                x = demo + x
                if args.method == 'lm_retrieval_few_shot_cot_but_no_thinking':
                    x = x + ' I need you to straightly output the answer.'
                    x = [x, 'The answer is']
                pass




            else:
                raise ValueError("method is not properly defined ...")

            # Answer experiment by generating text ...

            # response = decoder.decode(args, x, max_length)
            x_list_to_decode.append(x)
            if (args.limit_dataset_size != 0) and ((i + 1) >= args.limit_dataset_size):
                break

    # with jsonlines.open('tmp_debug_2.jsonl', 'w') as out_f:
    #     for tmp_js in x_list_to_decode:
    #         out_f.write(tmp_js)

    for i in range(min(len(x_list_to_decode), 3)):
        print('*' * 50)
        logger.info('x_list_to_ddecode[{}]:'.format(i))
        print(x_list_to_decode[i])
        print('\n')

    # exit()

    # idx_x_list_to_decode = list(enumerate(x_list_to_decode))
    if 'no_thinking' in args.method:
        args.max_length_cot = args.max_length_direct

    max_length = args.max_length_cot if "cot" in args.method else args.max_length_direct

    logger.info('demos_num_clip_times:{} / {}'.format(demos_num_clip_times, len(x_list_to_decode)))

    # exit()

    if args.multi_thread < 2:
        # manager = get_manager()
        args.multi_thread_api = 0
        responses_with_idx = []
        for idx, x in tqdm.tqdm(list(enumerate(x_list_to_decode))):
            response = decoder.decode(args, x, max_length)
            responses_with_idx.append([idx, response])
    else:
        n = 1 if args.decoding_method == 'greedy' else args.self_consistency_paths
        hyper_parameter = dict(model=args.model, n=n, top_p=args.top_p, temperature=args.temperature,
                               max_length=max_length)
        logger.info('hyper_parameter:\n{}'.format(hyper_parameter))

        assert (len(dataloader) == len(x_list_to_decode))

        # start filtering too long input

        tmp = list(zip(datas, x_list_to_decode))

        def tmp_filter_func(x):
            tokenized = gpt2_tokenizer.tokenize(x[1])
            if len(tokenized) > 3650:
                print('excessive token number:{}'.format(len(tokenized)))
                return 0
            else:
                return 1

        filtered_tmp = list(
            filter(lambda x: tmp_filter_func(x), tqdm.tqdm(tmp, desc='length filtering')))
        # filtered_tmp = list(
        #     filter(lambda x: 1, tqdm.tqdm(tmp, desc='length filtering')))

        logger.info('examples before length filtering: {}'.format(len(tmp)))
        logger.info('examples after length filtering: {}'.format(len(filtered_tmp)))

        if len(filtered_tmp) < len(tmp):
            logger.info(
                'there are too long ones in x_list_to_decode, this makes the discrepancy of tested examples, so stop')
            exit()

        datas = list(map(lambda x: x[0], filtered_tmp))
        x_list_to_decode = list(map(lambda x: x[1], filtered_tmp))

        idx_x_list_to_decode = list(enumerate(x_list_to_decode))

        if args.inference_split == 'train':
            tmp_cache_fp = 'openai_result_caches/{}_{}_{}'.format(args.dataset, args.inference_split,
                                                                  args.limit_dataset_size)
            logger.info('tmp_cache_fp:{}'.format(tmp_cache_fp))

            @cache_results(_cache_fp=tmp_cache_fp)
            def tmp_get_response():
                pass
                responses = call_openai_multi_thread(x_list_to_decode, [hyper_parameter], args.multi_thread, 1,
                                                     args.turbo_system_message
                                                     )
                return responses

            responses = tmp_get_response()

        else:
            responses = call_openai_multi_thread(x_list_to_decode, [hyper_parameter], args.multi_thread, 1,
                                                 args.turbo_system_message
                                                 )

        # args.multi_thread_api = 1
        # inp_manager = OpenAI_API_inp_Manager_MultiThread(idx_x_list_to_decode)
        #
        # thread_list = []
        # manager = get_account_manager(1)
        # pbar = tqdm.tqdm(total=len(idx_x_list_to_decode))
        # for i in range(args.multi_thread):
        #     thread_list.append(MyThread(i, args, max_length, manager, 1, pbar, inp_manager))
        #
        # for t in thread_list:
        #     t.start()
        #
        # for i, t in enumerate(thread_list):
        #     t.join()
        #     logger.info('thread {} finish'.format(t.thread_id))
        #
        # responses_with_idx = []
        #
        # for t in thread_list:
        #     responses_with_idx.extend(t.responses_with_idx)
        #
        # responses_with_idx.sort(key=lambda x: x[0])

    assert (len(datas) == len(x_list_to_decode))
    assert len(responses) == len(x_list_to_decode)

    responses_with_idx = list(enumerate(responses))

    logger.info('responses_with_idx: [{}]'.format(len(responses_with_idx)))
    correct_list_for_every_demo_correct_p = {}
    for i in range(5):
        correct_list_for_every_demo_correct_p[i] = []
    with open(args.output_dir, "a") as wp:

        logger.info('output_num:{}'.format(len(list(zip(enumerate(datas), idx_x_list_to_decode,
                                                        responses_with_idx)))))

        for (idx_1, data), (idx_2, inp), (idx_3, response) in zip(enumerate(datas), idx_x_list_to_decode,
                                                                  responses_with_idx):

            output_line = {}

            # print('*************************')
            # print("{}st data".format(idx_1 + 1))
            x, y = data
            if args.dataset in nli_dataset:
                x = x + "\n" + "A:"
            else:
                x = "Q: " + x + "\n" + "A:"
            y = y

            if type(y) is str:
                y = y.strip()

            # print(x, y)

            output_line["question"] = x
            output_line["gold_ans"] = y
            max_length = args.max_length_cot if "cot" in args.method else args.max_length_direct
            # tqdm_bar.update(1)
            # second_now = time.time()
            # second_gap = second_now - second_start
            # second_every_example = second_gap / (i+1)
            # logger.info()
            # output_line["rationale"] = z

            if len(response['choices']) == 1:
                # Answer extraction for zero-shot-cot ...
                if 'turbo' in args.model:
                    z = response['choices'][0]['message']['content']
                else:
                    z = response['choices'][0]['text']
                if args.method == "zero_shot_cot":
                    z2 = x + z + " " + args.direct_answer_trigger_for_zeroshot_cot
                    max_length = args.max_length_direct
                    pred = decoder.decode(args, z2, max_length)
                    # print(z2 + pred)
                else:
                    pred = z
                    # print(inp + pred)

                # Clensing of predicted answer ...
                if args.method == 'lm_retrieval_few_shot_cot_but_no_thinking':
                    pred = pred.split('.')[0]
                pred = answer_cleansing(args, pred)
            elif len(response['choices']) > 1:
                preds_counter = Counter()
                response_set = set()
                if args.method == "zero_shot_cot":
                    for j, r in enumerate(response['choices']):
                        content = r['message']['content'] if 'turbo' in args.model else r['text']
                        response_set.add(content)
                        r_2_input = x + content + args.direct_answer_trigger_for_zeroshot_cot
                        tmp_pred = decoder.decode(args, r_2_input, max_length)
                        tmp_pred = answer_cleansing(args, tmp_pred, verbose=False)
                        preds_counter[tmp_pred] += 1
                        r['tmp_pred'] = tmp_pred
                else:
                    for j, r in enumerate(response['choices']):
                        content = r['message']['content'] if 'turbo' in args.model else r['text']
                        if args.method == 'lm_retrieval_few_shot_cot_but_no_thinking':
                            content = content.split('.')[0]
                        response_set.add(content)
                        tmp_pred = answer_cleansing(args, content, verbose=False)
                        r['tmp_pred'] = tmp_pred
                        preds_counter[tmp_pred] += 1

                pred = sorted(list(preds_counter.items()), key=lambda x: x[1], reverse=True)[0][0]

                output_line['pred_set_size'] = len(preds_counter)
                output_line['pred_count'] = sorted(list(preds_counter.items()), key=lambda x: x[1], reverse=True)
                output_line['response_set_size'] = len(response_set)
                print('response_set_size : {}'.format(len(response_set)))
                print('pred_counter : {}'.format(preds_counter))

            output_line['response'] = response

            # print(r_2_input + pred)

            output_line["pred_ans"] = pred
            output_line["wrap_question"] = inp

            output_json = json.dumps(output_line)
            wp.write(output_json + '\n')

            # Choose the most frequent answer from the list ...

            print("pred : {}".format(pred))
            print("GT : ", y)
            print('*************************')

            # Checking answer ...
            if args.dataset == 'drop':
                pass
                # tmp_exact_match, tmp_f1 = (pred, y)
                tmp_exact_match_single_gold, tmp_f1_single_gold = pred_to_one_answer_f1_metrics(pred, y[0],
                                                                                                numerically_strict=1)
                tmp_exact_match_multiple_gold, tmp_f1_multiple_gold = pred_to_many_f1_metrics(pred, y,
                                                                                              numerically_strict=1)
                em_list_single_gold.append(tmp_exact_match_single_gold)
                f1_list_single_gold.append(tmp_f1_single_gold)
                em_list_multiple_gold.append(tmp_exact_match_multiple_gold)
                f1_list_multiple_gold.append(tmp_f1_multiple_gold)

                # correct_list.append(tmp_exact_match)
                # f1_list.append(tmp_f1)
                total = idx_1 + 1

                em_single_gold = (sum(em_list_single_gold) * 1.0 / total) * 100
                f1_single_gold = (sum(f1_list_single_gold) * 1.0 / total) * 100

                em_multiple_gold = (sum(em_list_multiple_gold) * 1.0 / total) * 100
                f1_multiple_gold = (sum(f1_list_multiple_gold) * 1.0 / total) * 100

                print('{}/{} exact match single gold: {}'.format(total, total_example_number, em_single_gold))
                print('{}/{} f1 single gold: {}'.format(total, total_example_number, f1_single_gold))
                print('')
                print('{}/{} exact match multiple gold: {}'.format(total, total_example_number, em_multiple_gold))
                print('{}/{} f1 multiple gold: {}'.format(total, total_example_number, f1_multiple_gold))

                # exact_match_acc = (sum(correct_list) * 1.0 / total) * 100
                # total_f1 = (sum(f1_list) * 1.0 / total) * 100
                # print("{}/{} exact match : {}".format(total, total_example_number, exact_match_acc))
                # print("{}/{} exact match : {}".format(total, total_example_number, total_f1))
            elif args.dataset in ['hotpot_qa', 'qa_wikidata']:
                tmp_exact_match_single_gold, tmp_f1_single_gold = pred_to_one_answer_f1_metrics(pred, y,
                                                                                                numerically_strict=1)
                em_list_single_gold.append(tmp_exact_match_single_gold)
                f1_list_single_gold.append(tmp_f1_single_gold)

                total = idx_1 + 1

                em_single_gold = (sum(em_list_single_gold) * 1.0 / total) * 100
                f1_single_gold = (sum(f1_list_single_gold) * 1.0 / total) * 100

                em_multiple_gold = (sum(em_list_multiple_gold) * 1.0 / total) * 100
                f1_multiple_gold = (sum(f1_list_multiple_gold) * 1.0 / total) * 100

                print('{}/{} exact match single gold: {}'.format(total, total_example_number, em_single_gold))
                print('{}/{} f1 single gold: {}'.format(total, total_example_number, f1_single_gold))




            else:
                correct = (np.array([pred]) == np.array([y])).sum().item()
                correct_list.append(correct)
                correct_list_for_every_demo_correct_p[demos_correct_p_for_every_x[idx_1]].append(correct)
                total += 1  # np.array([y]).size(0)

                accuracy = (sum(correct_list) * 1.0 / total) * 100
                print("{}/{} accuracy : {}".format(total, total_example_number, accuracy))

        for j in range(5):
            tmp_correct_list = correct_list_for_every_demo_correct_p[j]
            print('demo_correct_p: {}, acc: {}'.format(j, (sum(tmp_correct_list) * 1.0 / (len(tmp_correct_list) if len(tmp_correct_list) > 0 else -1)) * 100))
        tmp_correct_list = correct_list
        print('demo_correct_p: {}, acc: {}'.format('all', (sum(tmp_correct_list) * 1.0 / len(tmp_correct_list)) * 100))
        logger.info('demos_correct_p_for_every_x count:{}'.format(Counter(demos_correct_p_for_every_x)))
            # logger.info('args.limit_dataset_size:{}'.format(args.limit_dataset_size))
            # logger.info('(args.limit_dataset_size != 0) and ((i + 1) >= args.limit_dataset_size):{}'.format(
            #     (args.limit_dataset_size != 0) and ((i + 1) >= args.limit_dataset_size)))

            # raise ValueError("Stop !!")

    # Calculate accuracy ...
    if args.dataset == 'drop':
        pass
        em_single_gold = (sum(em_list_single_gold) * 1.0 / total) * 100
        f1_single_gold = (sum(f1_list_single_gold) * 1.0 / total) * 100

        em_multiple_gold = (sum(em_list_multiple_gold) * 1.0 / total) * 100
        f1_multiple_gold = (sum(f1_list_multiple_gold) * 1.0 / total) * 100

        print('{}/{} exact match single gold: {}'.format(total, total_example_number, em_single_gold))
        print('{}/{} f1 single gold: {}'.format(total, total_example_number, f1_single_gold))
        print('')
        print('{}/{} exact match multiple gold: {}'.format(total, total_example_number, em_multiple_gold))
        print('{}/{} f1 multiple gold: {}'.format(total, total_example_number, f1_multiple_gold))

        fitlog.add_best_metric({'em_s': em_single_gold})
        fitlog.add_best_metric({'em_m': em_multiple_gold})
        fitlog.add_best_metric({'f1_s': f1_single_gold})
        fitlog.add_best_metric({'f1_m': f1_multiple_gold})

    elif args.dataset in ['hotpot_qa', 'qa_wikidata']:

        em_single_gold = (sum(em_list_single_gold) * 1.0 / total) * 100
        f1_single_gold = (sum(f1_list_single_gold) * 1.0 / total) * 100

        print('{}/{} exact match single gold: {}'.format(total, total_example_number, em_single_gold))
        print('{}/{} f1 single gold: {}'.format(total, total_example_number, f1_single_gold))

        fitlog.add_best_metric({'em_s': em_single_gold})
        fitlog.add_best_metric({'f1_s': f1_single_gold})


    else:
        accuracy = (sum(correct_list) * 1.0 / total) * 100
        print("accuracy : {}".format(accuracy))
        fitlog.add_best_metric({'test_acc': accuracy})
    fitlog.add_best_metric({'tmp': 2})


def parse_arguments():
    parser = argparse.ArgumentParser(description="Zero-shot-CoT")
    parser.add_argument('--filter_no_trigger', default=-1)
    parser.add_argument('--demo_c', default=-1)
    parser.add_argument('--entropy_threshold', default=-1)
    parser.add_argument('--how_to_divide_demos_for_retrieval', required=True)
    parser.add_argument('--lm_format_requirement_at_last', type=int, required=True)
    parser.add_argument('--shuffle_demos_for_lm_retrieval', type=int, required=True)
    parser.add_argument('--demos_for_retrieval_using_purely_question', required=True, type=int)
    # parser.add_argument('--retrieval_lm_system_message',required=True)
    parser.add_argument('--turbo_system_message', required=True)
    parser.add_argument('--retrieval_hybrid_with_task_demos', required=True)
    parser.add_argument('--do_not_retrieve_same_premise_demos', type=int, required=True)
    # whether there are the same premises in retrieved demos
    parser.add_argument('--do_not_retrieve_same_premise_demo_with_test', type=int, required=True)
    # whether there are the same premise as query example's premise in retrieved demos
    parser.add_argument('--limit_account_num', default=-1, type=int)
    parser.add_argument('--exp_tag', default='None')
    parser.add_argument('--demo_pool_path', )
    parser.add_argument('--demo_pool_from', choices=['gt', 'lm_inference'], required=True)
    parser.add_argument('--multi_thread', type=int, required=True)
    parser.add_argument('--inference_split', required=True, choices=['test', 'train'])
    parser.add_argument('--exp_name', required=True)
    parser.add_argument('--num_demo', type=int, default=-1)
    parser.add_argument('--demo_sampling_seed', type=int, required=True)
    parser.add_argument('--retriever_name', )
    parser.add_argument('--demo_encoding', choices=['x'], required=True)
    parser.add_argument('--query_encoding', choices=['x'], required=True)
    parser.add_argument('--clustered_retrieval', type=int, required=True)

    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--dataset", type=str, default="multiarith",
        choices=["aqua", "gsm8k", "commonsensqa", "addsub", "multiarith", "strategyqa", "svamp", "singleeq",
                 "coin_flip", "last_letters", "strategyqa_small", 'openbookqa', 'anli_a2', 'anli_a3', 'drop',
                 'elementary_math_qa', 'boolq', 'fact_checker', 'com_v', 'com_e', 'anli_a1', 'hotpot_qa',
                 'qa_wikidata'],
        help="dataset used for experiment"
    )
    parser.add_argument(
        "--demo_path", type=str, default="demos/multiarith", help="pre-generated demos used for experiment"
    )
    parser.add_argument(
        "--resume_id", type=int, default=0,
        help="resume from which question id (current line number in the output file), if the experiment fails accidently (e.g., network error)"
    )
    parser.add_argument("--minibatch_size", type=int, default=1, choices=[1],
                        help="minibatch size should be 1 because GPT-3 API takes only 1 input for each request")

    parser.add_argument("--max_num_worker", type=int, default=0, help="maximum number of workers for dataloader")

    # parser.add_argument(
    #     "--model", type=str, default="gpt3-xl", choices=["gpt3", "gpt3-medium", "gpt3-large", "gpt3-xl", "code-davinci-002"], help="model used for decoding. Note that 'gpt3' are the smallest models."
    # )
    parser.add_argument('--model', required=True)

    parser.add_argument(
        "--method", type=str, default="auto_cot",
        choices=["zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot", "auto_cot", "retrieval_few_shot_cot",
                 "retrieval_few_shot", "my_random_sample_few_shot", 'my_random_sample_few_shot_cot',
                 'lm_retrieval_few_shot_cot', 'lm_retrieval_few_shot', 'lm_retrieval_few_shot_cot_but_no_thinking'],
        help="method"
    )
    parser.add_argument(
        "--output_dir", type=str, default="experiment/multiarith", help="output directory"
    )
    parser.add_argument(
        "--max_length_cot", type=int, default=256,
        help="maximum length of output tokens by model for reasoning extraction"
    )
    parser.add_argument(
        "--max_length_direct", type=int, default=32,
        help="maximum length of output tokens by model for answer extraction"
    )
    parser.add_argument(
        "--limit_dataset_size", type=int, default=0,
        help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
    )
    parser.add_argument(
        "--api_time_interval", type=float, default=1.0,
        help="sleep between runs to avoid excedding the rate limit of openai api"
    )
    parser.add_argument(
        "--temperature", type=float, required=True, help="temperature for GPT-3"
    )
    parser.add_argument('--top_p', type=float, default=1)
    parser.add_argument(
        "--log_dir", type=str, default="./log/", help="log directory"
    )

    parser.add_argument('--decoding_method', required=True, choices=['greedy', 'sampling_once', 'self_consistency'])

    parser.add_argument('--self_consistency_paths', type=int, required=True)

    args = parser.parse_args()

    if 'cot' in args.method and args.dataset in ['boolq', 'qa_wikidata', 'hotpot_qa', 'boolq', 'fact_checker',
                                                 'qa_wikidata', 'com_v', 'com_e']:
        args.max_length = 128
        args.max_length_cot = 128

    manager = get_account_manager(1, limit_account_num=args.limit_account_num)

    if args.decoding_method == 'greedy':
        args.temperature = 0
        args.top_p = 1
    # if args.decoding_method in ['greedy','sampling_once']:
    #     args.

    if args.dataset == "aqua":
        args.dataset_path = "./downstream_datasets/AQuA/test.json"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
    elif args.dataset == 'openbookqa':
        args.direct_answer_trigger = "\nTherefore, the answer is"
    elif args.dataset in ['anli_a2', 'anli_a3', 'anli_a1']:
        args.direct_answer_trigger = "\nTherefore, the answer is"
    elif args.dataset == 'drop':
        args.direct_answer_trigger = "\nTherefore, the answer is"
    elif args.dataset == 'boolq':
        args.direct_answer_trigger = "\nTherefore, the answer is"
    elif args.dataset == 'fact_checker':
        args.direct_answer_trigger = "\nTherefore, the answer is"
    elif args.dataset == 'com_v':
        args.direct_answer_trigger = "\nTherefore, the answer is"
    elif args.dataset == 'qa_wikidata':
        args.direct_answer_trigger = "\nTherefore, the answer is"
    else:
        raise ValueError("dataset is not properly defined ...")

    # "Therefore, the answer ..." -> "The answer ..."
    trigger = args.direct_answer_trigger.replace("\nTherefore, ", "")
    args.direct_answer_trigger_for_zeroshot = trigger[0].upper() + trigger[1:]
    args.direct_answer_trigger_for_zeroshot_cot = args.direct_answer_trigger
    args.direct_answer_trigger_for_fewshot = "The answer is"
    args.cot_trigger = "Let's think step by step."

    return args


if __name__ == "__main__":
    main()
    fitlog.finish()
    1234578
