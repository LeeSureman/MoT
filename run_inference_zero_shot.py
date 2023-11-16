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
                                                                                               'gt' if 'gt' in args.demo_pool_path else 'confidence',
                                                                                               args.demo_pool_from),_refresh=False)
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
    else:
        demo_pool = None
        demo_embeddings = None
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
                                                                                                'gt' if 'gt' in args.demo_pool_path else 'confidence',
                                                                                                args.demo_pool_from),_refresh=False)
        def get_query_embeddings():
            query_embeddings = retriever.encode_multi_process(instruction_plus_test_questions, pool)
            return query_embeddings

        query_embeddings = get_query_embeddings()
        logger.info('query_embeddings:{}'.format(query_embeddings.shape))

        retriever.stop_multi_process_pool(pool)
    # second_start = time.time()



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
                x = x + "\n"
            else:
                x = "Q: " + x + "\n"

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
                x = x
                x = [x,'The answer is']
                # x = x + '\n' + 'You must directly answer this question. So, the answer is'
            elif args.method == "zero_shot_cot":
                x = x + "Your response must end with the format \"The answer is ...\". " \
                        "If my question is a multi-choice question and the answer is A, your response must end with \"The answer is A.\"" \
                        " If the answer is Bob, your response must end with \"The answer is Bob.\""
                x = [x, 'Let\'s think step by step.']
            else:
                raise ValueError("method is not properly defined ...")

            # Answer experiment by generating text ...

            # response = decoder.decode(args, x, max_length)
            x_list_to_decode.append(x)
            if (args.limit_dataset_size != 0) and ((i + 1) >= args.limit_dataset_size):
                break

    for i in range(min(len(x_list_to_decode), 3)):
        print('*' * 50)
        logger.info('x_list_to_ddecode[{}]:'.format(i))
        print(x_list_to_decode[i])
        print('\n')

    # exit()

    # idx_x_list_to_decode = list(enumerate(x_list_to_decode))
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

        datas = list(dataset_in_loader)
        tmp = list(zip(datas, x_list_to_decode))
        filtered_tmp = list(
            filter(lambda x: len(gpt2_tokenizer.tokenize(x[1])) < 3600, tqdm.tqdm(tmp, desc='length filtering')))
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
                # if args.method == "zero_shot_cot":
                if 0:
                    z2 = x + z + " " + args.direct_answer_trigger_for_zeroshot_cot
                    max_length = args.max_length_direct
                    pred = decoder.decode(args, z2, max_length)
                    # print(z2 + pred)
                else:
                    pred = z
                    # print(inp + pred)

                # Clensing of predicted answer ...
                if args.method == 'zero_shot':
                    pred = pred.split('.')[0]
                print('x:\n{}'.format(x))
                pred = answer_cleansing(args, pred)
            elif len(response['choices']) > 1:
                preds_counter = Counter()
                response_set = set()
                # if args.method == "zero_shot_cot":
                if 0:
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
                        if args.method == 'zero_shot':
                            content = content.split('.')[0]
                            #就算给turbo输入the answer is的前缀，它还是会在后面加解释，所以在这里直接把解释给去掉
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
                total += 1  # np.array([y]).size(0)

                accuracy = (sum(correct_list) * 1.0 / total) * 100
                print("{}/{} accuracy : {}".format(total, total_example_number, accuracy))
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
    parser.add_argument('--filter_no_trigger',default=-1)
    parser.add_argument('--demo_c',default=-1)
    parser.add_argument('--entropy_threshold',default=-1)
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
        choices=["zero_shot", "zero_shot_cot",'lm_retrieval_few_shot_cot_but_no_thinking'],
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

    if args.dataset in ['boolq', 'qa_wikidata', 'hotpot_qa', 'boolq', 'fact_checker', 'qa_wikidata', 'com_v', 'com_e']:
        args.max_length = 128
        args.max_length_cot = 128

    manager = get_account_manager(1, limit_account_num=args.limit_account_num)

    if args.decoding_method == 'greedy':
        args.temperature = 0
        args.top_p = 1
    # if args.decoding_method in ['greedy','sampling_once']:
    #     args.

    if args.dataset == "aqua":
        args.dataset_path = "./dataset/AQuA/test.json"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
    elif args.dataset == "gsm8k":
        args.dataset_path = "./dataset/grade-school-math/test.jsonl"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "commonsensqa":
        args.dataset_path = "./dataset/CommonsenseQA/dev_rand_split.jsonl"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
        args.plausible_answer_trigger = "Choose the most plausible answer from among choices A through E."
    elif args.dataset == "addsub":
        args.dataset_path = "./dataset/AddSub/AddSub.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "multiarith":
        args.dataset_path = "./dataset/MultiArith/MultiArith.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "strategyqa":
        args.dataset_path = "./dataset/StrategyQA/task.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "svamp":
        args.dataset_path = "./dataset/SVAMP/SVAMP.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "singleeq":
        args.dataset_path = "./dataset/SingleEq/questions.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "bigbench_date":
        args.dataset_path = "./dataset/Bigbench_Date/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through F, the answer is"
    elif args.dataset == "object_tracking":
        args.dataset_path = "./dataset/Bigbench_object_tracking/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through C, the answer is"
    elif args.dataset == "coin_flip":
        args.dataset_path = "./dataset/coin_flip/coin_flip.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "last_letters":
        args.dataset_path = "./dataset/last_letters/last_letters.json"
        args.direct_answer_trigger = "\nTherefore, the answer is"
    elif args.dataset == "strategyqa_small":
        args.direct_answer_trigger = "\nTherefore, the answer is"
        pass
    elif args.dataset == 'openbookqa':
        args.direct_answer_trigger = "\nTherefore, the answer is"
    elif args.dataset in ['anli_a2', 'anli_a3', 'anli_a1']:
        args.direct_answer_trigger = "\nTherefore, the answer is"
    elif args.dataset == 'drop':
        args.direct_answer_trigger = "\nTherefore, the answer is"
    elif args.dataset == 'elementary_math_qa':
        args.direct_answer_trigger = "\nTherefore, the answer is"
    elif args.dataset == 'boolq':
        args.direct_answer_trigger = "\nTherefore, the answer is"
    elif args.dataset == 'fact_checker':
        args.direct_answer_trigger = "\nTherefore, the answer is"
    elif args.dataset == 'com_v':
        args.direct_answer_trigger = "\nTherefore, the answer is"
    elif args.dataset == 'com_e':
        args.direct_answer_trigger = "\nTherefore, the answer is"
    elif args.dataset == 'hotpot_qa':
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
