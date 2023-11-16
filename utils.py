'''
Adapted from https://github.com/kojima-takeshi188/zero_shot_cot
'''

from statistics import mean
from torch.utils.data import Dataset
import openai
import os
import multiprocessing
import json
import numpy as np
import torch
import re
import random
import time
import datetime
import logging
from datasets import load_dataset
from data_process_utils import make_gsm8k_answer, make_gsm8k_rationale, make_aqua_x, concat_demos, create_single_demo, \
    make_aqua_rationale, transform_original_aqua_into_x_z_y, make_csqa_x, make_openbookqa_x, make_nli_question, \
    make_drop_question, make_drop_answer, make_elementary_math_qa_question, make_elementary_math_qa_answer
import string
from sklearn.cluster import KMeans
import jsonlines

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
# def setup_logger(logger):
#     logger.setLevel(logging.INFO)
#     if logger.hasHandlers():
#         logger.handlers.clear()
#     log_formatter = logging.Formatter(
#         "[%(thread)s] %(asctime)s [%(levelname)s] %(name)s: %(message)s"
#     )
#     console = logging.StreamHandler()
#     console.setFormatter(log_formatter)
#     logger.addHandler(console)

logger = logging.getLogger(__name__)


# from openai_account_manager import get_account_manager
#
# manager = get_account_manager()


# def null_decorator(func):
#     return func


# def repeat_until_success_call_openai_api(func):
#     def wrapper(*args, **kw):
#         while 1:
#             result = None
#             try:
#                 result = func(*args, **kw)
#             except openai.error.APIConnectionError as e:
#                 logger.info('openai connection error, so retry after sleep 5 seconds')
#                 logger.info(e)
#                 time.sleep(5)
#             except openai.error.RateLimitError as e:
#                 if 'quota' in e._message:
#                     logger.info('now openai account {} runs out. so use next.'.format(openai.api_key))
#                     logger.info(e)
#                     manager.use_next_account()
#                 else:
#                     logger.info('openai rate limit error, so retry after sleep 1 seconds')
#                     logger.info(e)
#                     time.sleep(0.5)
#             except Exception as e:
#                 logger.info('meet unexpected error, so retry after sleep 5 seconds')
#                 logger.info(e)
#                 time.sleep(3)
#
#             if result != None:
#                 return result
#             else:
#                 pass
#
#     return wrapper


def shuffleDict(d):
    keys = list(d.keys())
    random.shuffle(keys)
    [(key, d[key]) for key in keys]
    random.shuffle(keys)
    [(key, d[key]) for key in keys]
    random.shuffle(keys)
    keys = [(key, d[key]) for key in keys]
    # keys = d(keys)
    return dict(keys)


def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def print_now(return_flag=0):
    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, 'JST')
    now = datetime.datetime.now(JST)
    now = now.strftime('%Y/%m/%d %H:%M:%S')
    if return_flag == 0:
        print(now)
    elif return_flag == 1:
        return now
    else:
        pass


# Sentence Generator (Decoder) for GPT-3 ...
# because it needs the args from argumentParser, it is not general, so it is discarded from V6, use decoder_for_gpt3_new
def decoder_for_gpt3(args, inp, max_length, api_key=None):
    # because it needs the args from argumentParser, it is not general, so it is discarded from V6, use decoder_for_gpt3_new



    engine = args.model
    n = 1
    if args.decoding_method == 'self_consistency':
        n = args.self_consistency_paths

    if args.multi_thread_api:
        openai_api_call_decorator = null_decorator
    else:
        openai_api_call_decorator = repeat_until_success_call_openai_api

    if ("few_shot" in args.method or "auto" in args.method) and engine == "code-davinci-002":
        @openai_api_call_decorator
        def tmp_openai_completion():
            response = openai.Completion.create(
                engine=engine,
                prompt=inp,
                max_tokens=max_length,
                temperature=args.temperature,
                top_p=args.top_p,
                frequency_penalty=0,
                presence_penalty=0,
                stop=["\n"],
                n=n,
                api_key=api_key
            )
            return response

        response = tmp_openai_completion()
        # logger.info('get response')
        # logger.info(response)


    elif 'turbo' not in engine:
        @openai_api_call_decorator
        def tmp_openai_completion():
            response = openai.Completion.create(
                engine=engine,
                prompt=inp,
                max_tokens=max_length,
                temperature=args.temperature,
                top_p=args.top_p,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                n=n,
                api_key=api_key
            )
            return response

        response = tmp_openai_completion()
        # logger.info('get response')
        # logger.info(response)
    else:
        @openai_api_call_decorator
        def tmp_openai_completion():
            response = openai.ChatCompletion.create(
                model=engine,
                messages=[
                    {"role": "system", "content": ""},
                    {"role": "user", "content": inp},
                ],
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=max_length,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                n=n,
                api_key=api_key
            )
            return response

        response = tmp_openai_completion()
        # logger.info('get response')
        # logger.info(response)

    # return response["choices"][0]["text"]
    return response


def decoder_for_gpt3_new(inp, inference_hyper_parameter_dict, turbo_system_message, api_key=None):

    assert 'model' in inference_hyper_parameter_dict
    assert 'max_length' in inference_hyper_parameter_dict
    assert 'n' in inference_hyper_parameter_dict
    assert 'temperature' in inference_hyper_parameter_dict
    assert 'top_p' in inference_hyper_parameter_dict



    engine = inference_hyper_parameter_dict['model']

    max_length = inference_hyper_parameter_dict['max_length']
    n = inference_hyper_parameter_dict['n']
    temperature = inference_hyper_parameter_dict['temperature']
    top_p = inference_hyper_parameter_dict['top_p']

    if 'turbo' not in engine:
        if type(inp) is list:
            logger.info('为了保险起见，暂不支持davinci zero shot')
            exit()
            raise NotImplementedError

        def tmp_openai_completion():
            response = openai.Completion.create(
                engine=engine,
                prompt=inp,
                max_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                n=n,
                api_key=api_key
            )
            return response

        response = tmp_openai_completion()

    else:
        if turbo_system_message in [None, 'none', 'None']:
            turbo_system_message = ""
        else:
            pass

        if type(inp) is list:
            assert len(inp) == 2, 'inp是list的情况下就是zero-shot或者zero-shot-cot方式，直接给assistant强制The answer is的前缀，' \
                                  'inp[-1]应该就是The answer is（zero-shot），或者Let\'s think step by step. (zero-shot cot)'
            user_query = inp[0]
            assistant_prefix=inp[1]
            # user_query = user_query + '\n' + assistant_prefix
            messages = [
                    {"role": "system", "content": turbo_system_message},
                    {"role": "user", "content": user_query},
                    {"role": "assistant", 'content':assistant_prefix}
                ]
        elif type(inp) is str:
            messages = [
                    {"role": "system", "content": turbo_system_message},
                    {"role": "user", "content": inp},
                ]
            pass
        else:
            logger.info('inp type错误，所以退出')
            # raise NotImplementedError

        # print('messages:{}'.format(messages))


        def tmp_openai_completion():
            response = openai.ChatCompletion.create(
                model=engine,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_length,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                n=n,
                api_key=api_key
            )
            return response

        response = tmp_openai_completion()

    return response


class Decoder():
    def __init__(self):
        # print_now()
        pass

    def decode(self, args, input, max_length):
        response = decoder_for_gpt3(args, input, max_length)
        return response


def data_reader(args):
    questions = []
    answers = []
    rationales = []
    decoder = json.JSONDecoder()

    if args.dataset == "aqua":
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                choice = "(" + "(".join(json_res["options"])
                choice = choice.replace("(", " (").replace(")", ") ")
                choice = "Answer Choices:" + choice
                questions.append(json_res["question"].strip() + " " + choice)
                answers.append(json_res["correct"])

                rationale = json_res['rationale']
                rationale = list(filter(lambda x: len(x) > 0, rationale.strip().split('\n')))[:-1]
                rationale = list(map(lambda x: x + '.' if (x[-1] not in string.punctuation) else x, rationale))
                rationale = ' '.join(rationale)
                rationales.append(rationale)

    elif args.dataset == "gsm8k":
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                questions.append(json_res["question"].strip())
                answers.append(json_res["answer"].split("#### ")[-1])

                rationale = json_res["answer"].split("#### ")[0]
                rationale = list(filter(lambda x: len(x) > 0, rationale.strip().split('\n')))
                rationale = list(map(lambda x: x + '.' if (x[-1] not in string.punctuation) else x, rationale))
                rationale = ' '.join(rationale)

                rationales.append(rationale)

    elif args.dataset == "commonsensqa":
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                choice = "Answer Choices:"
                for c in json_res["question"]["choices"]:
                    choice += " ("
                    choice += c["label"]
                    choice += ") "
                    choice += c["text"]
                questions.append(json_res["question"]["stem"].strip() + " " + choice)
                answers.append(json_res["answerKey"])

    elif args.dataset in ("addsub", "multiarith", "singleeq"):
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            for line in json_data:
                q = line["sQuestion"].strip()
                a = str(line["lSolutions"][0])
                if a[-2:] == ".0":
                    a = a[:-2]
                questions.append(q)
                answers.append(a)

    elif args.dataset == "strategyqa":
        with open(args.dataset_path) as f:
            json_data = json.load(f)["examples"]
            for line in json_data:
                q = line["input"].strip()
                a = int(line["target_scores"]["Yes"])
                if a == 1:
                    a = "yes"
                else:
                    a = "no"
                questions.append(q)
                answers.append(a)

    elif args.dataset == 'strategyqa_small':
        strategy_qa_dataset = list(load_dataset("metaeval/strategy-qa")['train'])

        random.Random(1208).shuffle(strategy_qa_dataset)

        strategy_qa_dataset_test = strategy_qa_dataset[:250]

        for line in strategy_qa_dataset_test:
            q = line['question']
            a = 'yes' if line['answer'] else 'no'

            questions.append(q)
            answers.append(a)

            tmp_rationale = ' '.join(line['facts'])

            rationales.append(tmp_rationale)




    elif args.dataset == "svamp":
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            for line in json_data:
                q = line["Body"].strip() + " " + line["Question"].strip()
                a = str(line["Answer"])
                if a[-2:] == ".0":
                    a = a[:-2]
                questions.append(q)
                answers.append(a)

    elif args.dataset in ("bigbench_date", "object_tracking"):
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            json_data = json_data["examples"]
            if args.dataset == "bigbench_date":
                choice_index = ['A', 'B', 'C', 'D', 'E', 'F']
            elif args.dataset in ("object_tracking"):
                choice_index = ['A', 'B', 'C']
            else:
                raise ValueError("dataset is not properly defined ...")
            for line in json_data:
                q = line["input"].strip()
                if args.dataset == "bigbench_date":
                    choice = "Answer Choices:"
                    # Randomly shuffle the answer choice dictionary because the original answer is always A ...
                    choice_dic = shuffleDict(line["target_scores"])
                elif args.dataset == "object_tracking":
                    choice = "\nWhich choice is true ? Answer Choices:"
                    choice_dic = line["target_scores"]
                else:
                    raise ValueError("dataset is not properly defined ...")
                for i, key_value in enumerate(choice_dic.items()):
                    key, value = key_value
                    choice += " ("
                    choice += choice_index[i]
                    choice += ") "
                    choice += key
                    if value == 1:
                        a = choice_index[i]
                        # a = key
                q = q + " " + choice
                questions.append(q)
                answers.append(a)

    elif args.dataset in ("coin_flip", "last_letters"):
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            json_data = json_data["examples"]
            for line in json_data:
                q = line["question"]
                a = line["answer"]
                questions.append(q)
                answers.append(a)

    elif args.dataset == 'openbookqa':
        openbook_qa = list(load_dataset("openbookqa", 'additional')['test'])
        for line in openbook_qa:
            q = line['question_stem'].strip()
            a = line['answerKey'].strip()
            r = line['fact1'].strip()
            # tmp_choices = line['choices']['text']
            # assert line['label'] == ['A','B','C','D']
            #
            # tmp_choices[0] = 'A){}'.format(tmp_choices[0])
            # tmp_choices[1] = 'B){}'.format(tmp_choices[1])
            # tmp_choices[2] = 'C){}'.format(tmp_choices[2])
            # tmp_choices[3] = 'D){}'.format(tmp_choices[3])

            choice = "Answer Choices:"
            for label, text in zip(line['choices']['label'], line['choices']['text']):
                choice += " ("
                choice += label
                choice += ") "
                choice += text
            q = q + ' ' + choice
            questions.append(q)
            answers.append(a)
            rationales.append(r)

    elif args.dataset == 'anli_a1':
        anli_a2_dataset = list(load_dataset('anli')['test_r1'])
        for line in anli_a2_dataset:
            premise = line['premise']
            hypothesis = line['hypothesis']
            if line['label'] == 0:
                answer = 'yes'
            elif line['label'] == 1:
                answer = 'it is not possible to tell'
            elif line['label'] == 2:
                answer = 'no'
            else:
                raise NotImplementedError
            final_q = make_nli_question({'premise': premise, 'hypothesis': hypothesis})
            questions.append(final_q)
            answers.append(answer)
            rationales.append(line['reason'])

    elif args.dataset == 'anli_a2':
        anli_a2_dataset = list(load_dataset('anli')['test_r2'])
        for line in anli_a2_dataset:
            premise = line['premise']
            hypothesis = line['hypothesis']
            if line['label'] == 0:
                answer = 'yes'
            elif line['label'] == 1:
                answer = 'it is not possible to tell'
            elif line['label'] == 2:
                answer = 'no'
            else:
                raise NotImplementedError
            final_q = make_nli_question({'premise': premise, 'hypothesis': hypothesis})
            questions.append(final_q)
            answers.append(answer)
            rationales.append(line['reason'])
    elif args.dataset == 'anli_a3':
        anli_a2_dataset = list(load_dataset('anli')['test_r3'])
        for line in anli_a2_dataset:
            premise = line['premise']
            hypothesis = line['hypothesis']
            if line['label'] == 0:
                answer = 'yes'
            elif line['label'] == 1:
                answer = 'it is not possible to tell'
            elif line['label'] == 2:
                answer = 'no'
            else:
                raise NotImplementedError
            final_q = make_nli_question({'premise': premise, 'hypothesis': hypothesis})
            questions.append(final_q)
            answers.append(answer)
            rationales.append(line['reason'])
    elif args.dataset == 'drop':
        drop_dataset = list(jsonlines.open('downstream_datasets/drop/dev_nfl.jsonl')) + list(
            jsonlines.open('downstream_datasets/drop/dev_non_nfl.jsonl'))

        for i, example in enumerate(drop_dataset):
            final_q = make_drop_question(example)
            # answer = make_drop_answer(example)
            answer = list(map(lambda x: x[0], example['answers']))
            questions.append(final_q)
            answers.append(answer)

    elif args.dataset == 'elementary_math_qa':
        e_math_qa = list(jsonlines.open('new_dataset/dataset_from_bigbench/elementary_math/dev.jsonl'))

        for js in e_math_qa:
            final_q = make_elementary_math_qa_question(js)
            answer = make_elementary_math_qa_answer(js)
            questions.append(final_q)
            answers.append(answer)

    elif args.dataset == 'boolq':
        bool_q = list(load_dataset('boolq')['validation'])
        for js in bool_q:
            final_q = js['question'] + '?'
            answer = 'yes' if js['answer'] else 'no'
            questions.append(final_q)
            answers.append(answer)
    elif args.dataset == 'fact_checker':
        fact_ck = list(jsonlines.open('downstream_datasets/fact_cheker/dev.jsonl'))
        for js in fact_ck:
            questions.append(js['question'])
            answers.append(js['answer'])

    elif args.dataset == 'com_v':
        com_v_dataset = list(jsonlines.open('downstream_datasets/com_v/test.jsonl'))
        for js in com_v_dataset:
            questions.append(js['question'])
            answers.append(js['answer'])

    elif args.dataset == 'hotpot_qa':
        hotpot_qa = list(load_dataset('hotpot_qa','fullwiki')['validation'])
        for js in hotpot_qa:
            questions.append(js['question'])
            answers.append(js['answer'])

    elif args.dataset == 'qa_wikidata':
        qa_wikidata = list(jsonlines.open('downstream_datasets/qa_wikidata/test.jsonl'))
        for js in qa_wikidata:
            questions.append(js['question'])
            answers.append(js['answer'])

    else:
        raise ValueError("dataset is not properly defined ...")

    q_len_list = []
    for q in questions:
        q_len_list.append(len(q.split(" ")))
    q_len_mean = mean(q_len_list)

    print("dataset : {}".format(args.dataset))
    print("data size : {}".format(len(answers)))
    print("average num of words for each sample : {}".format(q_len_mean))

    for i in range(3):
        print('question:\n{}'.format(questions[i]))
        print('answer:\n{}'.format(answers[i]))

    # exit()

    return questions, answers, rationales


# Create dataset object before dataloader ...
class MyDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        if args.inference_split == 'test':
            self.questions, self.answers, self.rationales = data_reader(args)
        elif args.inference_split == 'train':
            self.questions, self.answers, self.rationales = load_train_set_for_test(args.dataset)

        self.args = args

        logger.info('original dataset size : {}'.format(len(self.questions)))

        # if args.limit_dataset_size > 0 and len(self.questions) > args.limit_dataset_size:
        #     self.questions = self.questions[:args.limit_dataset_size]
        #     self.answers = self.answers[:args.limit_dataset_size]
        #     self.rationales = self.rationales[:args.limit_dataset_size]

        logger.info('inference dataset size : {}'.format(len(self.questions)))

        if args.limit_dataset_size > 0 and len(self.questions) > args.limit_dataset_size:
            self.len = args.limit_dataset_size
        else:
            self.len = len(self.questions)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        input = self.questions[index]
        output = self.answers[index]
        return input, output


def setup_data_loader(args):
    # fix randomness of dataloader to ensure reproducibility
    # https://pytorch.org/docs/stable/notes/randomness.html
    fix_seed(args.random_seed)
    worker_seed = torch.initial_seed() % 2 ** 32
    print("worker_seed : {}".format(worker_seed))

    def seed_worker(worker_id):
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(worker_seed)

    dataloader_num_workers = multiprocessing.cpu_count()
    dataloader_num_workers = min(dataloader_num_workers, args.max_num_worker)
    print("dataloader_num_workers: " + str(dataloader_num_workers))

    dataset = MyDataset(args)

    # logger.info('original dataset size: {}'.format(len(dataset)))
    # logger.info('inference dataset size: {}'.format(len(dataset)))

    dataloader = torch.utils.data.DataLoader(dataset,
                                             shuffle=False,
                                             batch_size=args.minibatch_size,
                                             drop_last=False,
                                             num_workers=dataloader_num_workers,
                                             worker_init_fn=seed_worker,
                                             generator=g,
                                             pin_memory=True,
                                             sampler=torch.utils.data.SequentialSampler(dataset))

    return dataloader


# ver 0.2
def answer_cleansing(args, pred, must_choice=False, verbose=True):
    if verbose:
        print("pred_before : " + pred)

    if args.method == 'zero_shot':
        answer_flag = True

    if args.method in (
            "few_shot", "few_shot_cot", "auto_cot", "retrieval_few_shot_cot", "retrieval_few_shot",
            'my_random_sample_few_shot',
            'my_random_sample_few_shot_cot', 'lm_retrieval_few_shot_cot', 'lm_retrieval_few_shot','lm_retrieval_few_shot_cot_but_no_thinking'):
        preds = pred.split(args.direct_answer_trigger_for_fewshot)
        answer_flag = True if len(preds) > 1 else False
        pred = preds[-1]
    if args.method == 'zero_shot_cot':
        if 'the answer is' in pred:
            preds = pred.split('the answer is')
        elif 'The answer is' in pred:
            preds = pred.split('The answer is')
        else:
            preds = [pred]
        answer_flag = True if len(preds) > 1 else False
        pred = preds[-1]


    if args.dataset in ("aqua", "commonsensqa", 'openbookqa'):
        pred = re.findall(r'A|B|C|D|E', pred)
    elif args.dataset == "bigbench_date":
        pred = re.findall(r'A|B|C|D|E|F', pred)
    elif args.dataset in ("object_tracking"):
        pred = re.findall(r'A|B|C', pred)
    elif args.dataset in ("gsm8k", "addsub", "multiarith", "svamp", "singleeq"):
        if must_choice:
            pred = re.findall(r'A|B|C|D', pred)
        else:
            pred = pred.replace(",", "")
            pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]
    elif args.dataset in ("strategyqa", "coin_flip", "strategyqa_small"):
        pred = pred.lower()
        pred = re.sub("\"|\'|\n|\.|\s|\:|\,", " ", pred)
        pred = pred.split(" ")
        pred = [i for i in pred if i in ("yes", "no")]
    elif args.dataset == "last_letters":
        pred = re.sub("\"|\'|\n|\.|\s", "", pred)
        pred = [pred]
    elif 'nli' in args.dataset:
        pred = re.findall(r'yes|no|it is not possible to tell', pred)
    elif args.dataset == 'drop':
        # print('pred after split:{}'.format(repr(pred)))
        if len(pred) == 0:
            return ('answer_parsing_error',)
        if pred == '.':
            return pred

        # pred = ' '.join(pred.split(' and ')).split(', ')
        pred = pred.replace(' and ', ', ').split(', ')
        pred = list(map(lambda x: x.strip(), pred))
        pred = tuple(sorted(pred))
        if verbose:
            print("pred_after : ", pred)
        if len(pred) == 0:
            return ('answer_parsing_error',)
        return pred
    elif args.dataset == 'qa_wikidata':
        if len(pred) == 0:
            return 'answer_parsing_error'
        if pred == '.':
            return pred
        if pred[-1] == '.':
            pred = pred[:-1]
        # pred = ' '.join(pred.split(' and ')).split(', ')
        pred = pred.replace(' and ', ', ').split(', ')
        pred = list(map(lambda x: x.strip(), pred))
        pred = tuple(sorted(pred))
        if verbose:
            print("pred_after : ", pred)
        if len(pred) == 0:
            return 'answer_parsing_error'
        return pred
        # pred = [pred]
    elif args.dataset == 'elementary_math_qa':
        pred = re.findall(r'A|B|C|D|E|F|G|H|I|J|K|L|M', pred)
    elif args.dataset == 'boolq':

        pred = pred.lower()
        pred = re.sub("\"|\'|\n|\.|\s|\:|\,", " ", pred)
        pred = pred.split(" ")
        pred = [i for i in pred if i in ("yes", "no")]
        if len(pred) == 0:
            pred = 'answer_parsing_error'
            return pred
        else:
            pred = [pred[0]]

    elif args.dataset == 'fact_checker':
        pred = re.findall('true|false',pred)
    elif args.dataset == 'com_v':
        pred = re.findall('A|B',pred)
    elif args.dataset == 'com_e':
        pred = re.findall('A|B|C',pred)
    elif args.dataset == 'hotpot_qa':
        if len(pred) == 0:
            return 'answer_parsing_error'
        if pred == '.':
            return pred

        return pred


    else:
        raise ValueError("dataset is not properly defined ...")

    # If there is no candidate in list, null is set.
    if len(pred) == 0:
        pred = ""
    else:
        if args.method in ("few_shot", "few_shot_cot", "auto_cot", "retrieval_few_shot_cot", "retrieval_few_shot",
                           'my_random_sample_few_shot', 'my_random_sample_few_shot_cot', 'lm_retrieval_few_shot_cot',
                            'lm_retrieval_few_shot',
                           'zero_shot_cot','zero_shot','lm_retrieval_few_shot_cot_but_no_thinking'):
            if answer_flag or args.method == 'lm_retrieval_few_shot_cot_but_no_thinking':
                # choose the first element in list ...
                pred = pred[0]
            else:
                # choose the last element in list ...
                pred = pred[-1]
        elif args.method in ("zero_shot", "zero_shot_cot"):
            # choose the first element in list ...
            pred = pred[0]
        else:
            raise ValueError("method is not properly defined ...")

    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred != "":
        if pred[-1] == ".":
            pred = pred[:-1]
    if verbose:
        print("pred_after : " + pred)

    if len(pred) == 0:
        pred = 'answer_parsing_error'

    return pred


def create_demo_text(args, cot_flag):
    x, z, y = [], [], []
    if args.demo_path.endswith('json'):
        # original auto_cot format
        with open(args.demo_path, encoding="utf-8") as f:
            json_data = json.load(f)
            json_data = json_data["demo"]
            for line in json_data:
                if 'few_shot' in args.method:
                    assert 'A:' not in line['question']
                x.append(line["question"])
                z.append(line["rationale"])
                y.append(line["pred_ans"])
    elif args.demo_path.endswith('jsonl'):
        # my format
        lines = jsonlines.open(args.demo_path)
        for line in lines:
            x.append(line['question'])
            z.append(line['rationale'])
            y.append(line['answer'])

    index_list = list(range(len(x)))

    if args.method == 'auto_cot':
        demo_text = ""
        for i in index_list:
            if cot_flag:
                demo_text += x[i] + " " + z[i] + " " + \
                             args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
            else:
                demo_text += x[i] + " " + args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
    else:
        demos = []
        for i in index_list:
            tmp_demo = create_single_demo(x[i], z[i], y[i], args.direct_answer_trigger_for_fewshot, cot_flag,
                                          args.dataset)
            demos.append(tmp_demo)

        demos = demos[:args.num_demo]

        demo_text = concat_demos(demos)
        # create_single_demo
    return demo_text


def answer_cleansing_zero_shot(args, pred, must_choice=False):
    pred = pred.strip()
    if args.dataset in ("aqua", "commonsensqa"):
        pred = re.findall(r'A|B|C|D|E', pred)
    elif args.dataset == "bigbench_date":
        pred = re.findall(r'A|B|C|D|E|F', pred)
    elif args.dataset in ("object_tracking"):
        pred = re.findall(r'A|B|C', pred)
    elif args.dataset in ("gsm8k", "addsub", "multiarith", "svamp", "singleeq"):
        if must_choice:
            pred = re.findall(r'A|B|C|D', pred)
        else:
            pred = pred.replace(",", "")
            pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]
    elif args.dataset in ("strategyqa", "coin_flip", "strategyqa_small"):
        pred = pred.lower()
        pred = re.sub("\"|\'|\n|\.|\s|\:|\,", " ", pred)
        pred = pred.split(" ")
        pred = [i for i in pred if i in ("yes", "no")]
    elif args.dataset == "last_letters":
        pred = re.sub("\"|\'|\n|\.|\s", "", pred)
        pred = [pred]
    else:
        raise ValueError("dataset is not properly defined ...")

    # If there is no candidate in list, null is set.
    if len(pred) == 0:
        pred = ""
    else:
        # choose the first element in list ...
        pred = pred[0]

    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred != "":
        if pred[-1] == ".":
            pred = pred[:-1]

    return pred


def load_gt_demo_pool(dataset, direct_answer_trigger_for_fewshot, cot_flag):
    assert dataset in ['gsm8k', 'aqua', 'strategyqa_small']
    # assert method in
    if dataset == 'gsm8k':
        gsm8k_dataset = load_dataset('gsm8k', 'main')
        gsm8k_demo_pool = []
        for tmp_js in gsm8k_dataset['train']:
            tmp_rationale = make_gsm8k_rationale(tmp_js)
            tmp_answer = make_gsm8k_answer(tmp_js)
            tmp_question = tmp_js['question']

            tmp_demo = create_single_demo(tmp_question, tmp_rationale, tmp_answer, direct_answer_trigger_for_fewshot,
                                          cot_flag, 'gsm8k')
            gsm8k_demo_pool.append(
                {'demonstration': tmp_demo, 'question': tmp_question, 'rationale': tmp_rationale, 'answer': tmp_answer})

        return gsm8k_demo_pool
    elif dataset == 'aqua':
        aqua_dataset = load_dataset('aqua_rat')
        aqua_demo_pool = []
        for i, tmp_js in enumerate(aqua_dataset['train']):
            # print(i)

            tmp_demo_dict = transform_original_aqua_into_x_z_y(tmp_js, direct_answer_trigger_for_fewshot, cot_flag)
            # tmp_question = make_aqua_x(tmp_js)
            # if tmp_question.startswith('Q:'):
            #     tmp_question = tmp_question[2:]
            # elif tmp_question.startswith('Q: '):
            #     tmp_question = tmp_question[3:]
            #
            # tmp_rationale = make_aqua_rationale(tmp_js)
            # tmp_answer = tmp_js['correct']
            #
            # tmp_demo = create_single_demo(tmp_question,tmp_rationale,tmp_answer,direct_answer_trigger_for_fewshot,cot_flag)

            aqua_demo_pool.append(
                tmp_demo_dict
            )

        return aqua_demo_pool

    elif dataset == 'strategyqa_small':
        strategy_qa_demo_pool = []

        strategy_qa_dataset = list(load_dataset("metaeval/strategy-qa")['train'])
        random.Random(1208).shuffle(strategy_qa_dataset)
        assert strategy_qa_dataset[0]['question'] == 'Is Christmas celebrated during winter?'

        strategy_qa_dataset = strategy_qa_dataset[250:]

        for i, tmp_js in enumerate(strategy_qa_dataset):
            tmp_question = tmp_js['question']
            tmp_answer = 'yes' if tmp_js['answer'] else 'no'
            tmp_rationale = ' '.join(tmp_js['facts'])
            tmp_demo = create_single_demo(tmp_question, tmp_rationale, tmp_answer, direct_answer_trigger_for_fewshot,
                                          cot_flag, 'gsm8k')
            strategy_qa_demo_pool.append(
                {'demonstration': tmp_demo, 'question': tmp_question, 'rationale': tmp_rationale, 'answer': tmp_answer})

        return strategy_qa_demo_pool

        # raise NotImplementedError


    else:
        raise NotImplementedError


# from tmp_data_process.extract_demos_from_inference_result import remove_line_break
def load_lm_inference_demo_pool(inp_fp, cot_flag):
    # assert cot_flag

    if 1:
        demos = list(jsonlines.open(inp_fp))
        for d in demos:
            d['demonstration'] = d['demo']
            del d['demo']

            pred_ans = d['pred_ans']
            if pred_ans[-1] == '.':
                pred_ans = pred_ans[:-1]

            d['demostration_without_rationale'] = d['question'] + ' ' + 'The answer is {}.'.format(pred_ans)

    return demos


def load_train_set_for_test(dataset, ):
    questions = []
    answers = []
    rationales = []

    if dataset == 'aqua':
        aqua_dataset = list(load_dataset('aqua_rat')['train'])
        for i, line in enumerate(aqua_dataset):
            tmp_demo_dict = transform_original_aqua_into_x_z_y(line, "ABCDEFGHIJKLMNOPQRSTUVWXYZ", 0)
            questions.append(tmp_demo_dict['question'])
            answers.append(tmp_demo_dict['answer'])
            rationales.append(tmp_demo_dict['rationale'])

    elif dataset == 'gsm8k':
        gsm8k_dataset = list(load_dataset('gsm8k', 'main')['train'])
        for i, line in enumerate(gsm8k_dataset):
            tmp_rationale = make_gsm8k_rationale(line)
            tmp_answer = make_gsm8k_answer(line)
            tmp_question = line['question']
            questions.append(tmp_question)
            rationales.append(tmp_rationale)
            answers.append(tmp_answer)
    elif dataset == 'anli_a1':
        anli_a2_dataset = list(load_dataset('anli')['train_r1'])
        for line in anli_a2_dataset:
            premise = line['premise']
            hypothesis = line['hypothesis']
            if line['label'] == 0:
                answer = 'yes'
            elif line['label'] == 1:
                answer = 'it is not possible to tell'
            elif line['label'] == 2:
                answer = 'no'
            else:
                raise NotImplementedError
            final_q = make_nli_question({'premise': premise, 'hypothesis': hypothesis})
            questions.append(final_q)
            answers.append(answer)
            rationales.append(line['reason'])
    elif dataset == 'anli_a2':
        anli_a2_dataset = list(load_dataset('anli')['train_r2'])
        for line in anli_a2_dataset:
            premise = line['premise']
            hypothesis = line['hypothesis']
            if line['label'] == 0:
                answer = 'yes'
            elif line['label'] == 1:
                answer = 'it is not possible to tell'
            elif line['label'] == 2:
                answer = 'no'
            else:
                raise NotImplementedError
            final_q = make_nli_question({'premise': premise, 'hypothesis': hypothesis})
            questions.append(final_q)
            answers.append(answer)
            rationales.append(line['reason'])
    elif dataset == 'anli_a3':
        anli_a2_dataset = list(load_dataset('anli')['train_r3'])
        for line in anli_a2_dataset:
            premise = line['premise']
            hypothesis = line['hypothesis']
            if line['label'] == 0:
                answer = 'yes'
            elif line['label'] == 1:
                answer = 'it is not possible to tell'
            elif line['label'] == 2:
                answer = 'no'
            else:
                raise NotImplementedError
            final_q = make_nli_question({'premise': premise, 'hypothesis': hypothesis})
            questions.append(final_q)
            answers.append(answer)
            rationales.append(line['reason'])
    elif dataset == 'openbookqa':
        openbook_qa = list(load_dataset("openbookqa", 'additional')['train'])
        for line in openbook_qa:
            q = line['question_stem'].strip()
            a = line['answerKey'].strip()
            r = line['fact1'].strip()
            # tmp_choices = line['choices']['text']
            # assert line['label'] == ['A','B','C','D']
            #
            # tmp_choices[0] = 'A){}'.format(tmp_choices[0])
            # tmp_choices[1] = 'B){}'.format(tmp_choices[1])
            # tmp_choices[2] = 'C){}'.format(tmp_choices[2])
            # tmp_choices[3] = 'D){}'.format(tmp_choices[3])

            choice = "Answer Choices:"
            for label, text in zip(line['choices']['label'], line['choices']['text']):
                choice += " ("
                choice += label
                choice += ") "
                choice += text
            q = q + ' ' + choice
            questions.append(q)
            answers.append(a)
            rationales.append(r)
    elif dataset == 'commonsensqa':
        commonsens_qa = list(load_dataset('commonsense_qa')['train'])
        for i, json_res in enumerate(commonsens_qa):
            choice = "Answer Choices:"
            for label, text in zip(json_res["choices"]['label'], json_res["choices"]['text']):
                choice += " ("
                choice += label
                choice += ") "
                choice += text
            questions.append(json_res["question"].strip() + " " + choice)
            answers.append(json_res["answerKey"])

    elif dataset == 'strategyqa_small':
        strategy_qa_dataset = list(load_dataset("metaeval/strategy-qa")['train'])

        random.Random(1208).shuffle(strategy_qa_dataset)

        strategy_qa_dataset_test = strategy_qa_dataset[250:]

        for line in strategy_qa_dataset_test:
            q = line['question']
            a = 'yes' if line['answer'] else 'no'

            questions.append(q)
            answers.append(a)

            tmp_rationale = ' '.join(line['facts'])

            rationales.append(tmp_rationale)
    elif dataset == 'drop':
        drop_dataset = list(jsonlines.open('downstream_datasets/drop/train_nfl.jsonl')) + list(
            jsonlines.open('downstream_datasets/drop/train_non_nfl.jsonl'))

        passage_to_examples_dict = {}
        for i, example in enumerate(drop_dataset):
            if example['passage'] in passage_to_examples_dict:
                passage_to_examples_dict[example['passage']].append(example)
            else:
                passage_to_examples_dict[example['passage']] = [example]

        filtered_drop_dataset = []
        for p, examples in passage_to_examples_dict.items():
            num_examples_after_filtering = max(1, 1 + len(examples) // 2)
            filtered_drop_dataset.extend(examples[:num_examples_after_filtering])
            # filtered_drop_dataset.extend(examples)

        for i, example in enumerate(filtered_drop_dataset):
            final_q = make_drop_question(example)
            # answer = make_drop_answer(example)
            answer = list(map(lambda x: x[0], example['answers']))
            questions.append(final_q)
            answers.append(answer)
    elif dataset == 'elementary_math_qa':
        e_math_qa = list(jsonlines.open('new_dataset/dataset_from_bigbench/elementary_math/train.jsonl'))

        for js in e_math_qa:
            final_q = make_elementary_math_qa_question(js)
            answer = make_elementary_math_qa_answer(js)
            questions.append(final_q)
            answers.append(answer)

    elif dataset == 'boolq':
        bool_q = list(load_dataset('boolq')['train'])
        for js in bool_q:
            final_q = js['question'] + '?'
            answer = 'yes' if js['answer'] else 'no'
            questions.append(final_q)
            answers.append(answer)

    elif dataset =='fact_checker':
        fact_ck = list(jsonlines.open('downstream_datasets/fact_cheker/train.jsonl'))
        for js in fact_ck:
            questions.append(js['question'])
            answers.append(js['answer'])

    elif dataset == 'com_v':
        com_v_dataset = list(jsonlines.open('downstream_datasets/com_v/train.jsonl'))
        for js in com_v_dataset:
            questions.append(js['question'])
            answers.append(js['answer'])


    elif dataset == 'hotpot_qa':
        hotpot_qa = list(load_dataset('hotpot_qa','fullwiki')['train'])
        for js in hotpot_qa:
            questions.append(js['question'])
            answers.append(js['answer'])
    elif dataset == 'qa_wikidata':
        qa_wikidata = list(jsonlines.open('downstream_datasets/qa_wikidata/train.jsonl'))
        random.Random(42).shuffle(qa_wikidata)
        qa_wikidata = qa_wikidata[:100000]
        for js in qa_wikidata:
            questions.append(js['question'])
            answers.append(js['answer'])







    else:
        logger.info('do not support dataset {}'.format(dataset))
        raise NotImplementedError

    print("dataset : {}".format(dataset))
    print('dataset size : {}'.format(len(questions)))
    print("question size : {}".format(len(questions)))
    print("answer size : {}".format(len(answers)))
    assert len(questions) == len(answers)
    q_len_list = []
    for q in questions:
        q_len_list.append(len(q.split(" ")))
    q_len_mean = mean(q_len_list)

    # print("q size : {}".format(len(questions)))
    print("average num of words for each sample : {}".format(q_len_mean))

    return questions, answers, rationales
    pass


def get_kmeans_clustered_idx(embeddings, num_clusters):
    clustering_model = KMeans(n_clusters=num_clusters)
    clustering_model.fit(embeddings)
    return clustering_model.labels_


# from openai_account_manager import OpenAI_Account_Manager

# account_manager = OpenAI_Account_Manager()

def second_transform(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)

    return d, h, m, s


def generate_one_demo_prompt(demo_dict, question):
    demo = demo_dict['demonstration']
    result = concat_demos([demo]) + question

    return result
