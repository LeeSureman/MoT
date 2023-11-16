import sys
sys.path.append('../')
import argparse
import json
import os
from V7.data_process_utils import make_aqua_x, make_aqua_rationale, make_gsm8k_rationale, make_gsm8k_answer, \
    make_nli_question, make_openbookqa_x, make_csqa_x,make_drop_question

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['openbookqa', 'aqua', 'gsm8k', 'strategyqa', 'commonsensqa', 'nli', 'drop'])

    args = parser.parse_args()

    dataset_dir = 'manual_demos_original/{}'.format(args.dataset)

    if args.dataset == 'openbookqa':
        questions = open('{}/question.txt'.format(dataset_dir)).readlines()
        rationales = open('{}/rationale.txt'.format(dataset_dir)).readlines()
        answers = open('{}/answer.txt'.format(dataset_dir)).readlines()
        # choices = json.load(open('{}/choice.txt'.format(dataset_dir)))
        choices = open('{}/choice.txt'.format(dataset_dir)).readlines()
        demos = []

        for i, q in enumerate(questions):
            z = rationales[i].strip()
            y = answers[i].strip()
            choice = json.loads(choices[i])
            final_q = make_aqua_x({'question': q, 'options': choice})
            final_tmp_demo = {'question': final_q, 'rationale': z, 'answer': y}
            demos.append(final_tmp_demo)
        pass
    elif args.dataset == 'aqua':
        demos = []
        questions = open('{}/question.txt'.format(dataset_dir)).readlines()
        rationales = open('{}/rationale.txt'.format(dataset_dir)).readlines()
        answers = open('{}/answer.txt'.format(dataset_dir)).readlines()
        choices = (open('{}/choice.txt'.format(dataset_dir))).readlines()
        for i, q in enumerate(questions):
            x = q.strip()
            z = rationales[i].strip()
            y = answers[i].strip()
            choice = json.loads(choices[i].strip())
            # tmp_demo = {'question': x, 'rationale': z, 'answer': y, 'options': choice}
            final_q = make_aqua_x({'question': x, 'options': choice})
            final_tmp_demo = {'question': final_q, 'rationale': z, 'answer': y}
            demos.append(final_tmp_demo)
        pass
    elif args.dataset == 'gsm8k':
        demos = []
        questions = open('{}/question.txt'.format(dataset_dir)).readlines()
        rationales = open('{}/rationale.txt'.format(dataset_dir)).readlines()
        answers = open('{}/answer.txt'.format(dataset_dir)).readlines()
        for i, q in enumerate(questions):
            x = q.strip()
            z = rationales[i].strip()
            y = answers[i].strip()
            tmp_demo = {'question': x, 'rationale': z, 'answer': y}
            demos.append(tmp_demo)
        pass
    elif args.dataset == 'strategyqa':
        demos = []
        questions = open('{}/question.txt'.format(dataset_dir)).readlines()
        rationales = open('{}/rationale.txt'.format(dataset_dir)).readlines()
        answers = open('{}/answer.txt'.format(dataset_dir)).readlines()
        for i, q in enumerate(questions):
            x = q.strip()
            z = rationales[i].strip()
            y = answers[i].strip()
            tmp_demo = {'question': x, 'rationale': z, 'answer': y}
            demos.append(tmp_demo)
        pass
    elif args.dataset == 'commonsensqa':
        demos = []
        questions = open('{}/question.txt'.format(dataset_dir)).readlines()
        rationales = open('{}/rationale.txt'.format(dataset_dir)).readlines()
        answers = open('{}/answer.txt'.format(dataset_dir)).readlines()
        choices = (open('{}/choice.txt'.format(dataset_dir))).readlines()
        for i, q in enumerate(questions):
            x = q.strip()
            z = rationales[i].strip()
            y = answers[i].strip()
            choice = json.loads(choices[i].strip())
            # tmp_demo = {'question': x, 'rationale': z, 'answer': y, 'options': choice}
            final_q = make_aqua_x({'question':x,'options':choice})
            final_tmp_demo = {'question':final_q,'rationale':z,'answer':y}
            demos.append(final_tmp_demo)
        pass
    elif 'nli' in args.dataset:
        demos = []
        premises = open('{}/premise.txt'.format(dataset_dir)).readlines()
        hypotheses = open('{}/hypothesis.txt'.format(dataset_dir)).readlines()
        answers = open('{}/answer.txt'.format(dataset_dir)).readlines()
        rationales = open('{}/rationale.txt'.format(dataset_dir)).readlines()
        for i in range(len(premises)):
            premise = premises[i].strip()
            hypothesis = hypotheses[i].strip()
            answer = answers[i].strip()
            rationale = rationales[i]
            final_q = make_nli_question({'premise':premise,'hypothesis':hypothesis})
            final_tmp_demo = {'question':final_q,'rationale':rationale, 'answer':answer}
            demos.append(final_tmp_demo)
    elif 'drop' == args.dataset:
        demos = []
        passages = open('{}/passage.txt'.format(dataset_dir)).readlines()
        qs_for_passage = open('{}/q_for_passage.txt'.format(dataset_dir)).readlines()
        rationales = open('{}/rationale.txt'.format(dataset_dir)).readlines()
        answers = open('{}/answer.txt'.format(dataset_dir)).readlines()
        for i in range(len(passages)):
            passage = passages[i].strip()
            q_for_passage = qs_for_passage[i].strip()
            rationale = rationales[i].strip()
            answer = answers[i].strip()
            final_q = make_drop_question({'passage':passage,'question':q_for_passage})
            final_tmp_demo = {'question':final_q,'rationale':rationale,'answer':answer}
            demos.append(final_tmp_demo)






    # return demos
    manual_demos_transformed_dir = 'manual_demos_transformed'
    os.makedirs(manual_demos_transformed_dir, exist_ok=True)
    manual_demos_transformed_fp = 'manual_demos_transformed/{}.jsonl'.format(args.dataset)
    manual_demos_transformed_f = open(manual_demos_transformed_fp, 'w')
    for demo in demos:
        print(json.dumps(demo), file=manual_demos_transformed_f)

        pass

12345
