import copy
import string


def make_aqua_x(tmp_js):
    question = tmp_js['question']
    options = tmp_js['options']

    # print('options:{}'.format(options))

    question = question.strip()

    choice = "(" + "(".join(options)
    choice = choice.replace("(", " (").replace(")", ") ")
    choice = "Answer Choices:" + choice

    # print('choice:{}'.format(choice))

    question = question + " " + choice

    return question


def make_gsm8k_answer(tmp_js):
    answer = tmp_js["answer"].split("#### ")[-1]
    return answer


def make_gsm8k_rationale(tmp_js):
    rationale = tmp_js["answer"].split("#### ")[0]
    rationale = list(filter(lambda x: len(x) > 0, rationale.strip().split('\n')))
    rationale = list(map(lambda x: x + '.' if (x[-1] not in string.punctuation) else x, rationale))
    rationale = ' '.join(rationale)

    return rationale


def make_aqua_rationale(tmp_js):
    rationale = tmp_js['rationale']
    rationale = list(filter(lambda x: len(x) > 0, rationale.strip().split('\n')))[:-1]
    rationale = list(map(lambda x: x + '.' if (x[-1] not in string.punctuation) else x, rationale))
    rationale = ' '.join(rationale)

    return rationale


def make_nli_question(tmp_js):
    premise = tmp_js['premise']
    hypothesis = tmp_js['hypothesis']
    question = 'Premise:\n"{}"\nBased on this premise, can we conclude the hypothesis "{}" is true?\nOPTIONS:\n- yes\n- no\n- it is not possible to tell' \
        .format(premise, hypothesis)

    return question


def make_openbookqa_x(tmp_js):
    q = tmp_js['question_stem'].strip()

    choice = "Answer Choices:"
    for c in tmp_js['choices']:
        choice += " ("
        choice += c["label"]
        choice += ") "
        choice += c["text"]
    q = q + ' ' + choice
    return q


def make_csqa_x(tmp_js):
    q = tmp_js["question"]

    choice = "Answer Choices:"
    for c in tmp_js["choices"]:
        choice += " ("
        choice += c["label"]
        choice += ") "
        choice += c["text"]

    q = q + ' ' + choice
    return q


def create_single_demo(x, z, y, direct_answer_trigger_for_fewshot, cot_flag, dataset):
    '''

    :param x: question (maybe including options)
    :param z: rationale
    :param y: answer
    :return:
    '''
    if 'nli' not in dataset:
        assert not x.startswith('Q:')
        x = 'Q: ' + x + '\n' + 'A:'
        if cot_flag:
            result = ' '.join([x, z, direct_answer_trigger_for_fewshot, y])
        else:
            result = ' '.join([x, direct_answer_trigger_for_fewshot, y])
    else:
        x = x + '\n' + 'A:'
        if cot_flag:
            result = ' '.join([x, z, direct_answer_trigger_for_fewshot, y])
        else:
            result = ' '.join([x, direct_answer_trigger_for_fewshot, y])

    return result


def concat_demos(demos):
    demos = copy.deepcopy(demos)
    # result = ''
    for i, demo in enumerate(demos):
        if demo[-1] != '.':
            demos[i] += '.'
        demos[i] += '\n\n'

    result = ''
    for demo in demos:
        result += demo

    # result = ".\n\n".join(demos)
    # result += ".\n\n"

    return result


def transform_original_aqua_into_x_z_y(tmp_js, direct_answer_trigger_for_fewshot, cot_flag):
    tmp_question = make_aqua_x(tmp_js)

    if tmp_question.startswith('Q:'):
        tmp_question = tmp_question[2:]
    elif tmp_question.startswith('Q: '):
        tmp_question = tmp_question[3:]

    tmp_rationale = make_aqua_rationale(tmp_js)
    tmp_answer = tmp_js['correct']
    tmp_demo = create_single_demo(tmp_question, tmp_rationale, tmp_answer, direct_answer_trigger_for_fewshot, cot_flag,
                                  'aqua')
    tmp_demo_dict = {'demonstration': tmp_demo, 'question': tmp_question, 'rationale': tmp_rationale,
                     'answer': tmp_answer}
    return tmp_demo_dict


def transform_original_strategyqa_into_x_z_y(tmp_js, direct_answer_trigger_for_fewshot, cot_flag):
    tmp_question = tmp_js['question']
    tmp_answer = 'yes' if tmp_js['answer'] else 'no'
    tmp_rationale = ' '.join(tmp_js['facts'])
    tmp_demo = create_single_demo(tmp_question, tmp_rationale, tmp_answer, direct_answer_trigger_for_fewshot,
                                  cot_flag, 'strategyqa')
    tmp_demo_dict = {'demonstration': tmp_demo, 'question': tmp_question, 'rationale': tmp_rationale,
                     'answer': tmp_answer}
    return tmp_demo_dict


def transform_original_gsm8k_into_x_z_y(tmp_js, direct_answer_trigger_for_fewshot, cot_flag):
    tmp_rationale = make_gsm8k_rationale(tmp_js)
    tmp_answer = make_gsm8k_answer(tmp_js)
    tmp_question = tmp_js['question']

    tmp_demo = create_single_demo(tmp_question, tmp_rationale, tmp_answer, direct_answer_trigger_for_fewshot,
                                  cot_flag, 'gsm8k')
    tmp_demo_dict = {'demonstration': tmp_demo, 'question': tmp_question, 'rationale': tmp_rationale,
                     'answer': tmp_answer}

    return tmp_demo_dict


def extract_premise_and_hypothesis(nli_question):
    tmp_split = nli_question.split('\nBased on this premise, can we conclude the hypothesis ')
    assert tmp_split[0].startswith('Premise:\n')
    tmp_split[0] = tmp_split[0][len('Premise:\n'):]
    tmp_split[0] = tmp_split[0][1:-1]
    assert tmp_split[1].endswith(' is true?\nOPTIONS:\n- yes\n- no\n- it is not possible to tell\nA:')
    tmp_split[1] = tmp_split[1][:len(' is true?\nOPTIONS:\n- yes\n- no\n- it is not possible to tell\nA:')]
    tmp_split[1] = tmp_split[1][1:-1]

    return {'premise': tmp_split[0], 'hypothesis': tmp_split[1]}


def make_drop_question(tmp_js):
    passage = tmp_js['passage']
    question = tmp_js['question']
    # answers_spans = tmp_js['answers_spans']['spans']
    # answer = ', '.join(answers_spans)
    final_q = '{} {}'.format(passage, question)
    return final_q


def make_drop_answer(tmp_js):
    answers_spans = tmp_js['answers_spans']['spans']
    answer = ', '.join(answers_spans)
    return answer


def make_elementary_math_qa_question(tmp_js):
    CHOICE_LABEL_S = "ABCDEFGHIJKLMN"


    question = tmp_js['input']
    target_scores = tmp_js['target_scores']

    choice = "Answer Choices:"

    for i, target_score in enumerate(target_scores):
        choice += " ("
        choice += CHOICE_LABEL_S[i]
        choice += ") "
        choice += target_score[0]

    result = question + ' ' + choice
    return result


def make_elementary_math_qa_answer(tmp_js):
    CHOICE_LABEL_S = "ABCDEFGHIJKLMN"
    target_scores = tmp_js['target_scores']

    for i, target_score in enumerate(target_scores):
        if target_score[1] == 1:
            return CHOICE_LABEL_S[i]