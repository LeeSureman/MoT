import argparse
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--filter_error',required=True)
    parser.add_argument('--output_dir',required=True)

    args = parser.parse_args()

    args.zero_shot_log_fp = 'log/{}_zero_shot_cot.log'.format(args.dataset)

    corpus = []
    questions = []
    rationales = []
    gold_ans_s = []
    pred_ans_s = []

    with open(args.zero_shot_log_fp, "r", encoding="utf-8") as fp:
        answer_seg = ""
        for line in fp:
            if "Q: " in line:
                c_question = line.strip()
            if "A: " in line:
                answer_seg = line
            elif "Therefore" in line and "the answer" in line:
                c_rationale = answer_seg

            elif answer_seg != "":
                answer_seg += line
            if "pred_mode" in line:
                c_pred_ans = line.split(":")[1].strip()
            if "GT :" in line:
                c_gold_ans = line.split(":")[1].strip()

                c_rationale = c_rationale.replace("A: Let's think step by step.", "Let's think step by step.")
                c_question = c_question + "\nA:"

                corpus.append(c_question)
                questions.append(c_question)
                rationales.append(c_rationale)
                pred_ans_s.append(c_pred_ans)
                gold_ans_s.append(c_gold_ans)
                answer_seg = ""

    n_example = len(questions)

    logger.info('questions:{}'.format(len(questions)))
    logger.info('gold_ans_s:{}'.format(len(gold_ans_s)))
    logger.info('pred_ans_s:{}'.format(len(pred_ans_s)))

    assert len(questions) == len(rationales)
    assert len(questions) == len(gold_ans_s)
    assert len(questions) == len(pred_ans_s)





    demos = []

    for i, q in enumerate(questions):
        tmp_demo = {}
        tmp_demo['question'] = q
        tmp_demo['rationale'] = rationales[i]
        tmp_demo['pred'] = pred_ans_s[i]
        tmp_demo['gold'] = gold_ans_s[i]

        demos.append(tmp_demo)

    logger.info('all demos:{}'.format(len(demos)))

    correct_demos = list(filter(lambda x:x['pred'] == x['gold'],demos))


    logger.info('correct demos:{}'.format(len(correct_demos)))









