import argparse
import json
import os
import jsonlines


def answer_json_to_strings(answer):
    """
    Takes an answer JSON blob from the DROP data release and converts it into strings used for
    evaluation.
    """
    if "number" in answer and answer["number"]:
        return tuple([str(answer["number"])]), "number"
    elif "spans" in answer and answer["spans"]:
        return tuple(answer["spans"]), "span" if len(answer["spans"]) == 1 else "spans"
    elif "date" in answer:
        return (
            tuple(
                [
                    "{0} {1} {2}".format(
                        answer["date"]["day"], answer["date"]["month"], answer["date"]["year"]
                    )
                ]
            ),
            "date",
        )
    else:
        raise ValueError(
            f"Answer type not found, should be one of number, spans or date at: {json.dumps(answer)}"
        )

def transform_dev_file_to_example_js_s(dev_dict):
    new_dev_dict = {}
    #这个新的是passage id对应的所有passage和question对 （）

    for k,v in dev_dict.items():
        passage = v['passage']

        tmp_js_s_list = []

        for qa_pair in v['qa_pairs']:
            tmp_js = {'passage':passage}
            tmp_js['question'] = qa_pair['question']
            tmp_js['answers'] = [answer_json_to_strings(qa_pair['answer'])]

            for a in qa_pair['validated_answers']:
                tmp_js['answers'].append(answer_json_to_strings(a))
            tmp_js_s_list.append(tmp_js)

        new_dev_dict[k] = tmp_js_s_list

    nfl_js_s = []
    non_nfl_js_s = []
    for k,v in new_dev_dict.items():
        if 'nfl' in k:
            nfl_js_s.extend(v)
        else:
            non_nfl_js_s.extend(v)

    return {'nfl':nfl_js_s,'non_nfl':non_nfl_js_s}


def transform_train_file_to_example_js_s(train_dict):
    new_train_dict = {}
    #这个新的是passage id对应的所有passage和question对 （）

    for k,v in train_dict.items():
        passage = v['passage']
        tmp_js_s_list = []


        for qa_pair in v['qa_pairs']:
            tmp_js = {'passage':passage}
            tmp_js['question'] = qa_pair['question']
            tmp_js['answers'] = [answer_json_to_strings(qa_pair['answer'])]

            # for a in qa_pair['validated_answers']:
            #     tmp_js['answers'].append(answer_json_to_strings(a))
            tmp_js_s_list.append(tmp_js)

        new_train_dict[k] = tmp_js_s_list

    nfl_js_s = []
    non_nfl_js_s = []
    for k,v in new_train_dict.items():
        if 'nfl' in k:
            nfl_js_s.extend(v)
        else:
            non_nfl_js_s.extend(v)

    return {'nfl':nfl_js_s,'non_nfl':non_nfl_js_s}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev_fp',default='new_dataset/drop_dataset/drop_dataset_dev.json')
    parser.add_argument('--train_fp',default='new_dataset/drop_dataset/drop_dataset_train.json')
    parser.add_argument('--out_dir',default='new_dataset/drop_dataset_transformed')

    args = parser.parse_args()

    os.makedirs(args.out_dir,exist_ok=True)

    dev_dict = json.load(open(args.dev_fp))
    train_dict = json.load(open(args.train_fp))

    # transform_dev_file_to_example_js_s(dev_dict)
    dev_js_s = transform_dev_file_to_example_js_s(dev_dict)
    train_js_s = transform_train_file_to_example_js_s(train_dict)

    dev_js_s_nfl = dev_js_s['nfl']
    dev_js_s_non_nfl = dev_js_s['non_nfl']

    with jsonlines.open('{}/dev_nfl.jsonl'.format(args.out_dir),'w') as out_f:
        for js in dev_js_s_nfl:
            out_f.write(js)

    with jsonlines.open('{}/dev_non_nfl.jsonl'.format(args.out_dir),'w') as out_f:
        for js in dev_js_s_non_nfl:
            out_f.write(js)


    train_js_s_nfl = train_js_s['nfl']
    train_js_s_non_nfl = train_js_s['non_nfl']

    with jsonlines.open('{}/train_nfl.jsonl'.format(args.out_dir),'w') as out_f:
        for js in train_js_s_nfl:
            out_f.write(js)

    with jsonlines.open('{}/train_non_nfl.jsonl'.format(args.out_dir),'w') as out_f:
        for js in train_js_s_non_nfl:
            out_f.write(js)



