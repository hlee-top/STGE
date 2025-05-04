from utils import *
import argparse


def few_shot_data_process(args):
    all_data = read_json(args.convert_path+"train.json")
    random.shuffle(all_data)
    result_data = []
    for data in all_data:
        result_data.append(data)
        if len(result_data) == args.shot_num:
            break
    save_json(result_data, args.save_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int, help="random seed.")
    parser.add_argument("--task_type", default="eae", type=str, choices=['ner', 're', 'eae'],
                        help="task name.")
    parser.add_argument("--dataset_type", default="ace", type=str, choices=['rams', 'wiki', 'ace'],
                        help="dataset name.")
    parser.add_argument("--shot_num", default=20, type=int, help="shot number")

    args = parser.parse_args()
    if args.dataset_type == "rams":
        args.convert_path = "../../dataset/RAMS_1.0/data/convert/{}/".format(args.task_type)
    elif args.dataset_type == "wiki":
        args.convert_path = "../../dataset/WIKIEVENT/convert/{}/".format(args.task_type)
    elif args.dataset_type == "ace":
        args.convert_path = "../../dataset/ACE2005/convert/{}/".format(args.task_type)
    args.template_path = args.convert_path + "template.json"
    args.save_path = args.convert_path + "{}-shot_data.json".format(str(args.shot_num))
    args.shot_data_path = args.convert_path + "{}-shot_data.json".format(str(args.shot_num))

    print(args)
    set_seed(args)

    few_shot_data_process(args)
    