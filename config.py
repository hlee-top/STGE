import argparse
import os
from utils import set_seed, get_current_time
import torch


def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--llm_name', default="llama3.1_8b_instruct", type=str, help="large language model.")
    parser.add_argument("--seed", default=42, type=int, help="random seed.")
    parser.add_argument("--task_type", default="eae", type=str, choices=['ner', 're', 'eae'],
                        help="task name.")
    parser.add_argument("--dataset_type", default="ace", type=str, choices=['rams', 'wiki', 'ace', 'nerd'],
                        help="dataset name.")
    parser.add_argument("--max_new_tokens", default=256, type=int, help="model max_new_tokens")
    parser.add_argument("--method_type", default="base", type=str, choices=["stge"],
                        help="select method type")
    parser.add_argument("--logit_strategy", default="llm", type=str, choices=["fusion"], help="logit strategy.")
    parser.add_argument("--window_size", default=250, type=int,
                        help="for document exceeding the length constraint, add a window centering at "
                             "the trigger word and drop words outside this window")
    parser.add_argument("--shot_num", default=20, type=int, help="shot number")
    parser.add_argument("--demo_num", default=2, type=int, choices=[0, 1, 2, 3], help="demo number")
    parser.add_argument('--value_num', type=int, default=2)
    parser.add_argument('--filter', action='store_true')
    parser.add_argument('--rule', action='store_true')


    # scorer_data_process argument
    parser.add_argument("--max_num", type=int, help="max number")
    parser.add_argument("--pred_max_num", type=int, help="pred max number")

    parser.add_argument("--train_save_path", type=str)
    parser.add_argument("--dev_save_path", type=str)
    parser.add_argument('--threshold', type=float, default=0.5)


    # scorer_train argument
    parser.add_argument("--scorer_train_data_path", default="",
                        type=str, help="train data")
    parser.add_argument("--scorer_dev_data_path", default="", type=str,
                        help="dev data")
    parser.add_argument("--scorer_test_data_path", default="", type=str,
                        help="test data")
    parser.add_argument("--load_scorer_path", default="", type=str, help="model path")
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--warmup_proportion', type=int, default=0.1)
    parser.add_argument('--max_socrer_data_length', type=int, default=512)
    parser.add_argument('--model_name', type=str,
                        default="../../pretrain_model/roberta-large")
    parser.add_argument('--hidden_size', type=int, default=1024)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--learning_rate', type=float, default=2e-05)
    parser.add_argument('--save_step', type=int, default=1000000)
    parser.add_argument('--model_save', type=str, default="nerd_train_large_all")
    parser.add_argument("--train_type", default="supervise", type=str, choices=["supervise", "stge"],
                        help="select train type")

    args = parser.parse_args()
    if args.dataset_type == "rams":
        args.train_path = "../../dataset/RAMS_1.0/data/train.jsonlines"
        args.dev_path = "../../dataset/RAMS_1.0/data/dev.jsonlines"
        args.test_path = "../../dataset/RAMS_1.0/data/test.jsonlines"
        args.convert_path = "../../dataset/RAMS_1.0/data/convert/{}/".format(args.task_type)
        args.scorer_data_path = "../../dataset/RAMS_1.0/scorer_data/{}/".format(args.task_type)
        args.invalid_arg_num = 1
    elif args.dataset_type == "wiki":
        args.train_path = "../../dataset/WIKIEVENT/train.jsonl"
        args.dev_path = "../../dataset/WIKIEVENT/dev.jsonl"
        args.test_path = "../../dataset/WIKIEVENT/test.jsonl"
        args.convert_path = "../../dataset/WIKIEVENT/convert/{}/".format(args.task_type)
        args.scorer_data_path = "../../dataset/WIKIEVENT/scorer_data/{}/".format(args.task_type)
        args.invalid_arg_num = 0
        args.head_only = True
    elif args.dataset_type == "ace":
        args.train_path = "../../dataset/ACE2005/train_convert.json"
        args.dev_path = "../../dataset/ACE2005/dev_convert.json"
        args.test_path = "../../dataset/ACE2005/test_convert.json"
        args.convert_path = "../../dataset/ACE2005/convert/{}/".format(args.task_type)
        args.scorer_data_path = "../../dataset/ACE2005/scorer_data/{}/".format(args.task_type)
        args.invalid_arg_num = 0
    elif args.dataset_type == "nerd":
        args.convert_path = "../../dataset/Few-NERD/convert/{}/".format(args.task_type)
        args.scorer_data_path = "../../dataset/Few-NERD/scorer_data/{}/".format(args.task_type)
    args.template_path = args.convert_path + "template.json"
    args.shot_event_map_path = args.convert_path + "shot_event_map.json"
    args.current_time = get_current_time()
    args.save_result_path = "output/{}/{}_{}_{}_{}_{}_shot_result_{}.json".format(
        args.dataset_type, args.llm_name,
        args.method_type, args.task_type, args.logit_strategy,
        args.shot_num,
        args.current_time)

    args.shot_data_path = args.convert_path + "{}-shot_data.json".format(str(args.shot_num))

    if args.dataset_type == "ace" and args.task_type == "ner":
        args.value_num = 0


    args.scorer_test_data_path = args.scorer_data_path + "/test.tsv"
    if args.dataset_type == "nerd":
        if not os.path.exists("model/{}/".format(args.dataset_type)):
            os.makedirs("model/{}/".format(args.dataset_type))
        args.model_save_path = "model/{}/{}".format(args.dataset_type, args.model_save)
        args.scorer_test_data_path = None
    else:
        if "/" not in args.model_save:
            if not os.path.exists("model/{}/{}/".format(args.dataset_type, args.task_type)):
                os.makedirs("model/{}/{}/".format(args.dataset_type, args.task_type))
            args.model_save_path = "model/{}/{}/{}".format(args.dataset_type, args.task_type, args.model_save)
        else:
            args.model_save_path = args.model_save


    if args.scorer_train_data_path == "":
        args.scorer_train_data_path = args.scorer_data_path + "{}-shot_train_per.tsv".format(args.shot_num)

    if not os.path.exists("output/{}".format(args.dataset_type)):
        os.makedirs("output/{}".format(args.dataset_type))

    if not os.path.exists(args.scorer_data_path):
        os.makedirs(args.scorer_data_path)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(args)
    return args
