from utils import *
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import ast
import itertools
import os
from config import get_args_parser


def save_data(data, save_path):
    content = []
    label = []
    for da in data:
        content.append(da[0])
        label.append(da[1])
    save_df = pd.DataFrame({'content': content, 'label': label})
    save_df.to_csv(save_path, index=None, sep='\t')
    print("file {} saved, lens {}".format(save_path, len(save_df)))


def get_subset(orginal_list, max_num=None):
    return_list = [[]]
    n = len(orginal_list)
    for num in range(1, n+1):
        for sublist in itertools.combinations(orginal_list, num):
            return_list.append(list(sublist))
            if n > 6 and len(return_list) > 128:
                break
        if n > 6 and len(return_list) > 128:
            break
    if max_num is not None and len(return_list) > max_num:
        return_list = random.sample(return_list, max_num)
    return return_list


def judge_dict(pred_dict, gold_dict):
    pred_num, gold_num, cor_num = 0, 0, 0
    for key in pred_dict.keys():
        pred_num += len(pred_dict[key])
        if key in gold_dict:
            for value in pred_dict[key]:
                if value in gold_dict[key]:
                    cor_num += 1
    for key in gold_dict.keys():
        gold_num += len(gold_dict[key])
    p, r, f1 = compute_f1(pred_num, gold_num, cor_num)
    return p


def get_eae_str(context, event_type, trigger_word, current_dict, arg_role=None):
    arg_info = []
    for label in current_dict:
        if len(current_dict[label]) != 0:
            arg_info.append("the argument with the role of {} is {}".format(label, " and ".join(current_dict[label])))
    if len(arg_info) == 0:
        arg_str = ", "
    else:
        arg_str = ", {}, ".format(", ".join(arg_info))
    if arg_role is None:
        return_str = "{} In this {} event with {} as the trigger word{}" \
                     "are there other arguments?".format(context, event_type, trigger_word, arg_str)
    else:
        return_str = "{} In this {} event with {} as the trigger word{}" \
                     "are there other arguments that play the role of {}?".format(context, event_type, trigger_word,
                                                                                  arg_str, arg_role)

    return return_str


def get_ner_str(context, current_dict, entity_type=None):
    arg_info = []
    for label in current_dict:
        if len(current_dict[label]) != 0:
            arg_info.append("the entity tagged as {} is {}".format(label, " and ".join(current_dict[label])))
    if len(arg_info) == 0:
        arg_str = ", "
    else:
        arg_str = ", {}, ".format(", ".join(arg_info))
    if entity_type is None:
        return_str = "{} In this context{}are there other entities?".format(context, arg_str)
    else:
        return_str = "{} In this context{}are there other {} entities?".format(context, arg_str, entity_type)
    return return_str


def get_re_str(context, current_dict, re_type=None):
    arg_info = []
    for label in current_dict:
        ent_list = []
        for ent_pair in current_dict[label]:
            ent_list.append("{} and {}".format(ent_pair[0], ent_pair[1]))
        re_str = ", ".join(ent_list)
        if len(current_dict[label]) != 0:
            arg_info.append("the relation between {} is {}".format(re_str, label))
    if len(arg_info) == 0:
        arg_str = ", "
    else:
        arg_str = ", {}, ".format(", ".join(arg_info))
    if re_type is None:
        return_str = "{} In this context{}are there other entity relations?".format(context, arg_str)
    else:
        return_str = "{} In this context{}are there other {} entity relations?".format(context, arg_str, re_type)
    return return_str


def construct_gold_data(args, content, output_dict, ent_type_list, max_num=None, event_type=None, trigger_word=None, sample=False):
    return_data, all_pos_data, all_neg_data = [], [], []
    pos_num, neg_num = 0, 0
    if len(list(output_dict.keys())) == 0:
        if args.task_type == "ner":
            return_data.append([get_ner_str(content, {}), 0])
            all_neg_data.append([get_ner_str(content, {}), 0])
        elif args.task_type == "eae":
            return_data.append([get_eae_str(content, event_type, trigger_word, {}), 0])
            all_neg_data.append([get_eae_str(content, event_type, trigger_word, {}), 0])
        elif args.task_type == "re":
            return_data.append([get_re_str(content, {}), 0])
            all_neg_data.append([get_re_str(content, {}), 0])
        neg_num += 1
    else:
        pos_data, neg_data = [], []
        exist_key_list = list(output_dict.keys())
        if args.task_type == "ner":
            pos_data.append([get_ner_str(content, {}), 1])
        elif args.task_type == "eae":
            pos_data.append([get_eae_str(content, event_type, trigger_word, {}), 1])
        elif args.task_type == "re":
            pos_data.append([get_re_str(content, {}), 1])
        all_key_sublist = get_subset(exist_key_list, max_num)
        for sublist in all_key_sublist:
            for idx in range(len(sublist)):
                all_value_sublist = get_subset(output_dict[sublist[idx]], max_num)
                for value_sublist in all_value_sublist:
                    current_dict = {}
                    for key in sublist:
                        if key != sublist[idx]:
                            current_dict[key] = output_dict[key]
                        else:
                            current_dict[key] = list(value_sublist)
                    if args.task_type == "ner":
                        data_str = get_ner_str(content, current_dict)
                    elif args.task_type == "eae":
                        data_str = get_eae_str(content, event_type, trigger_word, current_dict)
                    elif args.task_type == "re":
                        data_str = get_re_str(content, current_dict)
                    if current_dict == output_dict:
                        data = [data_str, 0]
                        if data not in neg_data:
                            neg_data.append(data)
                    else:
                        data = [data_str, 1]
                        if data not in pos_data:
                            pos_data.append(data)
                    for ent_type in ent_type_list:
                        if args.task_type == "ner":
                            data_str = get_ner_str(content, current_dict, ent_type)
                        elif args.task_type == "eae":
                            data_str = get_eae_str(content, event_type, trigger_word, current_dict, ent_type)
                        elif args.task_type == "re":
                            data_str = get_re_str(content, current_dict, ent_type)
                        if ent_type not in output_dict.keys():
                            data = [data_str, 0]
                            if data not in neg_data:
                                neg_data.append(data)
                        else:
                            if ent_type not in current_dict.keys():
                                data = [data_str, 1]
                                if data not in pos_data:
                                    pos_data.append(data)
                            elif ent_type in current_dict.keys() and current_dict[ent_type] == output_dict[ent_type]:
                                data = [data_str, 0]
                                if data not in neg_data:
                                    neg_data.append(data)
                            else:
                                data = [data_str, 1]
                                if data not in pos_data:
                                    pos_data.append(data)

        if sample:
            all_pos_data.extend(pos_data)
            all_neg_data.extend(neg_data)
        else:
            if max_num is not None:
                sample_num = min(max_num, min(len(pos_data), len(neg_data)))
                pos_data = random.sample(pos_data, sample_num)
                neg_data = random.sample(neg_data, sample_num)
            else:
                sample_num = min(len(pos_data), len(neg_data))
                pos_data = random.sample(pos_data, sample_num)
                neg_data = random.sample(neg_data, sample_num)
            return_data.extend(pos_data)
            return_data.extend(neg_data)
    if sample:
        return all_pos_data, all_neg_data
    else:
        return return_data


def construct_pred_data(args, content, output_dict, pred_dict, ent_type_list, max_num=None, event_type=None, trigger_word=None):
    return_data = []
    pos_data, neg_data = construct_gold_data(args, content, output_dict, ent_type_list, max_num=args.max_num,
                                              event_type=event_type, trigger_word=trigger_word, sample=True)
    if len(pred_dict) != 0 and pred_dict != output_dict:
        exist_key_list = list(pred_dict.keys())
        all_key_sublist = get_subset(exist_key_list, max_num)
        for sublist in all_key_sublist:
            for idx in range(len(sublist)):
                all_value_sublist = get_subset(pred_dict[sublist[idx]], max_num)
                for value_sublist in all_value_sublist:
                    current_dict = {}
                    for key in sublist:
                        if key != sublist[idx]:
                            current_dict[key] = pred_dict[key]
                        else:
                            current_dict[key] = list(value_sublist)
                    value_num = 0
                    for key in current_dict.keys():
                        value_num += len(current_dict[key])
                    score = judge_dict(current_dict, output_dict)
                    if value_num == 0: 
                        continue
                    if score >= args.threshold:
                        current_label = 1
                    else:
                        current_label = 0
                    if args.task_type == "ner":
                        data_str = get_ner_str(content, current_dict)
                    elif args.task_type == "eae":
                        data_str = get_eae_str(content, event_type, trigger_word, current_dict)
                    elif args.task_type == "re":
                        data_str = get_re_str(content, current_dict)
                    if current_dict == output_dict:
                        data = [data_str, current_label]
                        if current_label == 0 and data not in neg_data:
                            neg_data.append(data)
                        elif current_label == 1 and data not in pos_data:
                            pos_data.append(data)
                    for ent_type in ent_type_list:
                        if args.task_type == "ner":
                            data_str = get_ner_str(content, current_dict, ent_type)
                        elif args.task_type == "eae":
                            data_str = get_eae_str(content, event_type, trigger_word, current_dict, ent_type)
                        elif args.task_type == "re":
                            data_str = get_re_str(content, current_dict, ent_type)
                        if ent_type not in output_dict.keys():
                            data = [data_str, 0]
                            if data not in neg_data:
                                neg_data.append(data)
                        else:
                            if current_label == 1:
                                if ent_type not in current_dict.keys():
                                    data = [data_str, 1]
                                    if data not in pos_data:
                                        pos_data.append(data)
                                elif ent_type in current_dict.keys() and current_dict[ent_type] == output_dict[ent_type]:
                                    data = [data_str, 0]
                                    if data not in neg_data:
                                        neg_data.append(data)
                                else:
                                    data = [data_str, 1]
                                    if data not in pos_data:
                                        pos_data.append(data)
                            else:
                                data = [data_str, 0]
                                if data not in neg_data:
                                    neg_data.append(data)
    if max_num is not None:
        sample_num = min(max_num, min(len(pos_data), len(neg_data)))
        pos_data = random.sample(pos_data, sample_num)
        neg_data = random.sample(neg_data, sample_num)
    else:
        sample_num = min(len(pos_data), len(neg_data))
        pos_data = random.sample(pos_data, sample_num)
        neg_data = random.sample(neg_data, sample_num)
    return_data.extend(pos_data)
    return_data.extend(neg_data)
    return return_data


def get_data(args, all_data):
    template = read_json(args.template_path)
    pred_pos_num, pred_neg_num = 0, 0
    return_all_data = []
    for data in all_data:
        content = data["inputs"]
        output = data["outputs"]
        output_dict = ast.literal_eval(output)
        if args.task_type == "ner":
            type_list = template["entity_type"]
            return_data = construct_gold_data(args, content, output_dict, type_list, max_num=args.max_num)
        elif args.task_type == "eae":
            trigger_word = data["gold_dict"]["trigger"]["text"]
            event_type = data["gold_dict"]["event_type"]
            type_list = template[event_type]["role_list"]
            return_data = construct_gold_data(args, content, output_dict, type_list, max_num=args.max_num,
                                              event_type=event_type, trigger_word=trigger_word)
        elif args.task_type == "re":
            type_list = template["relation_type"]
            return_data = construct_gold_data(args, content, output_dict, type_list, max_num=args.max_num)
        else:
            print("invalid task type")
        return_all_data.extend(return_data)
    pos_data, neg_data = [], []
    for data in return_all_data:
        if data[1] == 1:
            pos_data.append(data)
        else:
            neg_data.append(data)
    random.shuffle(return_all_data)
    return return_all_data


def get_pred_data(args, all_data):
    template = read_json(args.template_path)
    pred_pos_num, pred_neg_num = 0, 0
    return_all_data = []
    for data in all_data:
        content = data["inputs"]
        output = data["outputs"]
        output_dict = ast.literal_eval(output)
        try:
            pred_output_dict = ast.literal_eval(data["pred_output"])
            pred_dict = {}
            for role in pred_output_dict.keys():
                if args.task_type == "ner" or args.task_type == "eae":
                    values = list(set(pred_output_dict[role]))
                else:
                    values = []
                    for current_value in pred_output_dict[role]:
                        if current_value not in values:
                            values.append(current_value)
                if len(values) != 0:
                    pred_dict[role] = values
        except Exception as e:
            print("e", e)
            pred_dict = {}
        return_pred_data = []
        if args.task_type == "ner":
            type_list = template["entity_type"]
            return_pred_data = construct_pred_data(args, content, output_dict, pred_dict, type_list,
                                                   max_num=args.pred_max_num)
        elif args.task_type == "eae":
            trigger_word = data["gold_dict"]["trigger"]["text"]
            event_type = data["gold_dict"]["event_type"]
            type_list = template[event_type]["role_list"]
            return_pred_data = construct_pred_data(args, content, output_dict, pred_dict, type_list,
                                                   max_num=args.pred_max_num, event_type=event_type,
                                                   trigger_word=trigger_word)
        elif args.task_type == "re":
            type_list = template["relation_type"]
            return_pred_data = construct_pred_data(args, content, output_dict, pred_dict, type_list,
                                                   max_num=args.pred_max_num)
        else:
            print("invalid task type")
        return_all_data.extend(return_pred_data)

    for data in return_all_data:
        if data[1] == 1:
            pred_pos_num += 1
        else:
            pred_neg_num += 1
    random.shuffle(return_all_data)
    return return_all_data

def construct_dataset(args, data_path, train_save_path, dev_save_path):
    all_data = read_json(data_path)
    train_data = get_data(args, all_data)
    save_data(train_data, train_save_path)


def get_dataloaber(args, data_path, tokenizer):
    dataset = ScorerDataset(tokenizer, data_path, device=args.device, max_length=args.max_socrer_data_length,
                                  train=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False,
                            collate_fn=dataset.collate_fn)
    return dataloader


class ScorerDataset(Dataset):
    def __init__(self, tokenizer, path, device, max_length=512, train=True):
        self.tokenizer = tokenizer
        self.path = path
        self.train = train
        self.data = []
        self.device = device
        self.max_length = max_length
        self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def load_data(self):
        self.data = np.array(pd.read_csv(self.path, sep='\t')).tolist()

    def collate_fn(self, batch):
        input_ids = []
        input_attens = []
        for x in batch:
            inputs = self.tokenizer(x[0], padding="max_length", truncation=True, max_length=self.max_length)
            input_ids.append(inputs["input_ids"])
            input_attens.append(inputs["attention_mask"])
        input_ids = torch.LongTensor(input_ids).to(self.device)
        input_attens = torch.LongTensor(input_attens).to(self.device)
        if self.train:
            label = [x[-1] for x in batch]
            label = torch.LongTensor(label).to(self.device)
            return input_ids, input_attens, label
        else:
            return input_ids, input_attens


if __name__ == '__main__':
    args = get_args_parser()
    print(args)
    if not os.path.exists(args.scorer_data_path):
        os.makedirs(args.scorer_data_path)
    if args.dataset_type == "nerd":
        template = read_json(args.template_path)
        ent_type_list = template["entity_type"]
        for data_name in ["train", "dev"]:
            all_data = read_json(args.convert_path+"few-nerd_{}_all.json".format(data_name))
            return_all_data = []
            for data in all_data:
                content = " ".join(data["token_list"])
                output = data["ent_info"]
                return_data = construct_gold_data(args, content, output, ent_type_list, max_num=args.max_num)
                return_all_data.extend(return_data)
            random.shuffle(return_all_data)
            pos_data, neg_data = [], []
            for data in return_all_data:
                if data[1] == 1:
                    pos_data.append(data)
                else:
                    neg_data.append(data)
            if data_name == "train":
                save_data(return_all_data, args.scorer_data_path+args.train_save_path)
            else:
                save_data(return_all_data, args.scorer_data_path + args.dev_save_path)
    else:
        if args.shot_num == 0:
            data_path = args.convert_path + "/test.json"
            all_data = read_json(data_path)
            all_dataset = get_data(args, all_data)
            save_data(all_dataset, args.scorer_data_path+"test.tsv")
        else:
            construct_dataset(args, args.convert_path + "{}-shot_data.json".format(args.shot_num),
                              args.scorer_data_path+args.train_save_path, None)

