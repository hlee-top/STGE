import json
from utils import *
import copy
import re
from itertools import chain
from typing import Union
from config import get_args_parser
import os
import shutil
from random import sample

ace_ner_map = {"PER": "Person", "LOC": "Location", "ORG": "Organization",
            "GPE": "Geo-Political Entity", "VEH": "Vehicle", "FAC": "Facility", "WEA": "Weapon"}

ace_re_map = {"ORG-AFF": "Organization-Affiliation", "ART": "Artifact", "PART-WHOLE": "Part-Whole", "PHYS": "Physical",
              "GEN-AFF": "General-Affiliation", "PER-SOC": "Person-Social"}


def build_template(role_list):
    template = []
    for index, role in enumerate(role_list):
        template.append(role)
        template.append(":")
        template.append("<arg{}>".format(index + 1))
        template.append(";")
    return " ".join(template)


def get_rams_event_argument(data_path, event_argument_list):
    with open(data_path, "r", encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            events = line["evt_triggers"]
            for event in events:
                event_type = event[2][0][0]
                if event_type not in event_argument_list:
                    event_argument_list[event_type] = []
                for arg_info in line["gold_evt_links"]:
                    if arg_info[0][0] == event[0] and arg_info[0][1] == event[1]:
                        role = arg_info[2].split("arg", maxsplit=1)[-1][2:]
                        if role not in event_argument_list[event_type]:
                            event_argument_list[event_type].append(role)


def get_wiki_event_argument(data_path, event_argument_list):
    with open(data_path, "r", encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            events = line["event_mentions"]
            for event in events:
                event_type = event['event_type']
                if event_type not in event_argument_list:
                    event_argument_list[event_type] = []
                for arg_info in event['arguments']:
                    role = arg_info['role']
                    if role not in event_argument_list[event_type]:
                        event_argument_list[event_type].append(role)


def get_ace_event_argument(data_path, event_argument_list):
    with open(data_path, "r", encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            events = line["event"]
            for event in events:
                event_type = event[0][1]
                if event_type not in event_argument_list:
                    event_argument_list[event_type] = []
                for arg_info in event[1:]:
                    role = arg_info[2]
                    if role not in event_argument_list[event_type]:
                        event_argument_list[event_type].append(role)


def get_template(dataset_type, train_data_path, dev_data_path, test_data_path, save_path):
    get_template_method = {
        "rams": get_rams_event_argument,
        "wiki": get_wiki_event_argument,
        "ace": get_ace_event_argument,
    }
    event_argument = {}
    get_template_method[dataset_type](train_data_path, event_argument)
    get_template_method[dataset_type](dev_data_path, event_argument)
    get_template_method[dataset_type](test_data_path, event_argument)
    event2argument = {}
    for event_type in event_argument.keys():
        event2argument[event_type] = {
            "role_list": event_argument[event_type],
        }
    print("save template ", save_path)
    print("event2argument", event2argument)
    save_json(event2argument, save_path)


def get_ace_entity(data_path, ent_tmplate):
    with open(data_path, "r", encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            ent_span_list = line["ner"]
            for ent_span in ent_span_list:
                if ent_span[-1] not in ent_tmplate["entity_type"]:
                    ent_tmplate["entity_type"].append(ent_span[-1])


def get_ace_relation(data_path, re_tmplate):
    with open(data_path, "r", encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            re_span_list = line["relation"]
            for re_span in re_span_list:
                relation_type = re_span[-1].split(".")[0]
                if relation_type not in re_tmplate["relation_type"]:
                    re_tmplate["relation_type"].append(relation_type)


def get_ner_template(dataset_type, train_data_path, dev_data_path, test_data_path, save_path):
    get_template_method = {
        "ace": get_ace_entity,
    }
    ent_tmplate = {"entity_type": []}
    get_template_method[dataset_type](train_data_path, ent_tmplate)
    get_template_method[dataset_type](dev_data_path, ent_tmplate)
    get_template_method[dataset_type](test_data_path, ent_tmplate)
    ent_tmplate["entity_type"] = [ace_ner_map[i] if i in ace_ner_map else i for i in ent_tmplate["entity_type"]]
    print("save template ", save_path)
    print("ent_tmplate", ent_tmplate)
    save_json(ent_tmplate, save_path)


def get_re_template(dataset_type, train_data_path, dev_data_path, test_data_path, save_path):
    get_template_method = {
        "ace": get_ace_relation,
    }
    re_tmplate = {"relation_type": []}
    get_template_method[dataset_type](train_data_path, re_tmplate)
    get_template_method[dataset_type](dev_data_path, re_tmplate)
    get_template_method[dataset_type](test_data_path, re_tmplate)
    re_tmplate["relation_type"] = [ace_re_map[i] if i in ace_re_map else i for i in re_tmplate["relation_type"]]
    print("save template ", save_path)
    print("re_tmplate", re_tmplate)
    save_json(re_tmplate, save_path)


def construct_generate_output(role2text):
    output_dict = {}
    for role in role2text.keys():
        output_dict[role] = role2text[role]
    output = json.dumps(output_dict)
    return output



def get_rams_output(data_path, template_path, save_path, window_size):
    event2argument = read_json(template_path)
    W = window_size
    assert (W % 2 == 0)

    all_data = []
    event_num, argument_num, invalid_arg_num = 0, 0, 0
    with open(data_path, "r", encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            doc_key = line["doc_key"]
            events = line["evt_triggers"]
            full_text = copy.deepcopy(list(chain(*line["sentences"])))
            cut_text = list(chain(*line['sentences']))
            sent_length = sum([len(sent) for sent in line['sentences']])
            for event_idx, event in enumerate(events):
                gold_event = dict()
                event_trigger = {
                        "start": event[0],
                        "end": event[1] + 1,
                        "text": " ".join(full_text[event[0]:event[1] + 1]),
                }
                offset, min_s, max_e = 0, 0, W + 1
                event_trigger['offset'] = offset
                if sent_length > W + 1:
                    if event_trigger['end'] <= W // 2:  # trigger word is located at the front of the sents
                        cut_text = full_text[:(W + 1)]
                    else:  # trigger word is located at the latter of the sents
                        offset = sent_length - (W + 1)
                        min_s += offset
                        max_e += offset
                        event_trigger['start'] -= offset
                        event_trigger['end'] -= offset
                        event_trigger['offset'] = offset
                        cut_text = full_text[-(W + 1):]
                gold_event["trigger"] = event_trigger
                gold_event["event_type"] = event[2][0][0]
                event_args = list()
                role2text = {}
                for arg_info in line["gold_evt_links"]:
                    if arg_info[0][0] == event[0] and arg_info[0][1] == event[1]:
                        evt_arg = {
                            "start": arg_info[1][0],
                            "end": arg_info[1][1] + 1,
                            "text": " ".join(full_text[arg_info[1][0]:arg_info[1][1] + 1]),
                            "role": arg_info[2].split("arg", maxsplit=1)[-1][2:]
                        }
                        if evt_arg['start'] < min_s or evt_arg['end'] > max_e:
                            invalid_arg_num += 1
                        else:
                            evt_arg['start'] -= offset
                            evt_arg['end'] -= offset
                            if evt_arg["role"] not in role2text:
                                role2text[evt_arg["role"]] = []
                            role2text[evt_arg["role"]].append(evt_arg['text'])
                            event_args.append(evt_arg)
                            
                gold_event["argument"] = event_args
                output = construct_generate_output(role2text)
                event_num += 1
                argument_num += len(event_args)
                tmp_result = {
                    "event_idx": doc_key + "_{}".format(event_idx),
                    "inputs": " ".join(cut_text),
                    "token_list": cut_text,
                    "gold_dict": gold_event,
                    "outputs": output,
                }
                all_data.append(tmp_result)
    print("event_num:{}, argument_num:{}, invalid_arg_num:{}".format(event_num, argument_num, invalid_arg_num))
    save_json(all_data, save_path)


def get_wiki_output(data_path, template_path, save_path, window_size):
    W = window_size
    assert (W % 2 == 0)
    all_data = []
    event_num, argument_num, invalid_arg_num = 0, 0, 0
    with open(data_path, "r", encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            doc_key = line["doc_id"]
            events = line["event_mentions"]
            entity_dict = {entity['id']: entity for entity in line['entity_mentions']}
            full_text = line['tokens']
            sent_length = len(full_text)
            for event_idx, event in enumerate(events):
                gold_event = dict()
                cut_text = full_text
                event_trigger = event['trigger']
                offset, min_s, max_e = 0, 0, W + 1
                if sent_length > W + 1:
                    if event_trigger['end'] <= W // 2:  # trigger word is located at the front of the sents
                        cut_text = full_text[:(W + 1)]
                    elif event_trigger['start'] >= sent_length - W / 2:  # trigger word is located at the latter of the sents
                        offset = sent_length - (W + 1)
                        min_s += offset
                        max_e += offset
                        event_trigger['start'] -= offset
                        event_trigger['end'] -= offset
                        cut_text = full_text[-(W + 1):]
                    else:
                        offset = event_trigger['start'] - W // 2
                        min_s += offset
                        max_e += offset
                        event_trigger['start'] -= offset
                        event_trigger['end'] -= offset
                        cut_text = full_text[offset:(offset + W + 1)]
                        
                event_trigger['offset'] = offset
                gold_event["trigger"] = event_trigger
                gold_event["event_type"] = event['event_type']
                event_args = list()
                role2text = {}
                for arg_info in event['arguments']:

                    evt_arg = dict()
                    arg_entity = entity_dict[arg_info['entity_id']]
                    evt_arg['start'] = arg_entity['start']
                    evt_arg['end'] = arg_entity['end']
                    evt_arg['text'] = arg_info['text']
                    evt_arg['role'] = arg_info['role']
                    if evt_arg['start']<min_s or evt_arg['end']>max_e:
                        invalid_arg_num += 1
                    else:
                        evt_arg['start'] -= offset
                        evt_arg['end'] -= offset
                        if evt_arg["role"] not in role2text:
                            role2text[evt_arg["role"]] = []
                        role2text[evt_arg["role"]].append(evt_arg['text'])
                        event_args.append(evt_arg)
                        
                gold_event["argument"] = event_args
                output = construct_generate_output(role2text)
                event_num += 1
                argument_num += len(event_args)
                tmp_result = {
                    "event_idx": doc_key + "_{}".format(event_idx),
                    "inputs": " ".join(cut_text),
                    "token_list": cut_text,
                    "gold_dict": gold_event,
                    "outputs": output,
                }
                all_data.append(tmp_result)
    print("event_num:{}, argument_num:{}, invalid_arg_num:{}".format(event_num, argument_num, invalid_arg_num))
    save_json(all_data, save_path)


def get_ace_ner_output(data_path, save_path):
    all_data = []
    entity_num = 0
    with open(data_path, "r", encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            if not line['ner']:
                continue
            ent_span_list = line["ner"]
            offset = line['s_start']
            full_text = line['sentence']
            gold_ent = {}
            type2ent = {}
            for ent_span in ent_span_list:
                entity_num += 1
                ent_type = ace_ner_map[ent_span[-1]]
                ent = {
                    "start": ent_span[0] - offset,
                    "end": ent_span[1] - offset + 1,
                    "text": " ".join(full_text[ent_span[0] - offset:ent_span[1] - offset + 1])
                }
                if ent_type not in gold_ent:
                    gold_ent[ent_type] = [ent]
                else:
                    gold_ent[ent_type].append(ent)
                if ent_type not in type2ent:
                    type2ent[ent_type] = [ent["text"]]
                else:
                    type2ent[ent_type].append(ent["text"])
            output = construct_generate_output(type2ent)
            tmp_result = {
                "event_idx": "{}".format(len(all_data)),
                "inputs": " ".join(full_text),
                "token_list": full_text,
                "gold_dict": gold_ent,
                "outputs": output,
            }
            all_data.append(tmp_result)

    print("entity_num:{}".format(entity_num))
    save_json(all_data, save_path)


def get_ace_re_output(data_path, save_path):
    all_data = []
    relation_num = 0
    with open(data_path, "r", encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            if not line['relation']:
                continue
            re_span_list = line["relation"]
            offset = line['s_start']
            full_text = line['sentence']
            gold_re = {}
            type2re = {}
            entity_list = []
            for re_span in re_span_list:
                relation_num += 1
                re_type = ace_re_map[re_span[-1].split(".")[0]]
                re = {
                    "entity1_start": re_span[0] - offset,
                    "entity1_end": re_span[1] - offset + 1,
                    "entity1_text": " ".join(full_text[re_span[0] - offset:re_span[1] - offset + 1]),
                    "entity2_start": re_span[2] - offset,
                    "entity2_end": re_span[3] - offset + 1,
                    "entity2_text": " ".join(full_text[re_span[2] - offset:re_span[3] - offset + 1]),
                }
                if re["entity1_text"] not in entity_list:
                    entity_list.append(re["entity1_text"])
                if re["entity2_text"] not in entity_list:
                    entity_list.append(re["entity2_text"])
                if re_type not in gold_re:
                    gold_re[re_type] = [re]
                else:
                    gold_re[re_type].append(re)
                if re_type not in type2re:
                    type2re[re_type] = [[re["entity1_text"], re["entity2_text"]]]
                else:
                    type2re[re_type].append([re["entity1_text"], re["entity2_text"]])

            output = construct_generate_output(type2re)
            tmp_result = {
                "event_idx": "{}".format(len(all_data)),
                "inputs": " ".join(full_text),
                "entity_list": entity_list,
                "token_list": full_text,
                "gold_dict": gold_re,
                "outputs": output,
            }
            all_data.append(tmp_result)

    print("relation_num:{}".format(relation_num))
    save_json(all_data, save_path)


def get_ace_output(data_path, template_path, save_path, window_size):
    all_data = []
    event_num, argument_num = 0, 0
    with open(data_path, "r", encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            doc_key = str(event_num)
            events = line["event"]
            offset = line['s_start']
            full_text = line['sentence']
            for event_idx, event in enumerate(events):
                gold_event = dict()
                event_trigger = {
                    "start": event[0][0] - offset,
                    "end":  event[0][0] - offset + 1,
                    "text": " ".join(full_text[event[0][0] - offset:event[0][0] - offset + 1])
                }
                gold_event["trigger"] = event_trigger
                gold_event["event_type"] = event[0][1]
                event_args = list()
                role2text = {}
                for arg_info in event[1:]:
                    evt_arg = {"start": arg_info[0] - offset, "end": arg_info[1]-offset+1,
                               "text": " ".join(full_text[arg_info[0] - offset:(arg_info[1] - offset + 1)]),
                               "role": arg_info[2]}
                    if evt_arg["role"] not in role2text:
                        role2text[evt_arg["role"]] = []
                    role2text[evt_arg["role"]].append(evt_arg['text'])
                    event_args.append(evt_arg)
                gold_event["argument"] = event_args
                output = construct_generate_output(role2text)
                event_num += 1
                argument_num += len(event_args)
                tmp_result = {
                    "event_idx": doc_key + "_{}".format(event_idx),
                    "inputs": " ".join(full_text),
                    "token_list": full_text,
                    "gold_dict": gold_event,
                    "outputs": output,
                }
                all_data.append(tmp_result)
    print("event_num:{}, argument_num:{}".format(event_num, argument_num))
    save_json(all_data, save_path)


def precess_data(args):
    if args.task_type == "eae":
        get_output_method = {
            "rams": get_rams_output,
            "wiki": get_wiki_output,
            "ace": get_ace_output,
        }
        get_template(args.dataset_type, args.train_path, args.dev_path, args.test_path, args.convert_path + "/template.json")
        get_output_method[args.dataset_type](args.train_path, args.convert_path + "/template.json", args.convert_path + "/train.json", args.window_size)
        get_output_method[args.dataset_type](args.dev_path, args.convert_path + "/template.json", args.convert_path + "/dev.json", args.window_size)
        get_output_method[args.dataset_type](args.test_path, args.convert_path + "/template.json", args.convert_path + "/test.json", args.window_size)
    elif args.task_type == "ner":
        get_output_method = {
            "ace": get_ace_ner_output,
        }
        get_ner_template(args.dataset_type, args.train_path, args.dev_path, args.test_path,
                     args.convert_path + "/template.json")
        get_output_method[args.dataset_type](args.train_path,  args.convert_path + "/train.json")
        get_output_method[args.dataset_type](args.dev_path, args.convert_path + "/dev.json")
        get_output_method[args.dataset_type](args.test_path, args.convert_path + "/test.json")
    elif args.task_type == "re":
        get_output_method = {
            "ace": get_ace_re_output,
        }
        get_re_template(args.dataset_type, args.train_path, args.dev_path, args.test_path,
                     args.convert_path + "/template.json")
        get_output_method[args.dataset_type](args.train_path,  args.convert_path + "/train.json")
        get_output_method[args.dataset_type](args.dev_path, args.convert_path + "/dev.json")
        get_output_method[args.dataset_type](args.test_path, args.convert_path + "/test.json")


if __name__ == '__main__':
    args = get_args_parser()
    if os.path.isdir(args.convert_path):
        shutil.rmtree(args.convert_path)
    os.makedirs(args.convert_path)
    precess_data(args)
