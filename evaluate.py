import argparse
# import spacy
import ast
from utils import *
import traceback


def find_arg_span(word_list, trigger_start, argument_text):
    min_value, min_start, min_end = 99999, -1, -1
    arg_token_list = argument_text.split(" ")
    for idx in range(len(word_list)):
        if " ".join(word_list[idx:idx+len(arg_token_list)]) == argument_text:
            start_pos, end_pos = idx, idx+len(arg_token_list)
            if abs(start_pos - trigger_start) < min_value:
                min_value = abs(start_pos - trigger_start)
                min_start, min_end = start_pos, end_pos
    return min_start, min_end


def get_rule_output(output, format_symbols=["{", "}"]):
    if len(output) > 0 and (output[0] != format_symbols[0] or output[-1] != format_symbols[1]
                                or output.count(format_symbols[0]) != 1 or output.count(format_symbols[1]) != 1):
        start_index = output.find(format_symbols[0])
        end_index = output.find(format_symbols[1])
        if start_index != -1 and end_index != -1:
            return output[start_index:end_index+1]
    return output


def ner_evaluate(args):
    template = read_json(args.template_path)
    output_data = read_json(args.save_result_path)
    json_error_num, key_error_num, value_error_num = 0, 0, 0
    pred_arg_num, gold_arg_num, arg_class_num = 0, args.invalid_arg_num, 0
    pred_json, pred_key, pred_value = 0, 0, 0
    for data in output_data:
        gold_output = data["gold_dict"]
        if "pred_output" in data:
            response = data["pred_output"]
        elif "output" in data:
            response = data["output"]

        if args.rule:
            response = get_rule_output(response)
        if "inputs" in data:
            inputs = data["inputs"]
        else:
            inputs = data["original_input"]
        type_list = template["entity_type"]
        try:
            pred_arg_dict = ast.literal_eval(response)
            for key in pred_arg_dict.keys():
                if key not in type_list:
                    key_error_num += 1
            for role in pred_arg_dict.keys():
                ent_list = pred_arg_dict[role]
                for ent in ent_list:
                    if ent not in inputs:
                        value_error_num += 1
        except Exception as e:
            print("error json", response)
            traceback.print_exc()
            json_error_num += 1
            pred_arg_dict = {}
        pred_json += 1
        pred_key += len(list(pred_arg_dict.keys()))
        for role in pred_arg_dict.keys():
            pred_value += len(pred_arg_dict[role])

        gold_arg_dict = {}
        for ent_type in gold_output.keys():
            for ent in gold_output[ent_type]:
                if ent_type not in gold_arg_dict:
                    gold_arg_dict[ent_type] = []
                gold_arg_dict[ent_type].append([ent['start'], ent['end']])
        for ent_type in gold_arg_dict.keys():
            gold_arg_list = gold_arg_dict[ent_type]
            gold_arg_num += len(gold_arg_list)

        for ent_type in type_list:
            if ent_type in pred_arg_dict:
                if isinstance(pred_arg_dict[ent_type], str):
                    pred_arg_list = [pred_arg_dict[ent_type]]
                else:
                    try:
                        pred_arg_list = list(set(pred_arg_dict[ent_type]))
                    except:
                        pred_arg_list = []
                for pred in pred_arg_list:
                    if len(pred) > 0:
                        arg_start, arg_end = find_arg_span(data["token_list"], 0, str(pred))
                        if (args.filter is True and arg_start != -1 and arg_end != -1) or args.filter is False:
                            arg = [arg_start, arg_end]
                            pred_arg_num += 1
                            if ent_type in gold_arg_dict and arg in gold_arg_dict[ent_type]:
                                arg_class_num += 1


    role_prec, role_rec, role_f = compute_f1(
        pred_arg_num, gold_arg_num, arg_class_num)

    print("json_error_num: {:.2f}({}/{}), key_error_num: {:.2f}({}/{}), value_error_num: {:.2f}({}/{})"
          .format(json_error_num/pred_json*100.0, json_error_num, pred_json,
                 key_error_num / pred_key * 100.0, key_error_num, pred_key,
                  value_error_num / pred_value * 100.0, value_error_num, pred_value))
    print('Entity classification: pred_num: {}, gold_num: {}, correct_num: {}'.format(
        pred_arg_num, gold_arg_num, arg_class_num))
    print('Entity classification: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
        role_prec * 100.0, role_rec * 100.0, role_f * 100.0))
    return role_f


def re_evaluate(args):
    template = read_json(args.template_path)
    output_data = read_json(args.save_result_path)
    json_error_num, key_error_num, value_error_num = 0, 0, 0
    pred_arg_num, gold_arg_num, arg_class_num = 0, args.invalid_arg_num, 0
    pred_json, pred_key, pred_value = 0, 0, 0
    for data in output_data:
        gold_output = data["gold_dict"]
        if "pred_output" in data:
            response = data["pred_output"]
        elif "output" in data:
            response = data["output"]
        if args.rule:
            response = get_rule_output(response)
        if "inputs" in data:
            inputs = data["inputs"]
        else:
            inputs = data["original_input"]
        type_list = template["relation_type"]
        try:
            pred_arg_dict = ast.literal_eval(response)
            for key in pred_arg_dict.keys():
                if key not in type_list:
                    key_error_num += 1
            for re_type in pred_arg_dict.keys():
                re_list = pred_arg_dict[re_type]
                for re_span in re_list:
                    if re_span[0] not in inputs:
                        value_error_num += 1
                    if re_span[1] not in inputs:
                        value_error_num += 1
        except Exception as e:
            print("error json", response)
            traceback.print_exc()
            json_error_num += 1
            pred_arg_dict = {}
        pred_json += 1
        pred_key += len(list(pred_arg_dict.keys()))
        for role in pred_arg_dict.keys():
            pred_value += len(pred_arg_dict[role])

        gold_arg_dict = {}
        for re_type in gold_output.keys():
            for re_span in gold_output[re_type]:
                if re_type not in gold_arg_dict:
                    gold_arg_dict[re_type] = []
                gold_arg_dict[re_type].append([re_span['entity1_start'], re_span['entity1_end'],
                                                       re_span["entity2_start"], re_span["entity2_end"]])
        for re_type in gold_arg_dict.keys():
            gold_arg_list = gold_arg_dict[re_type]
            gold_arg_num += len(gold_arg_list)

        for re_type in type_list:
            if re_type in pred_arg_dict:
                if isinstance(pred_arg_dict[re_type], str):
                    pred_arg_list = [pred_arg_dict[re_type]]
                else:
                    try:
                        pred_arg_list = list(set([tuple(t) for t in pred_arg_dict[re_type]]))
                    except:
                        pred_arg_list = []
                for pred in pred_arg_list:
                    if len(pred) == 2:
                        entity1_start, entity1_end = find_arg_span(data["token_list"], 0, str(pred[0]))
                        entity2_start, entity2_end = find_arg_span(data["token_list"], 0, str(pred[1]))
                        if (args.filter is True and entity1_start != -1 and entity1_end != -1
                            and entity2_start != -1 and entity2_end != -1) or args.filter is False:
                            pred_arg_num += 1
                            if re_type in gold_arg_dict and ([entity1_start, entity1_end, entity2_start, entity2_end]
                                                             in gold_arg_dict[re_type] or
                                                             [entity2_start, entity2_end, entity1_start, entity1_end]
                                                             in gold_arg_dict[re_type]):
                                arg_class_num += 1
                    else:
                        pred_arg_num += 1

    role_prec, role_rec, role_f = compute_f1(
        pred_arg_num, gold_arg_num, arg_class_num)
    print("json_error_num: {:.2f}({}/{}), key_error_num: {:.2f}({}/{}), value_error_num: {:.2f}({}/{})"
          .format(json_error_num / pred_json * 100.0, json_error_num, pred_json,
                  key_error_num / pred_key * 100.0, key_error_num, pred_key,
                  value_error_num / pred_value * 100.0, value_error_num, pred_value))
    print('Relation classification: pred_num: {}, gold_num: {}, correct_num: {}'.format(
        pred_arg_num, gold_arg_num, arg_class_num))
    print('Relation classification: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
        role_prec * 100.0, role_rec * 100.0, role_f * 100.0))
    return role_f


def eae_evaluate(args):
    event2argument = read_json(args.template_path)
    output_data = read_json(args.save_result_path)
    json_error_num, key_error_num, value_error_num = 0, 0, 0
    pred_arg_num, gold_arg_num, arg_class_num = 0, args.invalid_arg_num, 0
    pred_idn_num, gold_idn_num, arg_idn_num = 0, args.invalid_arg_num, 0
    pred_json, pred_key, pred_value = 0, 0, 0
    for data in output_data:
        gold_event = data["gold_dict"]
        if "pred_output" in data:
            response = data["pred_output"]
        elif "output" in data:
            response = data["output"]
        if args.rule:
            response = get_rule_output(response)
        event_type = gold_event["event_type"]
        if "inputs" in data:
            inputs = data["inputs"]
        else:
            inputs = data["original_input"]
        gold_arg = gold_event["argument"]
        role_list = event2argument[event_type]["role_list"]
        try:
            pred_arg_dict = ast.literal_eval(response)
            for key in pred_arg_dict.keys():
                if key not in role_list:
                    key_error_num += 1
            for role in pred_arg_dict.keys():
                argument_list = pred_arg_dict[role]
                for argument in argument_list:
                    if argument not in inputs:
                        value_error_num += 1
        except Exception as e:
            print("error json", response)
            traceback.print_exc()
            json_error_num += 1
            pred_arg_dict = {}

        pred_json += 1
        pred_key += len(list(pred_arg_dict.keys()))
        for role in pred_arg_dict.keys():
            pred_value += len(pred_arg_dict[role])
        gold_arg_dict = {}
        for arg in gold_arg:
            role = arg["role"]
            if role not in gold_arg_dict:
                gold_arg_dict[role] = []
            gold_arg_dict[role].append([arg['start'], arg['end']])

        all_gold_list, all_pred_list = [], []
        for role in gold_arg_dict.keys():
            gold_arg_list = gold_arg_dict[role]
            gold_arg_num += len(gold_arg_list)
            for gold_arg in gold_arg_list:
                if gold_arg not in all_gold_list:
                    all_gold_list.append(gold_arg)

        for role in role_list:
            if role in pred_arg_dict:
                if isinstance(pred_arg_dict[role], str):
                    pred_arg_list = [pred_arg_dict[role]]
                else:
                    try:
                        pred_arg_list = list(set(pred_arg_dict[role]))
                    except:
                        pred_arg_list = []
                for pred in pred_arg_list:
                    if len(pred) > 0:
                        arg_start, arg_end = find_arg_span(data["token_list"], gold_event["trigger"]["start"], str(pred))
                        if (args.filter is True and arg_start != -1 and arg_end != -1) or args.filter is False:
                            arg = [arg_start, arg_end]
                            if arg not in all_pred_list:
                                all_pred_list.append(arg)
                            pred_arg_num += 1
                            if role in gold_arg_dict and arg in gold_arg_dict[role]:
                                arg_class_num += 1

        gold_idn_num += len(all_gold_list)
        pred_idn_num += len(all_pred_list)
        for pred in all_pred_list:
            if pred in all_gold_list:
                arg_idn_num += 1

    role_id_prec, role_id_rec, role_id_f = compute_f1(
        pred_idn_num, gold_idn_num, arg_idn_num)
    role_prec, role_rec, role_f = compute_f1(
        pred_arg_num, gold_arg_num, arg_class_num)

    print("json_error_num: {:.2f}({}/{}), key_error_num: {:.2f}({}/{}), value_error_num: {:.2f}({}/{})"
          .format(json_error_num/pred_json*100.0, json_error_num, pred_json,
                 key_error_num / pred_key * 100.0, key_error_num, pred_key,
                  value_error_num / pred_value * 100.0, value_error_num, pred_value))
    print('Role identification: pred_num: {}, gold_num: {}, correct_num: {}'.format(
        pred_idn_num, gold_idn_num, arg_idn_num))
    print('Role classification: pred_num: {}, gold_num: {}, correct_num: {}'.format(
        pred_arg_num, gold_arg_num, arg_class_num))
    print('Role identification: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
        role_id_prec * 100.0, role_id_rec * 100.0, role_id_f * 100.0))
    print('Role classification: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
        role_prec * 100.0, role_rec * 100.0, role_f * 100.0))
    return role_f


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_result_path', type=str)
    parser.add_argument('--head-only', action='store_true')
    parser.add_argument('--filter', action='store_true')
    parser.add_argument('--dataset_type', type=str, default='rams', choices=['rams', 'wiki', 'ace'])
    parser.add_argument("--task_type", default="eae", type=str, choices=['ner', 're', 'eae'],
                        help="task name.")
    parser.add_argument('--rule', action='store_true')
    args = parser.parse_args()
    print("args", args)
    if args.dataset_type == "rams":
        args.template_path = "../../dataset/RAMS_1.0/data/convert/{}/template.json".format(args.task_type)
        args.invalid_arg_num = 1
    elif args.dataset_type == "wiki":
        args.template_path = "../../dataset/WIKIEVENT/convert/{}/template.json".format(args.task_type)
        args.invalid_arg_num = 0
    elif args.dataset_type == "ace":
        args.template_path = "../../dataset/ACE2005/convert/{}/template.json".format(args.task_type)
        args.invalid_arg_num = 0

    if args.task_type == "eae":
        eae_evaluate(args)
    elif args.task_type == "ner":
        ner_evaluate(args)
    elif args.task_type == "re":
        re_evaluate(args)
    else:
        pass
