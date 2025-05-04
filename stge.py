import torch
from stge_logits import *
from utils import *
from evaluate import eae_evaluate, ner_evaluate, re_evaluate
from prompt_construct import get_eae_prompt, get_response, get_ner_prompt, get_re_prompt
from config import get_args_parser
import time


class BaseModel:
    def __init__(self, args):
        self.args = args
        self.model_class = construct_model_class(self.args)

    def inference(self, original_input, event_info, input_content, entity_list=None):
        eos_token_id = truncation_symbol[self.args.llm_name]

        all_response = self.model_class.inference(original_input, event_info,
                                                  input_content, eos_token_id, entity_list)
        response = get_response(all_response, split_str=answer_symbol[self.args.llm_name], format_symbols=["{", "}"])
        return response


def select_shot_data(all_shot_data, current_emb, topk=1):
    demo_data = []
    cos_sim_list = []
    for data in all_shot_data:
        emb = data["emb"]
        cos_sim = computer_similarity(current_emb, emb)
        cos_sim_list.append(cos_sim)
    topk_idx_list = get_topk_idx(cos_sim_list, topk)
    for idx in topk_idx_list:
        demo_data.append(all_shot_data[idx])
    return demo_data

def sort_label(all_shot_data):
    return_data = []
    for data in all_shot_data:
        print("data", data)
        sort_data = sorted(data.items(), key=lambda item:len(item[1]), reverse=True)
        sort_dict = {}
        for key, value in sort_data:
            sort_dict[key] = value
        print("sort_dict", sort_dict)
        return_data.append(sort_dict)
    return return_data


def extract(args,data_path):
    test_data = read_json(data_path)
    print("data_num", len(test_data))
    template = read_json(args.template_path)
    if args.demo_num != 0:
        embedding_model = EmbeddingModel()
        all_shot_data = read_json(args.shot_data_path)
        for data in all_shot_data:
            data["emb"] = embedding_model.get_embedding(data["inputs"])

    model = BaseModel(args)
    all_result = []
    begin_time = time.time()
    for idx, data in enumerate(test_data):
        print("-" * 30)
        if args.demo_num != 0:
            current_emb = embedding_model.get_embedding(data["inputs"])
            current_data = []
            for shot in all_shot_data:
                if shot["inputs"] != data["inputs"]:
                    current_data.append(shot)
            shot_data = select_shot_data(current_data, current_emb, topk=args.demo_num)
        else:
            shot_data = None
        if args.task_type == "eae":
            event_type = data["gold_dict"]["event_type"]
            trigger_word = data["gold_dict"]["trigger"]["text"]
            input_content = get_eae_prompt(data, shot_data, template)
            label_list = template[event_type]["role_list"]
            event_info = {
                "event_type": event_type,
                "trigger_word": trigger_word,
                "label_list": label_list,
                "ground_output": data["outputs"]
            }
            entity_list = None
        elif args.task_type == "ner":
            input_content = get_ner_prompt(data, shot_data, template)
            label_list = template["entity_type"]
            event_info = {
                "label_list": label_list,
                "ground_output": data["outputs"]
            }
            entity_list = None
        elif args.task_type == "re":
            input_content = get_re_prompt(data, shot_data, template)
            label_list = template["relation_type"]
            event_info = {
                "label_list": label_list,
                "ground_output": data["outputs"]
            }
            entity_list = None
        else:
            print("task type error")
        print("input_content", input_content)
        extract_result = model.inference(data["inputs"], event_info, input_content, entity_list)
        print('extract_result\n', filt_space(extract_result))
        print("gold_result\n", data["outputs"])
        result = {
            "pred_output": filt_space(extract_result),
            "token_list": data["token_list"],
            "inputs": data["inputs"],
            "outputs": data["outputs"],
            "gold_dict": data["gold_dict"],
        }
        all_result.append(result)
    end_time = time.time()
    print("{} dataset {}-shot time {} S".format(args.dataset_type, args.shot_num, end_time-begin_time))
    save_json(all_result, args.save_result_path)
    del model, embedding_model
    torch.cuda.empty_cache()



def extract_and_evaluate(args, data_path):
    extract(args, data_path)
    if args.task_type == "eae":
       f1 = eae_evaluate(args)
    elif args.task_type == "ner":
        f1 = ner_evaluate(args)
    elif args.task_type == "re":
        f1 = re_evaluate(args)
    else:
        f1 = None
        print("error task type")
    return f1


if __name__ == '__main__':
    args = get_args_parser()
    extract_and_evaluate(args, args.convert_path + "/test.json")