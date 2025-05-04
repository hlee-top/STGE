import json
import numpy as np
from numpy.linalg import norm
import random
import time
import torch
import torch.nn.functional as F
from transformers import AutoModel

llm_path = {
    "llama3.1_8b_instruct": "../../pretrain_model/Meta-Llama-3.1-8B-Instruct",
}

truncation_symbol = {
    "llama3.1_8b_instruct": 335,
}

answer_symbol = {
    "llama3.1_8b_instruct": "Answer:",
}

vocab_length = {
    "llama3.1_8b_instruct": 128256,
}




def filt_space(outputs):
    return outputs.replace(' " ', '"').replace("] ", "]").replace(",", ", ").replace(' "', '"').replace('" ', '"')


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.loads(f.read())


def save_json(data, save_path):
    with open(save_path, "w") as f:
        f.write(json.dumps(data, sort_keys=False, indent=4))
    print("save file: {}".format(save_path))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def get_current_time():
    return time.strftime('%m%d_%H%M', time.localtime())


def get_all_role(event2argument):
    all_role_list = []
    for event_type in event2argument.keys():
        role_list = event2argument[event_type]["role_list"]
        for role in role_list:
            if role not in all_role_list:
                all_role_list.append(role)
    return all_role_list


def computer_similarity(vec1, vec2):
    cos_sim = (vec1 @ vec2.T) / (norm(vec1)*norm(vec2))
    return cos_sim


class EmbeddingModel:
    def __init__(self):
        self.model = AutoModel.from_pretrained('../../pretrain_model/jina-embeddings-v2-base-en/',
                                               trust_remote_code=True, local_files_only=True).to("cuda")

    def get_embedding(self, sentence):
        embedding = self.model.encode(sentence)
        return embedding


def compute_metrics(pred):
    print("pred", pred)
    return {
        "acc": 0,
    }


def flat_list(input_list):
    input_list = iter(input_list)
    convert_list = []
    while 1:
        try:
            convert_list += str(input_list.__next__())
        except StopIteration:
            break
    return convert_list


def custom_predict(args, test_dataset, model, tokenizer, generation_config):
    for data in test_dataset:
        input_tokens = tokenizer(data["input_ids"], max_length=args.max_seq_length, truncation=True, padding=True)
        print("input_tokens", input_tokens)
        generation_output = model.generate(
            input_ids=input_tokens["input_ids"],
            generation_config=generation_config,
            return_dict_in_generate=True,
            max_new_tokens=64,
        )
        print('\\nAnswer: ', tokenizer.decode(generation_output.sequences[0]))
        print('Ground truth: ', data["labels"])


def safe_div(num, denom):
    if denom > 0:
        return num / denom
    else:
        return 0


def compute_f1(predicted, gold, matched):
    precision = safe_div(matched, predicted)
    recall = safe_div(matched, gold)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1


def find_arg_span(arg, context_words, trigger_start, trigger_end, head_only=False, doc=None):
    match = None
    arg_len = len(arg)
    min_dis = len(context_words)  # minimum distance to trigger
    for i, w in enumerate(context_words):
        if context_words[i:i + arg_len] == arg:
            if i < trigger_start:
                dis = abs(trigger_start - i - arg_len)
            else:
                dis = abs(i - trigger_end)
            if dis < min_dis:
                match = (i, i + arg_len - 1)
                min_dis = dis

    if match and head_only:
        assert (doc != None)
        match = find_head(match[0], match[1], doc)
    return match


def find_head(arg_start, arg_end, doc):
    cur_i = arg_start
    while doc[cur_i].head.i >= arg_start and doc[cur_i].head.i <= arg_end:
        if doc[cur_i].head.i == cur_i:
            # self is the head
            break
        else:
            cur_i = doc[cur_i].head.i

    arg_head = cur_i

    return (arg_head, arg_head)


def get_topk_idx(data, k):
    data = np.array(data)
    idx = data.argsort()[-k:][::-1]
    return list(idx)
