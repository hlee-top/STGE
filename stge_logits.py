import torch
import torch.nn.functional as F
from transformers import LogitsProcessor, LogitsProcessorList, AutoTokenizer, AutoModelForCausalLM
from utils import *
from scorer import StgeScorer


name2class = {
    "llama3.1_8b_instruct": "llama",
}


def construct_model_class(args):
    if name2class[args.llm_name] == "llama":
        model_class = LLamaClass(args, args.max_new_tokens)
    else:
        print("connot find model class")
        model_class = None
    return model_class


def multi_token_force(current_id, original_input_id):
    force_ids = []
    for ids in range(len(original_input_id) - 1):
        if original_input_id[ids] == current_id:
            force_ids.append(original_input_id[ids + 1])
    return force_ids


def get_force_id(role_id_list, key_id_list=None):
    force_words_ids = []
    if key_id_list is None:
        for role_id in role_id_list:
            if role_id[0] not in force_words_ids:
                force_words_ids.append(role_id[0])
    else:
        for role_id in role_id_list:
            if key_id_list == role_id[:len(key_id_list)]:
                force_words_ids.append(role_id[len(key_id_list)])

    return force_words_ids


def stge_function(model_kwargs, current_state, output_ids, key_state=None, generate_dict=None, next_token_logits=None):
    state_dict = model_kwargs.state_dict
    force_words_ids = []
    if current_state == "initial":
        current_state = "start"
        force_words_ids = model_kwargs.state_dict["start"][0]
    elif current_state == "start":
        judge_logits = torch.index_select(next_token_logits, -1,
                                          torch.LongTensor([state_dict["end"][0][0],
                                                                                   state_dict["generate_key"][0][0]]).to(next_token_logits.device))
        judge_logits = F.softmax(judge_logits, dim=-1)
        flag = model_kwargs.score_model.state_scorer(current_state, output_ids, model_kwargs.event_info, judge_logits)
        if flag:
            force_words_ids = state_dict["generate_key"][0]
            current_state = "generate_key_process"
        else:
            force_words_ids = state_dict["end"][0]
            current_state = "end"
    elif current_state == "start_key":
        force_words_ids = state_dict["generate_key"][0]
        current_state = "generate_key_process"
    elif current_state == "start_value":
        force_words_ids = state_dict["generate_value"][0]
        current_state = "generate_value"
    elif current_state == "generate_value":
        current_key = generate_dict["key"][-1]
        if current_key not in generate_dict["value"]:
            generate_dict["value"][current_key] = 1
        else:
            generate_dict["value"][current_key] += 1
        force_words_ids = model_kwargs.original_input_id + state_dict["generate_value"][1]
        current_state = "generate_value_process"
    elif current_state == "end_value":
        judge_logits = torch.index_select(next_token_logits, -1,
                                          torch.LongTensor([state_dict["end"][0][0],
                                                                                   state_dict["delimiter"][0][0]]).to(next_token_logits.device))
        judge_logits = F.softmax(judge_logits, dim=-1)
        flag = model_kwargs.score_model.state_scorer(current_state, output_ids, model_kwargs.event_info, judge_logits)
        if flag and len(model_kwargs.role_id) > 0:
            force_words_ids = state_dict["delimiter"][0]
            current_state = "start_key"
        else:
            force_words_ids = state_dict["end"][0]
            current_state = "end"
    elif current_state == "generate_key_process":
        if key_state is None:
            current_state = "generate_key_process"
            key_state = {"key_id_list": []}
            force_words_ids = get_force_id(model_kwargs.role_id)
        else:
            key_state["key_id_list"].append(output_ids[-1])
            if len(key_state["key_id_list"]) == 1:
                generate_dict["key"].append(key_state["key_id_list"][0])
            end_flag = False
            for role_id in model_kwargs.role_id:
                if role_id == key_state["key_id_list"]:
                    end_flag = True
                    break
            if end_flag is False:
                current_state = "generate_key_process"
                force_words_ids = get_force_id(model_kwargs.role_id, key_state["key_id_list"])
            else:
                model_kwargs.role_id.remove(key_state["key_id_list"])
                key_state = None
                judge_logits = torch.index_select(next_token_logits, -1,
                                                  torch.LongTensor([state_dict["end_value"][0][0],
                                                                    state_dict["generate_value"][0][0]]).to(
                                                      next_token_logits.device))
                judge_logits = F.softmax(judge_logits, dim=-1)
                flag = model_kwargs.score_model.state_scorer(current_state, output_ids,
                                                                model_kwargs.event_info, judge_logits)
                if flag:
                    current_state = "generate_value"
                    force_words_ids = state_dict["generate_value"][0]
                else:
                    current_state = "end_value"
                    force_words_ids = state_dict["end_value"][0]
    elif current_state == "generate_value_process":
        current_max_tokens = output_ids[-1]
        next_token_scores = F.softmax(next_token_logits, dim=-1)
        next_max_tokens = torch.argmax(next_token_scores, dim=-1).cpu().numpy().tolist()[0]
        if current_max_tokens == state_dict["generate_value"][1][0]:
            judge_logits = torch.index_select(next_token_logits, -1,
                                              torch.LongTensor([state_dict["end_value"][0][0],
                                                                state_dict["delimiter"][0][0]]).to(
                                                  next_token_logits.device))
            judge_logits = F.softmax(judge_logits, dim=-1)
            flag = model_kwargs.score_model.state_scorer(current_state, output_ids, model_kwargs.event_info, judge_logits)
            if flag:
                current_state = "start_value"
                force_words_ids = state_dict["delimiter"][0]
            else:
                current_state = "end_value"
                force_words_ids = state_dict["end_value"][0]
        else:
            current_state = "generate_value_process"
            force_words_ids = multi_token_force(current_max_tokens, model_kwargs.original_input_id)
            force_words_ids.extend(state_dict["generate_value"][1])
    else:
        print("current_state", current_state)
        print("state error")
    return force_words_ids, current_state, key_state, generate_dict



def stge_re_function(model_kwargs, current_state, output_ids, key_state=None, value_state=None,
                      generate_dict=None, next_token_logits=None):
    state_dict = model_kwargs.state_dict
    force_words_ids = []
    if current_state == "initial":
        current_state = "start"
        force_words_ids = model_kwargs.state_dict["start"][0]
    elif current_state == "start":
        judge_logits = torch.index_select(next_token_logits, -1,
                                          torch.LongTensor([state_dict["end"][0][0],
                                                            state_dict["generate_key"][0][0]]).to(
                                              next_token_logits.device))
        judge_logits = F.softmax(judge_logits, dim=-1)
        flag = model_kwargs.score_model.state_scorer(current_state, output_ids, model_kwargs.event_info,
                                                        judge_logits)
        if flag:
            force_words_ids = state_dict["generate_key"][0]
            generate_dict["value_num"] = 0
            current_state = "generate_key_process"
        else:
            force_words_ids = state_dict["end"][0]
            current_state = "end"
    elif current_state == "start_key":
        generate_dict["value_num"] = 0
        force_words_ids = state_dict["generate_key"][0]
        current_state = "generate_key_process"
    elif current_state == "start_value":
        force_words_ids = state_dict["generate_value"][0]
        current_state = "generate_value"
    elif current_state == "generate_value":
        generate_dict["value_num"] += 1
        current_key = generate_dict["key"][-1]
        if current_key not in generate_dict["value"]:
            generate_dict["value"][current_key] = 1
        else:
            generate_dict["value"][current_key] += 1
        force_words_ids = model_kwargs.original_input_id + state_dict["generate_value"][1]
        value_state = None
        current_state = "generate_value_process"
    elif current_state == "end_value":
        judge_logits = torch.index_select(next_token_logits, -1,
                                          torch.LongTensor([state_dict["end"][0][0],
                                                            state_dict["delimiter"][0][0]]).to(
                                              next_token_logits.device))
        judge_logits = F.softmax(judge_logits, dim=-1)
        flag = model_kwargs.score_model.state_scorer(current_state, output_ids, model_kwargs.event_info,
                                                        judge_logits)
        if flag and len(model_kwargs.role_id) > 0:
            force_words_ids = state_dict["delimiter"][0]
            current_state = "start_key"
        else:
            force_words_ids = state_dict["end"][0]
            current_state = "end"
    elif current_state == "generate_key_process":
        if key_state is None:
            current_state = "generate_key_process"
            key_state = {"key_id_list": []}
            force_words_ids = get_force_id(model_kwargs.role_id)
        else:
            key_state["key_id_list"].append(output_ids[-1])
            if len(key_state["key_id_list"]) == 1:
                generate_dict["key"].append(key_state["key_id_list"][0])
            end_flag = False
            for role_id in model_kwargs.role_id:
                if role_id == key_state["key_id_list"]:
                    end_flag = True
                    break
            if end_flag is False:
                current_state = "generate_key_process"
                force_words_ids = get_force_id(model_kwargs.role_id, key_state["key_id_list"])
            else:
                model_kwargs.role_id.remove(key_state["key_id_list"])
                key_state = None
                judge_logits = torch.index_select(next_token_logits, -1,
                                                  torch.LongTensor([state_dict["end_value"][0][0],
                                                                    state_dict["init_key"][0][0]]).to(
                                                      next_token_logits.device))
                judge_logits = F.softmax(judge_logits, dim=-1)
                flag = model_kwargs.score_model.state_scorer(current_state, output_ids,
                                                                model_kwargs.event_info, judge_logits)
                if flag:
                    current_state = "start_value"
                    force_words_ids = state_dict["init_key"][0]
                else:
                    current_state = "end_value"
                    force_words_ids = state_dict["end_value"][0]
    elif current_state == "generate_value_process":
        current_max_tokens = output_ids[-1]
        next_token_scores = F.softmax(next_token_logits, dim=-1)
        if value_state is None:
            value_state = {"key_id_list": []}
        if current_max_tokens == state_dict["generate_value"][1][0]:
            if generate_dict["value_num"] % 2 != 0:
                current_state = "start_value"
                force_words_ids = state_dict["delimiter"][0]
            else:
                current_state = "middle_value"
                force_words_ids = state_dict["end_value"][0]
        else:
            value_state["key_id_list"].append(output_ids[-1])
            current_state = "generate_value_process"
            force_words_ids = multi_token_force(current_max_tokens, model_kwargs.original_input_id)
            force_words_ids.extend(state_dict["generate_value"][1])


    elif current_state == "middle_value":
        judge_logits = torch.index_select(next_token_logits, -1,
                                          torch.LongTensor([state_dict["end_value"][0][0],
                                                            state_dict["delimiter"][0][0]]).to(
                                              next_token_logits.device))
        judge_logits = F.softmax(judge_logits, dim=-1)
        flag = model_kwargs.score_model.state_scorer(current_state, output_ids,
                                                        model_kwargs.event_info, judge_logits)
        if flag:
            current_state = "more_value"
            force_words_ids = state_dict["delimiter"][0]
        else:
            current_state = "end_value"
            force_words_ids = state_dict["end_value"][0]
    elif current_state == "more_value":
        current_state = "start_value"
        force_words_ids = state_dict["init_key"][0]
    else:
        print("re current_state", current_state)
        print("state error")
    return force_words_ids, current_state, key_state, value_state, generate_dict


class STGEConfig:
    def __init__(self, args, original_input_id, role_id, state_dict, ground_id_dict, event_info,
                 score_model, task_type, entity_id_list, vocab_size):
        self.args = args
        self.original_input_id = original_input_id
        self.role_id = role_id
        self.state_dict = state_dict
        self.ground_id_dict = ground_id_dict
        self.event_info = event_info
        self.score_model = score_model
        self.task_type = task_type
        self.entity_id_list = entity_id_list
        self.vocab_size = vocab_size


class STGELogitsProcessor(LogitsProcessor):
    def __init__(self, model_kwargs):
        self.model_kwargs = model_kwargs
        self.current_state = "initial"
        self.output_ids = []
        self.generate_dict = {"key": [], "value": {}, "head_value_id": [], "previous_state": None}
        self.key_state = None
        self.value_state = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        """Process logits"""
        if self.current_state != "initial":
            self.output_ids.append(input_ids[:, -1].cpu().numpy().tolist()[0])
        force_words_ids = []
        if self.model_kwargs.args.task_type == "re":
            force_words_ids, new_current_state, self.key_state, self.value_state, self.generate_dict = stge_re_function(
                self.model_kwargs,
                self.current_state,
                self.output_ids,
                self.key_state,
                self.value_state,
                self.generate_dict,
                scores)
        else:
            force_words_ids, new_current_state, self.key_state, self.generate_dict = stge_function(
                self.model_kwargs,
                self.current_state,
                self.output_ids,
                self.key_state,
                self.generate_dict,
                scores)
            self.generate_dict["previous_state"] = self.current_state
            if self.generate_dict["previous_state"] == "generate_value":
                next_max_tokens = torch.argmax(scores, dim=-1).cpu().numpy().tolist()[0]
                self.generate_dict["value_max_logit"] = scores[:, next_max_tokens]
            self.current_state = new_current_state

        if len(force_words_ids) != 0:
            all_id_list = [i for i in range(self.model_kwargs.vocab_size)]
            fill_index = torch.LongTensor(list(set(all_id_list).difference(set(force_words_ids))))
            scores.index_fill_(-1, torch.LongTensor(fill_index).to(scores.device), -1000)
        return scores

class LLamaClass:
    def __init__(self, args, max_new_tokens):
        self.args = args
        model_path = llm_path[args.llm_name]
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
        self.score_model = StgeScorer(args)

        self.max_new_tokens = max_new_tokens
        self.state_dict = {
            "start": [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' {'))],
            "generate_key": [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' "')),
                                   self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' " : ['))],
            "generate_value": [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' "')),
                                     self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' "'))],
            "end_value": [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' ]'))],
            "end": [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' }'))],
            "delimiter": [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' ,'))],
            "init_key": [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' [')),]
        }
        print("self.state_dict", self.state_dict)

    def get_label_id(self, label_list):
        label2id = {}
        role_id = []
        for label in label_list:
            token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(" " + label))
            role_id.append(token_id + self.state_dict["generate_key"][1])
            label2id[label] = token_id[0]
        return role_id, label2id

    def inference(self, original_input, event_info, input_content, eos_token_id=None, entity_list=None):
        event_info["original_input"] = original_input
        input_ids = self.tokenizer(input_content, return_tensors="pt").to("cuda")
        original_input_id = self.tokenizer(" " + original_input)["input_ids"][1:]
        role_id, label2id = self.get_label_id(event_info["label_list"])
        ground_output_id = \
        self.tokenizer(" " + event_info["ground_output"].replace('"', ' " ').replace("[[", "[ [").replace("]", "] ").replace('  "', ' "'))[
            "input_ids"][1:]

        if entity_list is None:
            entity_id_list = None
        else:
            entity_id_list = []
            for entity in entity_list:
                entity_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(" " + entity))
                entity_id_list.append(entity_id)
        ground_id_dict = {"ground_output": event_info["ground_output"], "ground_output_id": ground_output_id, "label2id": label2id}

        model_kwargs = STGEConfig(self.args, original_input_id, role_id, self.state_dict, ground_id_dict, event_info,
                                  self.score_model, self.args.task_type, entity_id_list, vocab_length[self.args.llm_name])
        logits_processor = STGELogitsProcessor(model_kwargs)

        outputs = self.model.generate(**input_ids,
                                      eos_token_id=eos_token_id,
                                      pad_token_id=eos_token_id,
                                      max_new_tokens=self.max_new_tokens,
                                      do_sample=False,
                                      logits_processor=LogitsProcessorList([logits_processor]),
                                      )
        returned = self.tokenizer.decode(outputs[0])
        
        
        return returned




