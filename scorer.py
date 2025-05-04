import ast
from transformers import RobertaTokenizer, RobertaModel, AutoTokenizer
import torch
import torch.nn as nn
from utils import *
from scorer_data_precess import get_ner_str, get_eae_str, get_re_str


class StgeScorer:
    def __init__(self, args):
        self.args = args
        self.model = ScorerModel(args)
        print("StgeScorer load model_save_path", args.model_save_path)
        self.model.load_state_dict(torch.load(args.model_save_path, map_location="cpu"))
        self.model = self.model.to(args.device)
        self.tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
        self.model.eval()
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_path[args.llm_name])

    def get_state(self, inputs, judge_logits):
        model_input = self.tokenizer([inputs], padding="max_length", truncation=True,
                                     max_length=self.args.max_socrer_data_length)
        model_input["input_ids"] = torch.LongTensor(model_input["input_ids"]).to(self.args.device)
        model_input["attention_mask"] = torch.LongTensor(model_input["attention_mask"]).to(self.args.device)
        with torch.no_grad():
            logits, _ = self.model([model_input["input_ids"], model_input["attention_mask"]], is_train=False)
            new_logits = logits + judge_logits
            pred_list = new_logits.argmax(1).cpu().detach().numpy().tolist()

            if pred_list[0] == 1:
                return True
            else:
                return False

    def judge_value(self, output_dict, value_num):
        for key, value in output_dict.items():
            if len(value) >= value_num:
                return False
        return True

    def state_scorer(self, current_state, current_output_id, event_info, judge_logits):
        if current_state == "start":
            if self.args.task_type == "eae":
                inputs = get_eae_str(event_info["original_input"], event_info["event_type"], event_info["trigger_word"], {})
            elif self.args.task_type == "ner":
                inputs = get_ner_str(event_info["original_input"], {})
            elif self.args.task_type == "re":
                inputs = get_re_str(event_info["original_input"], {})
            return self.get_state(inputs, judge_logits)
        elif current_state == "end_value":
            current_outputs = filt_space(self.llm_tokenizer.decode(current_output_id))
            current_outputs += "}"
            pred_dict = ast.literal_eval(current_outputs)
            if self.args.value_num != 0 and self.judge_value(pred_dict, self.args.value_num) is False:
                return False
            if self.args.task_type == "re":
                test_span_list = []
                for re_type in pred_dict.keys():
                    for span_list in pred_dict[re_type]:
                        if span_list in test_span_list:
                            return False
                        test_span_list.append(span_list)
            if self.args.task_type == "eae":
                inputs = get_eae_str(event_info["original_input"], event_info["event_type"], event_info["trigger_word"],
                                     pred_dict)
            elif self.args.task_type == "ner":
                inputs = get_ner_str(event_info["original_input"], pred_dict)
            elif self.args.task_type == "re":
                inputs = get_re_str(event_info["original_input"], pred_dict)
            return self.get_state(inputs, judge_logits)
        elif current_state == "generate_key_process":
            current_outputs = filt_space(self.llm_tokenizer.decode(current_output_id))
            current_outputs += "]}"
            pred_dict = ast.literal_eval(current_outputs)
            if self.args.value_num != 0 and self.judge_value(pred_dict, self.args.value_num) is False:
                return False
            no_key = list(pred_dict.keys())[-1]
            if self.args.task_type == "eae":
                if no_key in pred_dict:
                    pred_dict.pop(no_key)
                inputs = get_eae_str(event_info["original_input"], event_info["event_type"], event_info["trigger_word"],
                                     pred_dict, no_key)
            elif self.args.task_type == "ner":
                inputs = get_ner_str(event_info["original_input"], pred_dict, no_key)
            elif self.args.task_type == "re":
                inputs = get_re_str(event_info["original_input"], pred_dict, no_key)
            return self.get_state(inputs, judge_logits)
        elif current_state == "generate_value_process":
            current_outputs = filt_space(self.llm_tokenizer.decode(current_output_id))
            current_outputs += "]}"
            pred_dict = ast.literal_eval(current_outputs)
            if self.args.value_num != 0 and self.judge_value(pred_dict, self.args.value_num) is False:
                return False
            for role in pred_dict.keys():
                if len(pred_dict[role]) != len(list(set(pred_dict[role]))):
                    return False
            if self.args.task_type == "eae":
                inputs = get_eae_str(event_info["original_input"], event_info["event_type"], event_info["trigger_word"],
                                     pred_dict, list(pred_dict.keys())[-1])
            elif self.args.task_type == "ner":
                inputs = get_ner_str(event_info["original_input"], pred_dict, list(pred_dict.keys())[-1])
            elif self.args.task_type == "re":
                inputs = ""
            return self.get_state(inputs, judge_logits)
        elif current_state == "middle_value":
            current_outputs = filt_space(self.llm_tokenizer.decode(current_output_id))
            current_outputs += "]}"
            pred_dict = ast.literal_eval(current_outputs)
            if self.args.value_num != 0 and self.judge_value(pred_dict, self.args.value_num) is False:
                return False
            test_span_list = []
            for re_type in pred_dict.keys():
                for span_list in pred_dict[re_type]:
                    if span_list in test_span_list:
                        return False
                    test_span_list.append(span_list)
            if self.args.task_type == "re":
                inputs = get_re_str(event_info["original_input"], pred_dict, list(pred_dict.keys())[-1])
            return self.get_state(inputs, judge_logits)
        else:
            print("current_state", current_state)
            print("state error")
            return False




class ScorerModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = RobertaModel.from_pretrained(args.model_name)
        self.classifier = nn.Linear(args.hidden_size, 2)

    def forward(self, batch, is_train=True):
        if is_train:
            input_ids, input_attens, labels = batch
        else:
            input_ids, input_attens = batch
            labels = None
        outputs = self.model(input_ids, attention_mask=input_attens)
        outputs = outputs["pooler_output"]
        logits = self.classifier(outputs)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            return logits, loss
        else:
            logits = torch.softmax(logits, dim=-1)
            return logits, None
