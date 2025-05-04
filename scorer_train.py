import torch
from utils import *
from transformers import RobertaTokenizer, AdamW, get_linear_schedule_with_warmup
from scorer_data_precess import get_dataloaber
from stge import extract_and_evaluate
from scorer_data_precess import save_data, get_pred_data
from config import get_args_parser
from scorer import ScorerModel


def train(args, model, train_data, optimizer):
    max_acc = 0
    for epoch_idx in range(args.epoch):
        pred_pos, pred_neg = 0, 0
        acc_num, num = 0, 0
        train_loss = 0
        for idx, batch in enumerate(train_data):
            model.train()
            pred, loss = model(batch)
            train_pred = pred.argmax(1).cpu().numpy().tolist()
            train_gold = batch[-1].cpu().numpy().tolist()
            num += len(train_gold)
            for pred_idx in range(len(train_pred)):
                if train_pred[pred_idx] == train_gold[pred_idx]:
                    acc_num += 1
                if train_pred[pred_idx] == 1:
                    pred_pos += 1
                else:
                    pred_neg += 1

            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            model.zero_grad()

        print("train pred_pos, pred_neg", pred_pos, pred_neg)
        train_acc = acc_num/num
        print("epoch={}, train_acc={}, train_loss={}".format(epoch_idx, train_acc, train_loss))
        torch.save(model.state_dict(), args.model_save_path)
        print("save model")
    print("max_acc: ", max_acc)


def begin_supervise_train(args):
    if args.load_scorer_path == "":
        model = ScorerModel(args).to(args.device)
    else:
        print("load {}".format(args.load_scorer_path))
        model = ScorerModel(args)
        model.load_state_dict(torch.load(args.load_scorer_path, map_location="cpu"))
        model = model.to(args.device)
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    train_data = get_dataloaber(args, args.scorer_train_data_path, tokenizer)
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    train(args, model, train_data, optimizer)
    extract_and_evaluate(args, args.convert_path + "/test.json")



def get_iterative_data(args, epoch_idx, tokenizer):
    all_data = read_json(args.save_result_path)
    pred_data_list = get_pred_data(args, all_data)
    if len(pred_data_list) != 0:
        train_save_path = args.scorer_data_path + "{}_train_epoch_{}.tsv".format(args.model_save, epoch_idx)
        save_data(pred_data_list, train_save_path)
        train_data = get_dataloaber(args, train_save_path, tokenizer)
        return train_data
    else:
        return []


def begin_iterative_train(args):
    if args.load_scorer_path == "":
        model = ScorerModel(args).to(args.device)
    else:
        print("load {}".format(args.load_scorer_path))
        model = ScorerModel(args)
        model.load_state_dict(torch.load(args.load_scorer_path, map_location="cpu"))
        model = model.to(args.device)
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    original_data = get_dataloaber(args, args.scorer_train_data_path, tokenizer)

    original_model_path = args.model_save_path
    args.model_save_path = args.load_scorer_path
    args.save_result_path = "output/{}/{}_{}_result_original.json".format(
        args.dataset_type, args.model_save, args.llm_name)
    with torch.no_grad():
        current_f1 = extract_and_evaluate(args, args.shot_data_path)
    print("current_f1", current_f1)
    print("begin train", "-"*30)
    for epoch_idx in range(args.epoch):
        if args.pred_max_num == 0:
            all_train_data = [iter(original_data)]
        else:
            pred_data = get_iterative_data(args, epoch_idx, tokenizer)
            all_train_data = [iter(pred_data)]
        pred_pos, pred_neg = 0, 0
        acc_num, num = 0, 0
        train_loss = 0
        for i in range(len(all_train_data)):
            iter_dataloader = all_train_data[i]
            for idx, batch in enumerate(iter_dataloader):
                model.train()
                pred, loss = model(batch)
                train_pred = pred.argmax(1).cpu().numpy().tolist()
                train_gold = batch[-1].cpu().numpy().tolist()
                num += len(train_gold)
                for pred_idx in range(len(train_pred)):
                    if train_pred[pred_idx] == train_gold[pred_idx]:
                        acc_num += 1
                    if train_pred[pred_idx] == 1:
                        pred_pos += 1
                    else:
                        pred_neg += 1

                loss.backward()
                train_loss += loss.item()
                optimizer.step()
                model.zero_grad()

        print("train pred_pos, pred_neg", pred_pos, pred_neg)
        train_acc = acc_num / num
        print("epoch={}, train_acc={}, train_loss={}".format(epoch_idx, train_acc, train_loss))
        args.model_save_path = original_model_path
        torch.save(model.state_dict(), args.model_save_path)
        print("save model", args.model_save_path)
        args.save_result_path = "output/{}/{}_{}_epoch{}.json".format(
            args.dataset_type, args.model_save, args.llm_name, epoch_idx)
        with torch.no_grad():
            current_f1 = extract_and_evaluate(args, args.shot_data_path)
        print("current_f1", current_f1)
    args.save_result_path = "output/{}/{}_{}_all.json".format(
        args.dataset_type, args.model_save, args.llm_name)
    with torch.no_grad():
        extract_and_evaluate(args, args.convert_path + "/test.json")


if __name__ == '__main__':
    args = get_args_parser()
    print(args)
    if args.train_type == "supervise":
        begin_supervise_train(args)
    elif args.train_type == "stge":
        begin_iterative_train(args)
    else:
        print("train_type error")


