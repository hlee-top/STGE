## STGE
Code for paper: Bridging the Gap: Aligning Language Model Generation with Structured Information Extraction via Controllable State Transition
## Environment
To run our code, please install all the dependency packages by using the following command:
```
conda create --name stge python=3.12.4
pip install -r requirements.txt
```
## Dateset
ACE05 dataset accessed from [LDC](https://catalog.ldc.upenn.edu/LDC2006T06) (not freely available) and pre-processed following [DyGIE++](https://github.com/dwadden/dygiepp).
Download the NERD dataset using the following command.
```
wget https://cloud.tsinghua.edu.cn/f/c1f71c011d6b461786bc/?dl=1
```
## Download model
* Llama-3.1-8B-Instruct https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
* jina-embeddings-v2-base-en https://huggingface.co/jinaai/jina-embeddings-v2-base-en
* Roberta-large https://huggingface.co/FacebookAI/roberta-large
## Data processing
```
python data_precess.py --dataset_type ace --task_type ner
python few_shot_setting.py --dataset_type ace --task_type ner --shot_num 20
python scorer_data_precess.py --dataset_type ace --task_type ner --shot_num 20 --train_save_path 20-shot_train_per.tsv
python scorer_data_precess.py --dataset_type nerd --task_type ner --train_save_path train_nerd_all.tsv --dev_save_path dev_nerd_all.tsv --max_num 1
```
## Pre-training scorer
```
nohup python -u scorer_train.py --dataset_type nerd --task_type ner --scorer_train_data_path ../../dataset/Few-NERD/scorer_data/ner/train_nerd_all.tsv --scorer_dev_data_path ../../dataset/Few-NERD/scorer_data/ner/dev_nerd_all.tsv --model_save nerd_train_large_all --batch_size 32 --model_name ../../pretrain_model/roberta-large --hidden_size 1024 --epoch 3 > log/nerd/socrer_train_nerd.txt 2>&1 &
```
## Evaluate
Run the following command to perform iterative training and extraction, and then run stge.py for extraction only.
```
nohup python -u scorer_train.py --dataset_type ace --task_type ner --max_new_tokens 256 --method_type stge --logit_strategy fusion --shot_num 20 --demo_num 2 --load_scorer_path  model/nerd/nerd_train_large_all --model_save nerd_train_large_20_shot_iterative --batch_size 32 --epoch 10 --model_name ../../pretrain_model/roberta-large --hidden_size 1024 --train_type stge > log/ace/ner.txt 2>&1 &
```

## Citation
```
@inproceedings{li2025bridging,
  title={Bridging the Gap: Aligning Language Model Generation with Structured Information Extraction via Controllable State Transition},
  author={Li, Hao and Ren, Yubing and Cao, Yanan and Li, Yingjie and Fang, Fang and Lin, Zheng and Wang, Shi},
  booktitle={Proceedings of the ACM on Web Conference 2025},
  pages={1811--1821},
  year={2025}
}
```