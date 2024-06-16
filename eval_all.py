from rouge_score import rouge_scorer
import json
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
import warnings
warnings.filterwarnings("ignore")


def calc_score(truth, predict, metric):
    truth = truth.lower().strip()
    predict = predict.lower().strip()
    if metric == 'BLEU-4':
        return sentence_bleu([truth.split()], predict.split())
    elif metric == 'BLEU-1':
        return sentence_bleu([truth.split()], predict.split(), weights=[1, 0, 0, 0])
    elif metric == 'ROUGE-L':
        rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = rouge.score(truth, predict)
        return scores['rougeL'].fmeasure
    elif metric == 'Exactly Match' or metric == 'EM':
        return int(truth == predict)
    elif metric == 'IN':
        return int(truth in predict)


attack_method = 'per'
model_name = 'llama-3'   # llama-3  gpt-35-turbo
dataset_name = 'chat'   # wiki  chat
folder = 'a2'    # 1  3  5
# 6  10  14
protect_method = 'sync'   # 'sync'  llm 'para'   'ori'  agent2  ZeroGen
now_k = 0


with open(f'truth/{attack_method}-{dataset_name}-truth.json', 'r', encoding='utf-8') as file:
    ground_truth = json.loads(file.read())

with open(f'outputs/{folder}/{attack_method}-{dataset_name}-{protect_method}-{model_name}-output.json', 'r', encoding='utf-8') as file:
    all_output = json.loads(file.read())

num_dataset = 4
output = [o[now_k] for o in all_output]
metric_lst = ['BLEU-1', 'ROUGE-L']

flag = 1    # 对o-final(聚合后)
if flag:
    print(f'{protect_method}-{now_k}', end='\t')
    print('\t'.join(metric_lst))
    all_shot_score = []
    for i in range(len(metric_lst)):
        all_shot_score.append([])
    for i in range(len(ground_truth)):
        for j in range(len(metric_lst)):
            all_shot_score[j].append(calc_score(str(ground_truth[i]), str(output[i]), metric=metric_lst[j]))
    all_ans = np.array(all_shot_score)
    if dataset_name != 'chat':
        for i in range(num_dataset):
            # print(f'dataset{i+1}', end='\t')
            all_score = [np.mean(all_ans[metric][i*250:(i+1)*250]) for metric in range(len(metric_lst))]
            all_score_str = [str(a) for a in all_score]
            print('\t'.join(all_score_str))
    else:
        all_score = [np.mean(all_ans[metric]) for metric in range(len(metric_lst))]
        all_score_str = [str(a) for a in all_score]
        print('\t'.join(all_score_str))
