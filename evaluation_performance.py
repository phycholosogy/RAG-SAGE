from rouge_score import rouge_scorer
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
import warnings
warnings.filterwarnings("ignore")
from datasets import Dataset
from ragas.metrics import (
    answer_correctness,
    answer_similarity
)
from ragas import evaluate
import json
import argparse


def calc_score(truth, predict, con, ques, metric):
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
    elif metric == 'answer_similarity' or metric == 'answer_correctness':
        data_samples = {
            'question': ques,
            'answer': predict,
            'contexts': con,
            'ground_truth': truth
        }
        dataset = Dataset.from_dict(data_samples)
        if metric == 'answer_correctness':
            score = evaluate(dataset, metrics=[answer_correctness], raise_exceptions=False,
                             llm='YOUR LLM MODEL', embeddings='YOUR EMBEDDING')
        else:
            score = evaluate(dataset, metrics=[answer_similarity], raise_exceptions=False,
                             llm='YOUR LLM MODEL', embeddings='YOUR EMBEDDING')
        return score[metric]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eval performance')

    parser.add_argument('--k', type=int, default=1, help='context numbers')
    parser.add_argument('--protect-method', type=str,
                        choices=["sync",        # Our proposed method, synthetic data
                                 "agent2",      # Our proposed method, using 2 agents to make the generation less risk
                                 "para",        # paragraph, the baseline for comparison
                                 "ZeroGen",     # the baseline for comparison
                                 "attrPrompt",  # the baseline for comparison
                                 "ori",         # do not use any protect method
                                 "llm",         # do not use RAG
                                 ])
    parser.add_argument('--model', type=str, default='gpt-35-turbo',
                        choices=['gpt-4', 'gpt-35-turbo', 'llama-3'])
    parser.add_argument('--dataset-name', type=str, default='chat', choices=['chat', 'wiki'])
    # llama-3  gpt-35-turbo
    attack_method = 'per'
    args = parser.parse_args()
    k = args.k
    protect_method = args.protect_method
    model = args.model
    dataset_name = args.dataset_name

    with open(f'questions/{attack_method}-{dataset_name}-question.json', 'r', encoding='utf-8') as f:
        questions = json.load(f)
    with open(f'outputs/{attack_method}-{dataset_name}-{protect_method}-{model}-output.json', 'r', encoding='utf-8') as f:
        all_answer = json.load(f)
    if args.protect_method != 'llm':
        with open(f'contexts/{attack_method}-{dataset_name}-{protect_method}-context.json', 'r', encoding='utf-8') as f:
            all_context = json.load(f)
    else:
        with open(f'contexts/{attack_method}-{dataset_name}-ori-context.json', 'r', encoding='utf-8') as f:
            all_context = json.load(f)
    with open(f'truth/{attack_method}-{dataset_name}-truth.json', 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)
    print(len(questions), len(all_answer), len(all_context), len(ground_truth))

    metric_lst = ['BLEU-1', 'ROUGE-L', 'answer_similarity', 'answer_correctness']

    for now_k in range(k):
        output = [o[now_k] for o in all_answer]
        context = [con_[:now_k + 1] for con_ in all_context]
        if protect_method == 'llm':
            context = [[t] for t in ground_truth]
        print(f'{protect_method}-{now_k}', end='\t')
        print('\t'.join(metric_lst))
        all_shot_score = []
        for i in range(len(metric_lst)):
            all_shot_score.append([])
        for i in range(len(ground_truth)):
            for j in range(len(metric_lst)):
                all_shot_score[j].append(calc_score(str(ground_truth[i]), str(output[i]), context[i], questions[i], metric=metric_lst[j]))
        all_ans = np.array(all_shot_score)
        if dataset_name != 'chat':
            for i in range(4):
                all_score = [np.mean(all_ans[metric][i*250:(i+1)*250]) for metric in range(len(metric_lst))]
                all_score_str = [str(a) for a in all_score]
                print('\t'.join(all_score_str))
        else:
            all_score = [np.mean(all_ans[metric]) for metric in range(len(metric_lst))]
            all_score_str = [str(a) for a in all_score]
            print('\t'.join(all_score_str))
