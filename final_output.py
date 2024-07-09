import os
import json
import argparse
from tqdm import tqdm
from openai import AzureOpenAI
import transformers
import torch


def get_llm_client(llm_name: str = 'gpt-35-turbo'):
    client = "YOUR CLIENT TO GENERATE LLM OUTPUT"
    return client


def get_llm_output(prompt, llm_client, model_name, system_content="You are a helpful assistant."):
    # PLEASE FIT THIS FUNCTION TO YOUR OWN LLM CLIENT
    if model_name.find('llama') != -1:
        out = llm_client(prompt,
                         max_new_tokens=256,
                         temperature=0.6,
                         do_sample=True,
                         pad_token_id=llm_client.tokenizer.eos_token_id)
        output = out[0]['generated_text'].strip(prompt)
    else:
        messages = [{"role": "system", "content": system_content},
                    {"role": "user", "content": prompt}, ]
        output = ''
        for _ in range(8):
            try:
                response = llm_client.chat.completions.create(model=model_name,
                                                              messages=messages,
                                                              max_tokens=256,
                                                              n=1,
                                                              temperature=0.6)
                output = response.choices[0].message.content
            except Exception as _:
                output = ''
            if output != '':
                break
        if output == '':
            global num_error
            num_error += 1
            print('Error!')
    return output


def get_query_output_k(questions, contexts, generate_llm):
    llm_client = get_llm_client(generate_llm)
    all_outputs = []
    for i in tqdm(range(len(questions)), desc="geenrate final out"):
        con = contexts[i]
        con = [str(c) for c in con]
        que = questions[i]
        output_ = []
        for j in range(len(con)):
            final_con = '\n\n'.join(con[:j+1])
            prompt = f"Context: {final_con}\nQuestion: {que}\nAnswer:"
            output = get_llm_output(prompt, llm_client, generate_llm, 'You are a helpful assistant.')
            output_.append(output)
        all_outputs.append(output_)
    return all_outputs


def get_performance_output_k(questions, generate_llm):
    llm_client = get_llm_client(generate_llm)
    all_outputs = []
    for i in tqdm(range(len(questions)), desc="generate final out"):
        que = questions[i]
        o_zero = get_llm_output(que, llm_client, generate_llm, 'You are a helpful assistant.')
        all_outputs.append([o_zero])
    return all_outputs


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='input question and context, generate answers')
    parser.add_argument('--dataset-name', type=str, default='chatdoctor')
    parser.add_argument('--attack-method', type=str, default='target')
    # For the above two parameters, only the following combination is valid
    # --dataset_name="chat" --attack_method="per"
    # --dataset_name="wiki" --attack_method="per"
    # --dataset_name="chatdoctor" --attack_method="target"
    # --dataset_name="chatdoctor" --attack_method="untarget"
    # --dataset_name="wiki_pii" --attack_method="target"
    # --dataset_name="wiki_pii" --attack_method="untarget"
    parser.add_argument('--k', type=int, default=1, help='context numbers')
    parser.add_argument('--protect-method', type=str,
                        choices=["sync",         # Our proposed method, synthetic data
                                 "agent2",       # Our proposed method, using 2 agents to make the generation less risk
                                 "para",         # paragraph, the baseline for comparison
                                 "ZeroGen",      # the baseline for comparison
                                 "attrPrompt",   # the baseline for comparison
                                 "ori",          # do not use any protect method
                                 "llm",          # do not use RAG
                                 ])
    parser.add_argument('--llm-generations', type=str, default='gpt-35-turbo', choices=['gpt-4', 'gpt-35-turbo', 'llama-3'])

    args = parser.parse_args()
    num_error = 0
    llm_generations = args.llm_generations
    folder = args.folder
    attack_method = args.attack_method
    dataset_name = args.dataset_name

    with open(f'questions/{attack_method}-{dataset_name}-question.json', 'r', encoding='utf-8') as f:
        question = json.load(f)
    if args.protect_method != 'llm':
        with open(f'contexts/{attack_method}-{dataset_name}-{args.protect_method}-context.json', 'r', encoding='utf-8') as f:
            context = json.load(f)
        context = [c[:args.k] for c in context]
        final_outputs = get_query_output_k(question, context, llm_generations)
    else:
        final_outputs = get_performance_output_k(question, llm_generations)
    print(f'Error num is {num_error}')
    with open(f'outputs/{attack_method}-{dataset_name}-{args.protect_method}-{llm_generations}-output.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(final_outputs))
