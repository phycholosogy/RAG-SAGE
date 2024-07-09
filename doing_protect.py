import os
import random
import re
import json
import argparse
from tqdm import tqdm
from openai import AzureOpenAI
from autogen import ConversableAgent, GroupChat, GroupChatManager
import spacy
import transformers
import torch
from transformers import AutoModelForCausalLM


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


def get_attributes_prompt(input_context, dataset):
    if dataset.find('chat') != -1:
        prompt = f"""
            Please summarize the key points from the following Doctor-Patient conversation:
    
    
            {input_context}
    
            Provide a summary for the Patient's information, including:
            [Attribute 1: Clear Symptom Description]
            [Attribute 2: Medical History]
            [Attribute 3: Current Concerns]  
            [Attribute 4: Recent Events]
            [Attribute 5: Specific Questions]
    
            Then, provide a summary for the Doctor's information, including:
            [Attribute 1: Clear Diagnosis or Assessment]
            [Attribute 2: Reassurance and Empathy]
            [Attribute 3: Treatment Options and Explanations]
            [Attribute 4: Follow-up and Next Steps]
            [Attribute 5: Education and Prevention]
    
            Please format your response as follows:
    
            Patient:
            - [Attribute 1: Clear Symptom Description]: 
            - [Attribute 2: Medical History]:
            - [Attribute 3: Current Concerns]:
            - [Attribute 4: Recent Events]:
            - [Attribute 5: Specific Questions]:
    
            Doctor:
            - [Attribute 1: Clear Diagnosis or Assessment]:
            - [Attribute 2: Reassurance and Empathy]:
            - [Attribute 3: Treatment Options and Explanations]:
            - [Attribute 4: Follow-up and Next Steps]:
            - [Attribute 5: Education and Prevention]:
    
            Please provide a concise summary for each attribute, capturing the most important information related to that attribute from the conversation.
            """
    elif dataset.find('wiki') != -1:
        prompt = f"""
            Please summarize the key points from the following wiki text:


            {input_context}

            Provide a summary the knowledge from the wiki text, including:
            [Attribute 1: Clear TOPIC or CENTRAL IDEA of the wiki text]
            [Attribute 2: Main details of the TOPIC or CENTRAL IDEA]
            [Attribute 3: Important facts, data, events, or viewpoints]

            Please format your response as follows:

            - [Attribute 1: Clear TOPIC or CENTRAL IDEA of the wiki text]:
            - [Attribute 2: Main details of the TOPIC or CENTRAL IDEA]:
            - [Attribute 3: Important facts, data, events, or viewpoints]:

            Please provide a concise summary for each attribute, capturing the most important information related to that attribute from the conversation. And remember to maintain logical order and accuracy.
            """
    else:
        prompt = 'prompt error'
    return prompt


def get_synthetic_prompt(input_attributes, dataset):
    if dataset.find('wiki') != -1:
        prompt = f"""Here is a summary of the key points:
    
        {input_attributes}
    
        Please generate a wiki text using the ALL key points provided. 
        The conversation should like a real-word wiki text.
        """
    elif dataset.find('chat') != -1:
        prompt = f"""Here is a summary of the key points:

        {input_attributes}

        Please generate a SINGLE-ROUND patient-doctor medical dialog using the ALL key points provided. 
        The conversation should like a real-word medical conversation and contain ONLY ONE question from the patient and ONE response from the doctor. The format should be as follows:

        Patient:[Patient's question contains ALL Patient's key points provided] 
        Doctor:[Doctor's response contains ALL Doctor's key points provided]

        Do not generate any additional rounds of dialog beyond the single question and response specified above."""
    else:
        prompt = 'prompt error'
    return prompt


def get_paraphrase_prompt(input_context, input_query):
    prompt = f"""Given the following context, extract the useful or important part of the Context.
    
    Remember, *DO NOT* edit the extracted parts of the context.
    
    > Context:
    > > >
    {input_context}
    > > >
    Extracted relevant parts:
    """
    return prompt


def get_synthetic_context(ori_contexts, dataset, attributes_llm='gpt-35-turbo', synthetic_llm='gpt-35-turbo'):
    """
    Generate a new synthesized context based on the input raw data
    :param:
        ori_context: [[C11, C12, ..., C1k], [C21, C22, ..., C2k], ..., [Cn1, Cn2, ..., Cnk]]
        attributes_llm: llm for attributes
        generation_llm: llm for generate synthetic data
    """
    attributes_llm_client = get_llm_client(attributes_llm)
    synthetic_llm_client = get_llm_client(synthetic_llm)
    all_attributes_con = []
    all_synthetic_con = []
    for ori_context in tqdm(ori_contexts, desc="generate synthetic context"):
        attributes_con = []
        synthetic_con = []
        for ori_con in ori_context:
            attributes_prompt = get_attributes_prompt(ori_con, dataset)
            attributes_context = get_llm_output(attributes_prompt, attributes_llm_client, attributes_llm, 'You are a helpful assistant.')
            synthetic_prompt = get_synthetic_prompt(attributes_context, dataset)
            synthetic_context = get_llm_output(synthetic_prompt, synthetic_llm_client, synthetic_llm, 'You are a helpful assistant.')
            attributes_con.append(attributes_context)
            synthetic_con.append(synthetic_context)
        all_attributes_con.append(attributes_con)
        all_synthetic_con.append(synthetic_con)
    return all_attributes_con, all_synthetic_con


def get_agent2_context(ori_contexts, sync_contexts):

    cathy = ConversableAgent(
        "cathy",
        system_message="You are a synthetic data generator and your role is to generate synthetic data based on provided feedback(words after SUGGESTIONS:) and to make sure the synthetic data is of high utility and privacy-preserving, you should put your generated data after the word 'GENERATED DATA:' ",
        llm_config={
            "YOU OWN CONFIGURATION HERE": 'xxx'
        },
        human_input_mode="NEVER",
        is_termination_msg=lambda msg: "thisissafe" in msg["content"].lower()
    )
    joe = ConversableAgent(
        "joe",
        system_message="""You are a privacy evaluation agent and your role is to provide comprehensive feedback on the synthetic data generated by the synthetic data generator. To be specific, you should analyze the synthetic data (the data after the word 'GENERATED DATA:') from the following aspects:

    1. Personally Identifiable Information (PII): Check if the synthetic data contains any PII, such as names, addresses, phone numbers, email addresses, or other information that can directly identify an individual. If found, suggest ways to remove or anonymize such information.

    2. Sensitive Attributes: Look for any sensitive attributes in the synthetic data, including but not limited to race, ethnicity, religion, political affiliation, sexual orientation, health status, or financial information. If present, provide suggestions on how to handle or obfuscate these attributes to mitigate potential privacy risks.

    3. Contextual Privacy: Evaluate if the synthetic data, when combined with other publicly available information, could potentially lead to the identification of individuals or reveal sensitive information about them. If such risks are identified, recommend strategies to mitigate these contextual privacy issues.

    4. Data Linkage: Assess if the synthetic data can be linked with other datasets to infer additional sensitive information about individuals. If linkage risks are found, suggest techniques such as data perturbation or aggregation to reduce these risks.

    5. Semantic Consistency: Ensure that the privacy-preserving transformations applied to the synthetic data maintain semantic consistency and do not introduce any unintended biases or inaccuracies. If inconsistencies are detected, provide feedback on how to maintain the balance between privacy protection and data utility.

    6. Original Data Recovery: Analyze the synthetic data to determine if it could potentially allow attackers to recover or reconstruct the original conversation data(word behind TRUE CONVERSATION:). If such vulnerabilities are identified, suggest methods to introduce additional randomness, noise, or perturbations to break direct correspondences between the synthetic data and the original conversation, making recovery attempts more difficult.

    Only if the generated data is completely safe and satisfies all the above privacy requirements and prevents the recovery of the original data, include the word 'THISISSAFE' anywhere in your response to signal the end of the evaluation process. Otherwise, provide detailed suggestions and guidance on how to improve the privacy aspects of the synthetic data(after the word "SUGGESTIONS:") and do not contain the word 'THISISSAFE' in your response.

    If the data is deemed safe, please also extract the safe synthetic data (the text after 'GENERATED CONVERSATION:') and return it in the following format:
    SAFE_DATA:
    [BEG]<safe_synthetic_data>[END]THISISSAFE

    Note that your job is only to assess the privacy level of generated data, you can answer either suggestions(SUGGESTIONS) or this data is safe(SAFE_DATA:
    [BEG]<safe_synthetic_data>[END]THISISSAFE), does not provide irrevenlent answers.

    """,
        llm_config={
            "YOU OWN CONFIGURATION HERE": 'xxx'
        },
        human_input_mode="NEVER",
    )
    joe.reset()
    cathy.reset()

    safe_count = 0
    safe_data_list = []
    unsafe_lst = []
    num_turns = []
    for i in tqdm(range(len(ori_contexts))):
        syn_conversation = sync_contexts[i]
        true_conversation = ori_contexts[i]
        safe_data_lst = []
        for j in range(len(syn_conversation)):
            syn_con = syn_conversation[j]
            true_con = true_conversation[j]
            message = f"Hi Joe, I will give you the real data(TRUE DATA) and synthetic data(GENERATED DATA), please help me assess and provide suggestions from the privacy level of TRUE DATA:{true_con}\n GENERATED DATA:{syn_con}"

            try:
                result = cathy.initiate_chat(joe, message=message, max_turns=5)
                safe_data_match = re.search(r'\[BEG\](.*?)\[END\]', result.chat_history[-1]['content'], re.DOTALL)
                num_turns.append(len(result.chat_history))
            except:
                result = 'No'
                safe_data_match = None
                num_turns.append(0)

            if safe_data_match:
                safe_count += 1
                safe_data = safe_data_match.group(1)
            else:
                if result == 'No':
                    safe_data = syn_con
                elif len(result.chat_history) >= 2:
                    safe_data = result.chat_history[-2]['content'][16:]
                else:
                    safe_data = syn_con
                unsafe_lst.append([i, j])
            safe_data_lst.append(safe_data)
            joe.reset()
            cathy.reset()
        safe_data_list.append(safe_data_lst)

    print(f'Number of safe data:{safe_count}')

    print(unsafe_lst)
    with open('sync-num.json', 'w') as f:
        f.write(json.dumps(num_turns))
    print(sum(num_turns)/len(num_turns))
    return safe_data_list


def get_paraphrase_context(ori_contexts, input_question, paraphrase_llm='gpt-35-turbo'):
    paraphrase_llm_client = get_llm_client(paraphrase_llm)
    all_paraphrase_con = []
    for i in tqdm(range(len(ori_contexts)), desc="generate paraphrase context"):
        paraphrase_con = []
        ori_context = ori_contexts[i]
        ques = input_question[i]
        for ori_con in ori_context:
            paraphrase_prompt = get_paraphrase_prompt(ori_con, ques)
            paraphrase_contexts = get_llm_output(paraphrase_prompt, paraphrase_llm_client, paraphrase_llm, 'You are a helpful assistant.')
            paraphrase_con.append(paraphrase_contexts)
        all_paraphrase_con.append(paraphrase_con)
    return all_paraphrase_con


def get_query_output(questions, contexts, generate_llm):
    llm_client = get_llm_client(generate_llm)
    all_outputs = []
    for i in tqdm(range(len(questions)), desc="generate final out"):
        final_con = '\n\n'.join(contexts[i])
        prompt = f"Context: {final_con}\nQuestion: {questions[i]}\nAnswer:"
        output = get_llm_output(prompt, llm_client, 'You are a helpful assistant.')
        all_outputs.append(output)
    return all_outputs


def rerun_error(dataset, atk_method, llm_name='gpt-35-turbo'):
    with open('error.json', 'r', encoding='utf-8') as f:
        error_context = json.load(f)
    with open(f'{atk_method}-{dataset}-sync-context.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    client = get_llm_client(llm_name)
    for item in error_context:
        synthetic_prompt = get_synthetic_prompt(item[0], dataset)
        synthetic_context = get_llm_output(synthetic_prompt, client, llm_name, 'You are a helpful assistant.')
        data[item[1]][item[2]] = synthetic_context
    with open(f'{atk_method}-{dataset}-sync-context.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(data))


def baseline_zero_gen(ori_contexts, num_qa, llm_baseline='gpt-35-turbo'):
    nlp = spacy.load("en_core_web_sm")
    zero_gen_llm_client = get_llm_client(llm_baseline)
    all_zero_gen_con = []
    random.shuffle(ori_contexts)
    all_new_qa = []
    for ori_all_context in tqdm(ori_contexts, desc="baseline-zero-gen生成synthetic context"):
        base_con = []
        for ori_con in ori_all_context:
            all_entity = list(nlp(ori_con).ents)
            random.shuffle(all_entity)

            for i in range(len(all_entity)):

                new_ans = all_entity[i]
                new_ques = get_llm_output(f'The context is: "{ori_con}"\n"{new_ans}" is the answer of the following question: "',
                                          zero_gen_llm_client, llm_baseline, 'You are a helpful assistant.')
                if new_ques is None:  # or new_ques.find("I'm sorry") != -1 or new_ques.find('there is no question') != -1:
                    continue
                new_ques = new_ques.strip('"')
                all_new_qa.append(f'question: {new_ques}\nanswer: {new_ans}')

    random.shuffle(all_new_qa)
    num_now = 0
    for ori_all_context in ori_contexts:
        base_con = []
        for _ in ori_all_context:

            base_con.append('\n\n'.join(all_new_qa[num_now*num_qa:(num_now+1)*num_qa]))
            num_now += 1
        all_zero_gen_con.append(base_con)
    return all_zero_gen_con


def baseline_attr_prompt(dataset, num_data=1000, llm_baseline='gpt-35-turbo'):
    all_prompt = []
    with open(f'contexts/attr_prompt_{dataset}.json', 'r') as f_attr:
        attr = json.load(f_attr)
    for i in range(num_data):
        all_att = []
        for j in range(len(attr)):
            all_att.append(random.choice(attr[j]))
        if dataset.find('chat') != -1:
            prompt = f"""Suppose you are a medical assistant, Please generate a conversation about {all_att[0]} following the requirements below:
            1. should include {all_att[1]}-class terms;
            2. should include {all_att[2]};
            3. should give {all_att[3]} as advice;
            4. should have characteristic {all_att[4]}."""
        else:
            prompt = f"""Suppose you are a writer for wikipedia, Please generate a wiki text about {all_att[0]} following the requirements below:
            1. should include part of {all_att[1]};
            2. should use {all_att[2]} to describe;
            3. should include {all_att[3]};
            4. should introduce {all_att[4]}."""
        all_prompt.append(prompt)
    attr_llm_client = get_llm_client(llm_baseline)
    all_ans = []
    for prompt in all_prompt:
        ans = get_llm_output(prompt, attr_llm_client, llm_baseline, 'You are a helpful assistant.')
        all_ans.append(ans)
    return all_ans


if __name__ == "__main__":

    os.environ['AZURE_OPENAI_API_KEY'] = "YOUR API KEY"
    parser = argparse.ArgumentParser(description='input question and origin-context, to generate protect context')
    parser.add_argument('--protect-method', type=str,
                        choices=["sync",         # Our proposed method, synthetic data
                                 "agent2",       # Our proposed method, using 2 agents to make the generation less risk
                                 "para",         # paragraph, the baseline for comparison
                                 "ZeroGen",      # the baseline for comparison
                                 "attrPrompt"    # the baseline for comparison
                                 ])
    parser.add_argument('--dataset-name', type=str, default='chatdoctor')
    parser.add_argument('--attack-method', type=str, default='target')
    # For the above two parameters, only the following combination is valid
    # --dataset_name="chat" --attack_method="per"
    # --dataset_name="wiki" --attack_method="per"
    # --dataset_name="chatdoctor" --attack_method="target"
    # --dataset_name="chatdoctor" --attack_method="untarget"
    # --dataset_name="wiki_pii" --attack_method="target"
    # --dataset_name="wiki_pii" --attack_method="untarget"
    parser.add_argument('--attributes-llm', type=str, default='gpt-35-turbo', choices=['gpt-4', 'gpt-35-turbo', 'llama-3'],
                        help='the llm to generate attributes of context')
    parser.add_argument('--synthetic-llm', type=str, default='gpt-35-turbo', choices=['gpt-4', 'gpt-35-turbo', 'llama-3'],
                        help='the llm to generate synthetic data by using attributes')
    parser.add_argument('--paraphrase-llm', type=str, default='gpt-35-turbo', choices=['gpt-4', 'gpt-35-turbo'],
                        help='the llm to generate paraphrase data')
    parser.add_argument('--agents-llm', type=str, default='gpt-35-turbo', choices=['gpt-4', 'gpt-35-turbo'],
                        help='the llm to generate agent2 data')
    parser.add_argument('--baseline-llm', type=str, default='gpt-35-turbo', choices=['gpt-4', 'gpt-35-turbo'],
                        help='the llm used for baseline')
    parser.add_argument('--k', type=int, default=1, help='number of contexts')
    args = parser.parse_args()
    protect_method = args.protect_method
    dataset_name = args.dataset_name
    attack_method = args.attack_method
    folder = args.folder
    num_error = 0
    # Getting question and context
    with open(f'contexts/{attack_method}-{dataset_name}-ori-context.json', 'r', encoding='utf-8') as f:
        ori_context = json.load(f)
    ori_context = [con[:args.k] for con in ori_context]
    with open(f'questions/{attack_method}-{dataset_name}-question.json', 'r', encoding='utf-8') as f:
        question = json.load(f)

    print(f'Test number is {len(ori_context)}, number of context is {len(ori_context[0])}')

    # getting synthetic data
    if protect_method == 'sync':
        attributes_contexts, synthetic_contexts = get_synthetic_context(ori_context, dataset_name, args.attributes_llm, args.synthetic_llm)
        with open(f'contexts/{attack_method}-{dataset_name}-attributes_context.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(attributes_contexts))
        with open(f'contexts/{attack_method}-{dataset_name}-sync-context.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(synthetic_contexts))
    # getting paraphrase data
    elif protect_method == 'para':
        paraphrase_context = get_paraphrase_context(ori_context, question, args.paraphrase_llm)
        with open(f'contexts/{attack_method}-{dataset_name}-para-context.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(paraphrase_context))
    # getting agent data
    elif protect_method == 'agent2':
        with open(f'contexts/{attack_method}-{dataset_name}-sync-context.json', 'r', encoding='utf-8') as f:
            sync_context = json.load(f)
        sync_context = [con[:args.k] for con in sync_context]
        agent_context = get_agent2_context(ori_context, sync_context)
        with open(f'contexts/{attack_method}-{dataset_name}-{protect_method}-context.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(agent_context))
    elif protect_method == 'ZeroGen':
        baseline_context = baseline_zero_gen(ori_context, 20, args.baseline_llm)
        with open(f'contexts/{attack_method}-{dataset_name}-{protect_method}-context.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(baseline_context))
    elif protect_method == 'attrPrompt':
        baseline_context = baseline_attr_prompt(dataset_name)
        with open(f'contexts/{attack_method}-{dataset_name}-{protect_method}-context.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(baseline_context))
