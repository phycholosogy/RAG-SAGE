from retrieval_database import load_retrieval_database_from_parameter
import os
import json
from tqdm import tqdm

path = './Inputs&Outputs/sync'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
file_name = 'untarget-chat'
with open(f'{path}/{file_name}-question.json', 'r', encoding='utf-8') as f:
    question = json.load(f)

database = load_retrieval_database_from_parameter(["chatdoctor"], 'bge-large-en-v1.5', 1024)
all_context_train = []
all_error = []
for i in tqdm(range(len(question))):
    que = question[i]
    try:
        con = database.similarity_search(que, k=5)
        all_con = [c.page_content for c in con]
        all_context_train.append(all_con)
    except:
        all_error.append(i)
print(all_error)

with open(f'{path}/{file_name}-ori-context.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(all_context_train))
