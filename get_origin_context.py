from retrieval_database import load_retrieval_database_from_parameter
import json
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Get origin context by retrieval database')
parser.add_argument('--dataset-name', type=str, required=True, help='dataset name')
parser.add_argument('--attack-method', type=str, required=True, help='attacking method or performance')
args = parser.parse_args()
"""
You can run this file by running following codes:
export CUDA_VISIBLE_DEVICES=1
python get_origin_context.py --dataset_name="chat" --attack_method="per"
You can also run:
python get_origin_context.py --dataset_name="wiki" --attack_method="per"
python get_origin_context.py --dataset_name="chatdoctor" --attack_method="target"
python get_origin_context.py --dataset_name="chatdoctor" --attack_method="untarget"
python get_origin_context.py --dataset_name="wiki_pii" --attack_method="target"
python get_origin_context.py --dataset_name="wiki_pii" --attack_method="untarget"
"""
with open(f'./questions/{args.attack_method}-{args.dataset_name}-question.json', 'r', encoding='utf-8') as f:
    question = json.load(f)
if args.dataset_name == 'chatdoctor':
    dataset = 'chat'
else:
    dataset = args.dataset_name

database = load_retrieval_database_from_parameter([dataset], 'bge-large-en-v1.5', 1024)
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

with open(f'./contexts/{args.attack_method}-{args.dataset_name}-ori-context.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(all_context_train))
