'''
zhihu:
https://zhuanlan.zhihu.com/p/634322783?utm_id=0
https://zhuanlan.zhihu.com/p/648412136

hf datasets:
alpaca: https://huggingface.co/datasets/tatsu-lab/alpaca
alpaca-gpt4: https://huggingface.co/datasets/vicgalle/alpaca-gpt4
guanaco: https://huggingface.co/datasets/JosephusCheung/GuanacoDataset
gpteacher: https://huggingface.co/datasets/teknium/GPTeacher-General-Instruct
sharegpt: https://huggingface.co/datasets/shibing624/sharegpt_gpt4
instructwild: https://github.com/XueFuzhao/InstructionWild
belle: https://github.com/LianjiaTech/BELLE
hc3: https://huggingface.co/datasets/Hello-SimpleAI/HC3
lamini: https://huggingface.co/datasets/MBZUAI/LaMini-instruction

Please run huggingface-cli login before running this script.

'''

import json
import pandas as pd
from datasets import load_dataset

LC = lambda x: x.shape[0]//1000

alpaca = load_dataset("tatsu-lab/alpaca")
alpaca = alpaca['train'].to_pandas()
alpaca = alpaca[['instruction', 'input', 'output']]
alpaca = alpaca[~alpaca.output.isna()]
alpaca['jsonl'] = alpaca.apply(lambda x: json.dumps(dict(instruction=x['instruction'], input=x['input'], output=x['output'])), axis=1)
alpaca = alpaca[['jsonl']]
alpaca.to_csv(f'open-alpaca-data-{LC(alpaca)}K.jsonl', sep="\t", index=False, header=None, quoting=3, escapechar=None,)

alpaca_gpt4 = load_dataset('vicgalle/alpaca-gpt4')
alpaca_gpt4 = alpaca_gpt4['train'].to_pandas()
alpaca_gpt4 = alpaca_gpt4[['instruction', 'input', 'output']]
alpaca_gpt4 = alpaca_gpt4[~alpaca_gpt4.output.isna()]
alpaca_gpt4['jsonl'] = alpaca_gpt4.apply(lambda x: json.dumps(dict(instruction=x['instruction'], input=x['input'], output=x['output'])), axis=1)
alpaca_gpt4 = alpaca_gpt4[['jsonl']]
alpaca_gpt4.to_csv(f'open-alpaca-gpt4-data-{LC(alpaca_gpt4)}K.jsonl', sep="\t", index=False, header=None, quoting=3, escapechar=None,)

guanaco = load_dataset('JosephusCheung/GuanacoDataset')
guanaco = guanaco['train'].to_pandas()
guanaco = guanaco[['instruction', 'input', 'output']]
guanaco = guanaco[~guanaco.output.isna()]
guanaco['jsonl'] = guanaco.apply(lambda x: json.dumps(dict(instruction=x['instruction'], input=x['input'], output=x['output'])), axis=1)
guanaco = guanaco[['jsonl']]
guanaco.to_csv(f'open-guanaco-data-{LC(guanaco)}K.jsonl', sep="\t", index=False, header=None, quoting=3, escapechar=None,)

instructwild = load_dataset('fuliucansheng/InstructionWild', 'en')
instructwild = instructwild['train'].to_pandas()
instructwild = instructwild[['instruction', 'input', 'output']]
instructwild = instructwild[~instructwild.output.isna()]
instructwild['jsonl'] = instructwild.apply(lambda x: json.dumps(dict(instruction=x['instruction'], input=x['input'], output=x['output'])), axis=1)
instructwild = instructwild[['jsonl']]
instructwild.to_csv(f'open-instructwild-data-{LC(instructwild)}K.jsonl', sep="\t", index=False, header=None, quoting=3, escapechar=None,)

gpteacher = load_dataset('teknium/GPTeacher-General-Instruct')
gpteacher = gpteacher['train'].to_pandas()
gpteacher['output'] = gpteacher['response']
gpteacher = gpteacher[['instruction', 'input', 'output']]
gpteacher = gpteacher[~gpteacher.output.isna()]
gpteacher['jsonl'] = gpteacher.apply(lambda x: json.dumps(dict(instruction=x['instruction'], input=x['input'], output=x['output'])), axis=1)
gpteacher = gpteacher[['jsonl']]
gpteacher.to_csv(f'open-gpteacher-data-{LC(gpteacher)}K.jsonl', sep="\t", index=False, header=None, quoting=3, escapechar=None,)

hc3 = load_dataset('Hello-SimpleAI/HC3', 'all')
hc3 = hc3['train'].to_pandas()
hc3['instruction'] = hc3['question']
hc3['input'] = ''
hc3['human_output'] = hc3['human_answers'].map(lambda x: '' if len(x) == 0 else x[0])
hc3['chatgpt_output'] = hc3['chatgpt_answers'].map(lambda x: '' if len(x) == 0 else x[0])
hc3['jsonl1'] = hc3[['instruction', 'input', 'human_output']].apply(lambda x: json.dumps(dict(instruction=x['instruction'], input=x['input'], output=x['human_output'])), axis=1)
hc3['jsonl2'] = hc3[['instruction', 'input', 'chatgpt_output']].apply(lambda x: json.dumps(dict(instruction=x['instruction'], input=x['input'], output=x['chatgpt_output'])), axis=1)
hc31 = hc3[~(hc3.human_output.isna() | hc3.human_output == '')]
hc32 = hc3[~(hc3.chatgpt_output.isna() | hc3.chatgpt_output == '')]
hc31[['jsonl1']].to_csv(f'open-hc3-human-data-{LC(hc31)}K.jsonl', sep="\t", index=False, header=None, quoting=3, escapechar=None,)
hc32[['jsonl2']].to_csv(f'open-hc3-chatgpt-data-{LC(hc32)}K.jsonl', sep="\t", index=False, header=None, quoting=3, escapechar=None,)

lamini = load_dataset('MBZUAI/LaMini-instruction')
lamini = lamini['train'].to_pandas()
lamini['input'] = ''
lamini['output'] = lamini['response']
lamini = lamini[['instruction', 'input', 'output']]
lamini = lamini[~lamini.output.isna()]
lamini['jsonl'] = lamini.apply(lambda x: json.dumps(dict(instruction=x['instruction'], input=x['input'], output=x['output'])), axis=1)
lamini = lamini[['jsonl']]
lamini.to_csv(f'open-lamini-data-{LC(lamini)}K.jsonl', sep="\t", index=False, header=None, quoting=3, escapechar=None,)
