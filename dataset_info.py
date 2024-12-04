import json

with open('data/ECF/all_data.json','r') as f:
    data = json.load(f)

speaker_set = set()
# ECF数据集的说话人数量
for conversation in data:
    for utterance in conversation['conversation']:
        speaker = utterance['speaker']
        speaker_set.add(speaker)
print(len(speaker_set)) # 312