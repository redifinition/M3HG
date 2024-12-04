import json
import argparse
import logging

import numpy as np

from utils.eval_func import calc_eval_result
from utils.global_variables import EMOTION_MAPPING


def main():
    parser = argparse.ArgumentParser(description='Cal results from json')
    parser.add_argument('--pred_json_file', default='results/GPT-4o-MECAD/TEC_pred.json', help='the path of the predicted data files')
    parser.add_argument('--gt_json_file', default='data/MECAD/test_data_pair.json', help='the path of the ground truth data files')
    args = parser.parse_args()

    # 读取JSON文件
    with open(args.gt_json_file, 'r', encoding='utf-8') as file:
        gt_data = json.load(file)

    with open(args.pred_json_file, 'r', encoding='utf-8') as file:
        pred_data = json.load(file)

    emotion_list_all = [] # 情绪ground truth列表
    emotion_pred_all = []
    ec_pair_all = [] # 情绪原因对ground truth列表
    couples_pred_all = []
    for conversation_id, conversation_data in gt_data.items():
        emotion_list = []
        couples_pred = []
        ec_pair = []
        emotion_pred = []
        for utterance_data in conversation_data[0]:
            emotion_id = EMOTION_MAPPING["MECAD"][utterance_data['emotion']]
            if utterance_data.get('expanded emotion cause evidence') is not None:
                for c_idx in utterance_data['expanded emotion cause evidence']:
                    ec_pair.append((utterance_data['turn'], c_idx))
            emotion_list.append(emotion_id)
        for pred_conversation_data in pred_data:
            if pred_conversation_data['conversation_ID'] == conversation_id:
                for utterance_data in pred_conversation_data['conversation']:
                    if utterance_data.get('emotion') is not None and EMOTION_MAPPING["ECF"].get(utterance_data['emotion']) is not None:
                        emotion_id = EMOTION_MAPPING["ECF"][utterance_data['emotion']]
                        emotion_pred.append(emotion_id)
                for pred_couples in pred_conversation_data['emotion-cause_pairs']:
                   e_idx = int(pred_couples[0].split('_')[0])
                   import re
                   if len(re.findall(r'\d+', pred_couples[-1])) == 0:
                       continue
                   c_idx = int(re.findall(r'\d+', pred_couples[-1])[0])
                   couples_pred.append([e_idx, c_idx])
        emotion_list_all.append(emotion_list)
        ec_pair_all.append(np.array(ec_pair))
        emotion_pred_all.append(emotion_pred)
        couples_pred_all.append(couples_pred)
    eval_results = calc_eval_result(emotion_pred_all, emotion_list_all, couples_pred_all, ec_pair_all)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(filename)s[line:%(lineno)d] - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.INFO)
    logger = logging.getLogger(__name__)
    for key in sorted(eval_results.keys()):
        logger.info("{}: {}".format(key, str(eval_results[key])))  


if __name__ == "__main__":
    main()