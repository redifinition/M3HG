

import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score

from utils.global_variables import EMOTION_MAPPING

def get_pred_result(em_logits, couples_pred, emo_caus_pos, uttr_mask):
    em_logits = np.asarray(em_logits.cpu().detach())
    # 将 em_logits 应用 softmax 转换为概率分布
    em_probs = np.exp(em_logits) / np.sum(np.exp(em_logits), axis=-1, keepdims=True)

    # 找到每个样本的最大概率对应的索引，即预测的情绪类别
    predicted_emotions = np.argmax(em_probs, axis=-1) + 1 # 加1是为了和原始的标签对齐

    # 情绪原因对预测
    # couples_pred = np.asarray(couples_pred)
    # 情绪原因对预测结果应用 sigmoid 函数转换为概率分布
    # 将 logits 或概率转换为二分类标签
    threshold = 0.5
    pred_labels = (torch.sigmoid(couples_pred) > threshold).float()

    pred_ec_pairs = []
    batch_size = couples_pred.size(0)
    for idx in range(batch_size):
        pred_ec_pairs_i = []
        for i, (pred_label, emo_caus) in enumerate(zip(pred_labels[idx], emo_caus_pos)):
            emo_idx, caus_idx = emo_caus
            if emo_idx <= uttr_mask[idx].sum().item() and caus_idx <= uttr_mask[idx].sum().item() and pred_label: # 如果情绪原因对的索引在有效的话语长度内且预测为真
                if predicted_emotions[idx][emo_idx-1] != 7: # 非中性情绪
                    pred_ec_pairs_i.append(emo_caus)
        pred_ec_pairs.append(pred_ec_pairs_i)
                
    return predicted_emotions, pred_ec_pairs

def calc_eval_result(emotion_pred_all, emotion_list_all, couples_pred_all, ec_pair_all):
    a = 1
    eval_result = {}
    # 计算情绪分类的准确率
    # 在测试时，因为进行了截断，所以需要根据实际的情绪列表进行填充，默认填充为中性
    # 确保 emotion_pred_all 中的每个列表长度与 emotion_list_all 中的对应列表长度相同
    for i in range(len(emotion_pred_all)):
        # 计算需要填充的元素数量
        extra_length = len(emotion_list_all[i]) - len(emotion_pred_all[i])
        # 如果 emotion_pred_all 的当前列表比 emotion_list_all 的短，则进行填充
        if extra_length > 0:
            emotion_pred_all[i].extend([7] * extra_length)
    # 将数据展平成一维列表
    flat_pred_emo = [pred for sublist in emotion_pred_all for pred in sublist]
    flat_true_emo = [true for sublist in emotion_list_all for true in sublist]

    # 转换为 NumPy 数组
    flat_pred_emo = np.array(flat_pred_emo)
    flat_true_emo = np.array(flat_true_emo)

    # 找出有效位置的掩码
    valid_mask = flat_true_emo != 0

    # 只保留有效位置的数据
    valid_pred_emo = flat_pred_emo[valid_mask]
    valid_true_emo = flat_true_emo[valid_mask]

    # 计算 P, R, F1
    precision_emo = precision_score(valid_true_emo, valid_pred_emo, average='macro')
    recall_emo = recall_score(valid_true_emo, valid_pred_emo, average='macro')
    f1_emo = f1_score(valid_true_emo, valid_pred_emo, average='macro')
    eval_result['precision_emo'] = precision_emo
    eval_result['recall_emo'] = recall_emo
    eval_result['f1_emo'] = f1_emo

    ec_pair_all = [arr.tolist() for arr in ec_pair_all]
    # 计算情绪原因对提取结果
    precision_ec, recall_ec, f1_ec = calc_f1_by_list(ec_pair_all, couples_pred_all)
    eval_result['precision_ec'] = precision_ec
    eval_result['recall_ec'] = recall_ec
    eval_result['f1_ec'] = f1_ec

    # 三重提取结果
    triples_all_true = get_triples(emotion_list_all, ec_pair_all)
    triples_all_pred = get_triples(emotion_pred_all, couples_pred_all)
    precision_triple, recall_triple, f1_triple = calc_f1_by_list(triples_all_true, triples_all_pred)
    eval_result['precision_triple'] = precision_triple
    eval_result['recall_triple'] = recall_triple
    eval_result['f1_triple'] = f1_triple

    # evg三重提取结果
    # 过滤掉每个对话中的 neutral 类别
    filtered_pred_triple = [[triple for triple in triples if triple[0] != EMOTION_MAPPING['ECF']['neutral']] for triples in triples_all_pred]
    filtered_true_triple = [[triple for triple in triples if triple[0] != EMOTION_MAPPING['ECF']['neutral']] for triples in triples_all_true]
    # 去除neutral的emotion index列表
    emotions_6 = list(set(EMOTION_MAPPING['ECF'].values()))[:-1]
    precision_per_class, recall_per_class, f1_per_class, weighted_avg_f1_6, weighted_avg_p_6, weighted_avg_r_6 = calc_weighted_avg_f1_by_list(filtered_true_triple, filtered_pred_triple, emotions_6)
    eval_result['precision_per_class'] = precision_per_class
    eval_result['recall_per_class'] = recall_per_class
    eval_result['f1_per_class'] = f1_per_class
    eval_result['weighted_avg_f1_6'] = weighted_avg_f1_6
    eval_result['weighted_avg_p_6'] = weighted_avg_p_6
    eval_result['weighted_avg_r_6'] = weighted_avg_r_6

    # 去除 'disgust', 'fear' 和 'neutral' 的键值对
    emotions_to_remove = {'disgust', 'fear', 'neutral'}
    filtered_emotion_mapping = {k: v for k, v in EMOTION_MAPPING['ECF'].items() if k not in emotions_to_remove}
    emotions_4 = list(set(filtered_emotion_mapping.values()))
    _, _, _, weighted_avg_f1_4, weighted_avg_p_4, weighted_avg_r_4 = calc_weighted_avg_f1_by_list(filtered_true_triple, filtered_pred_triple, emotions_4)
    eval_result['weighted_avg_f1_4'] = weighted_avg_f1_4
    eval_result['weighted_avg_p_4'] = weighted_avg_p_4
    eval_result['weighted_avg_r_4'] = weighted_avg_r_4

    return eval_result


def calc_weighted_avg_f1_by_list(triples_all_true, triples_all_pred, emotions):
    precision_per_class = []
    recall_per_class = []
    f1_per_class = []
    support_per_class = []

    for emotion in emotions:
        emotion_tp = 0
        emotion_fp = 0
        emotion_fn = 0
        emotion_support = 0
        
        for true_triples, pred_triples in zip(triples_all_true, triples_all_pred):
            true_emo_triples = [triple for triple in true_triples if triple[0] == emotion]
            pred_emo_triples = [triple for triple in pred_triples if triple[0] == emotion]

            TP = len([triple for triple in pred_emo_triples if triple in true_emo_triples])
            FP = len([triple for triple in pred_emo_triples if triple not in true_emo_triples])
            FN = len([triple for triple in true_emo_triples if triple not in pred_emo_triples])
            
            emotion_tp += TP
            emotion_fp += FP
            emotion_fn += FN
            emotion_support += len(true_emo_triples)

        precision = emotion_tp / (emotion_tp + emotion_fp) if (emotion_tp + emotion_fp) > 0 else 0
        recall = emotion_tp / (emotion_fn + emotion_tp) if (emotion_fn + emotion_tp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        precision_per_class.append(precision)
        recall_per_class.append(recall)
        f1_per_class.append(f1)
        support_per_class.append(emotion_support)

    weighted_avg_f1 = np.sum(np.array(f1_per_class) * np.array(support_per_class)) / np.sum(support_per_class)
    weighted_avg_p = np.sum(np.array(precision_per_class) * np.array(support_per_class)) / np.sum(support_per_class)
    weighted_avg_r = np.sum(np.array(recall_per_class) * np.array(support_per_class)) / np.sum(support_per_class)

    return precision_per_class, recall_per_class, f1_per_class, weighted_avg_f1, weighted_avg_p, weighted_avg_r
    


def calc_f1_by_list(true_list, pred_list):
    TP = 0
    FP = 0
    FN = 0

    for true_pairs, pred_pairs in zip(true_list, pred_list): # 对对话中的每一对进行比较
        
        for pair in pred_pairs:
            if pair in true_pairs:
                TP += 1
            else:
                FP += 1
        for pair in true_pairs:
            if pair not in pred_pairs:
                FN += 1

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1


def get_triples(emotion_list_all, ec_pair_all):
    uttr_len = len(emotion_list_all)
    triples_all = []
    for i in range(uttr_len):
        ec_pairs = ec_pair_all[i]
        triples = []
        for pair in ec_pairs:
            emo_idx, caus_idx = pair
            emotion = emotion_list_all[i][emo_idx-1]-1 # 减1是因为预测的索引是从1开始的,并且预测值是从1-7
            triple = [emotion, emo_idx, caus_idx]
            triples.append(triple)
        triples_all.append(triples)
    return triples_all
            


