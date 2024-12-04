
import csv
import json
import os
import pickle
import numpy as np
import torch
from torch.utils.data import IterableDataset
import logging
import dgl
from tqdm import tqdm
from utils.draw_graph import visualize_dgl_graph
from utils.extract_audio_features_from_wav import extract_opensmile_audio_features_from_wav
from utils.global_variables import EMOTION_MAPPING, GRAPH_CONFIG_T
from moviepy.editor import VideoFileClip
import torchaudio
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import arff
from pydub import AudioSegment

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(filename)s[line:%(lineno)d] - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_valid_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines
        
def get_ec_pairs(conversation_data:list):
    """
    获取情绪-原因对列表。
    参数：
    conversation_data (list)：对话数据列表。
    返回：
    ec_pairs (list)：情绪-原因对列表。
    """
    ec_pairs = []
    for turn in conversation_data:
        if 'expanded emotion cause evidence' in turn.keys():
            for cause_index in turn['expanded emotion cause evidence']:
                ec_pairs.append([turn['turn'], cause_index]) 
    return ec_pairs

class InputExample(object):
    """A single set of features of data."""
    def __init__(self, guid, conversation_id, context, emotions, emotion_categorys, video_dir_list, ec_pairs, uttr_len, cause_list, speaker_ids,meld_id_list):
        self.guid = guid
        self.conversation_id = conversation_id
        self.conversation_length = len(context)
        self.utterance_list = context
        self.emotion_list = emotions
        self.emotion_category_list = emotion_categorys
        self.video_dir_list = video_dir_list
        self.ec_pairs = ec_pairs
        self.uttr_len = uttr_len
        self.cause_list = cause_list
        self.speaker_ids = speaker_ids
        self.meld_id_list = meld_id_list

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,conversation_id, input_ids, input_mask, segment_ids, ec_pairs, speaker_ids, mention_ids, emotion_ids, 
                 emotion_list, uttr_indices, uttr_len, cause_list, uttr_speaker_ids, uttr_idx):
        self.conversation_id = conversation_id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.ec_pairs = ec_pairs
        self.speaker_ids = speaker_ids
        self.mention_ids = mention_ids
        self.emotion_ids = emotion_ids
        self.emotion_list = emotion_list
        self.uttr_indices = uttr_indices
        self.uttr_len = uttr_len
        self.cause_list = cause_list
        self.uttr_speaker_ids = uttr_speaker_ids
        self.uttr_idx = uttr_idx
      
class MECPECProcessor(DataProcessor):
    def __init__(self, input_dir, max_seq_length, data_name):
        self.dataset = [[], [], []]
        self.emotion_mapping = EMOTION_MAPPING[data_name]
        speakers = {}
        for idx, data_type in enumerate(['train', 'valid', 'test']):
            input_file = os.path.join(input_dir, '{}.json'.format(data_type+'_data_pair'))
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for conversation_id, conversation in data.items():
                context = []
                emotions = []
                emotion_categorys = []
                video_dir_list = []
                uttr_len = 0
                ec_pairs = get_ec_pairs(conversation[0])
                # 每个话语的说话人id
                speaker_ids = []
                meld_id_list = []
                for ut in conversation[0]:
                    speaker_id = speakers.get(ut['speaker'])
                    if ut.get('meld_idx') is not None:
                        meld_id_list.append(ut['meld_idx'])
                    if speaker_id is None:
                        speaker_id = "S{}".format(len(speakers) + 1)
                        speakers[ut['speaker']] = speaker_id
                    info = speaker_id + " " + ut['utterance'].lower()
                    speaker_ids.append(speaker_id)
                    context.append(info)
                    emotions.append(int(self.emotion_mapping[ut['emotion']])+1) # 这里的emotion标签+1
                    emotion_categorys.append(ut['emotion'])
                    video_dir_list.append(ut['video'])
                    uttr_len += 1
                    assert uttr_len == len(context), "{} \n {}".format(uttr_len, context)

                cause_list = self._generate_cause_list(emotions, ec_pairs)

                datasample = {
                    'conversation_id': conversation_id,
                    'conversation_length': len(context),
                    'utterance_list': context,
                    'emotion_list': emotions,   
                    'emotion_category_list': emotion_categorys,
                    'video_dir_list': video_dir_list,
                    'ec_pairs': ec_pairs, # 从1开始编号
                    'uttr_len': uttr_len,
                    'cause_list': cause_list,
                    'speaker_ids': speaker_ids,
                    'meld_id_list': meld_id_list
                }
                self.dataset[idx].append(datasample)    
        logger.info(
            "Train set:{} \t validation set:{} \t test set:{}.".format(len(self.dataset[0]), len(self.dataset[1]),
                                                                       len(self.dataset[2])))
        
    def _generate_cause_list(self, emotion_list, emotion_reasons):
        # 创建与 emotion_list 形状相同的 cause_list 并初始化为 -1
        cause_list = [0]*len(emotion_list)
        for ec_pair in emotion_reasons:
            assert ec_pair[1] <= len(cause_list)
            cause_list[ec_pair[1]-1] = 1
        return cause_list
    
    def _create_examples(self, data, data_type):
        examples = []
        for (i,d) in enumerate(data):
            guid = "{}-{}".format(data_type, i)
            # context = ' '.join(d['utterance_list'])
            examples.append(InputExample(guid=guid, conversation_id=d['conversation_id'],
                                         context=d['utterance_list'],
                                         emotions=d['emotion_list'],
                                         emotion_categorys=d['emotion_category_list'],
                                         video_dir_list=d['video_dir_list'],
                                         ec_pairs=d['ec_pairs'],
                                         uttr_len= d['uttr_len'],
                                         cause_list=d['cause_list'],
                                         speaker_ids=d['speaker_ids'],
                                         meld_id_list=d['meld_id_list']))
        return examples
            
    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        return self._create_examples(self.dataset[0], "train")
    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        return self._create_examples(self.dataset[1], "valid")
    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        return self._create_examples(self.dataset[2], "test")

class MECPECDataset(IterableDataset):
    def __init__(self, input_dir, saved_file, max_seq_length, max_speaker_num, tokenizer, audio_features_path, video_features_pkl_path, visual_model = None, video_processor = None, video_path=None, encoder_type="BERT",
                 data_name="ECF", data_type = 'train', K =1 ):
        super(MECPECDataset, self).__init__()
        self.data = None
        self.max_seq_length = max_seq_length


        logger.info("Reading data from directory {}.".format(input_dir))
        if os.path.exists(saved_file):
            with open(saved_file, 'rb') as f:
                self.data = pickle.load(f)
            logger.info("Loading feature data from {}.".format(saved_file))
        else:
            self.data = []
            processor = MECPECProcessor(input_dir, max_seq_length, data_name)

            if 'train' in saved_file:
                examples = processor.get_train_examples(saved_file)
            elif 'dev' in saved_file:
                examples = processor.get_dev_examples(saved_file)
            elif 'test' in saved_file:
                examples = processor.get_test_examples(saved_file)
            else:
                logging.error("Invalid output file:{}".format(saved_file))
            logger.info("{} examples are constructed.".format(len(examples)))
            features = convert_examples_to_features(examples, tokenizer, max_seq_length, max_speaker_num, encoder_type, data_type)
            
            if audio_features_path is not None:
                def normalize(x):
                    x1 = x[1:,:]
                    min_x = np.min(x1, axis=0, keepdims=True)
                    max_x = np.max(x1, axis=0, keepdims=True)
                    x1 = (x1-min_x)/(max_x-min_x+1e-8)
                    x[1:,:] = x1
                    return x
                # audio_data = normalize(np.load(audio_features_path, allow_pickle=True))
                with open(audio_features_path, 'rb') as file:  # 'rb' 表示以二进制读取模式打开文件
                    audio_data = pickle.load(file)
                with open(video_features_pkl_path, 'rb') as file:
                    video_data = pickle.load(file)
                v_id_map = eval(str(np.load('data/ECF/video_id_mapping.npy', allow_pickle=True)))
            for idx, f in enumerate(tqdm(features)):
                uttr_speaker_ids = f[0].uttr_speaker_ids
                # 图的构建
                if f[0].conversation_id == '10':
                    graph = self._build_conversation_graph(uttr_speaker_ids, K, True)
                else:
                    graph = self._build_conversation_graph(uttr_speaker_ids, K, False)
                # 音频模态处理
                audio_features, video_features = None, None
                if audio_features_path is not None:
                    # audio_features = self.get_audio_features(f[0].uttr_idx, audio_data, v_id_map)
                    audio_features = self.get_audio_features(examples[idx].meld_id_list, audio_data)
                if video_processor is not None:
                    
                    # video_features = prepare_video_features(video_processor, visual_model, video_path, examples[idx].video_dir_list, data_type, num_frames = 16)
                    # video_features = self.get_video_features(f[0].uttr_idx, video_data, v_id_map)
                    video_features = self.get_video_features(examples[idx].meld_id_list, video_data)
                self.data.append({
                    'conversation_id': f[0].conversation_id,
                    'input_id': np.array(f[0].input_ids),
                    'mention_id': np.array(f[0].mention_ids),
                    'segment_id': np.array(f[0].segment_ids),
                    'speaker_id': np.array(f[0].speaker_ids),
                    'input_mask': np.array(f[0].input_mask),
                    'turn_mask': mention2mask(np.array(f[0].mention_ids)),
                    'emotion_id': np.array(f[0].emotion_ids),
                    'ec_pairs': np.array(f[0].ec_pairs),
                    'emotion_list': f[0].emotion_list,
                    'uttr_indices': f[0].uttr_indices,
                    'uttr_len': f[0].uttr_len,
                    'cause_list': f[0].cause_list,
                    'graph': graph,
                    'audio_features': audio_features,
                    'video_features': video_features
                })

                if idx < 2:
                    logger.info("-------- Input Feature --------")
                    logger.info("tokens: %s" % graph)
        with open(saved_file, mode='wb') as f:
            pickle.dump(self.data, f)
        logger.info("Finish reading {} and save preprocessed data to {}.".format(input_dir, saved_file))
    
    # 读取 .pkl 文件的函数
    def _load_pkl_file(self, file_path):
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data


    # def get_audio_features(self, uttr_idx, audio_data, v_id_map):
    #     audio_features = []
    #     for video_idx in uttr_idx:
    #         idx = v_id_map[video_idx]
    #         audio_features.append(audio_data[idx])
    #     return audio_features

    def get_audio_features(self, meld_id_list, audio_data):
        audio_features = []
        for meld_id in meld_id_list:
            audio_features.append(audio_data[meld_id['conversation_id']][meld_id['utterance_idx']])
        
        return audio_features
    
    def get_video_features(self, meld_id_list, video_data):
        video_features = []
        for meld_id in meld_id_list:
            video_features.append(video_data[meld_id['conversation_id']][meld_id['utterance_idx']])
        return video_features


    def _build_conversation_graph(self, uttr_speaker_ids, K, visualize=False):
        # 计算话语的数量
        num_utterances = len(uttr_speaker_ids)

        # 创建一个空的异构图
        G = dgl.heterograph(GRAPH_CONFIG_T['graph_structure'])

        node_types = GRAPH_CONFIG_T['node_types']
        # 添加一个对话节点，对话节点类型为 'conversation'
        G.add_nodes(1, ntype=node_types[0])
        
        # 添加话语节点，节点类型为 'utterance'
        G.add_nodes(num_utterances, ntype=node_types[1])

        # 添加情绪节点，节点类型为 'emotion'
        G.add_nodes(num_utterances, ntype=node_types[2])

        # 添加原因节点，节点类型为 'cause'
        G.add_nodes(num_utterances, ntype=node_types[3])


        # 为了方便创建边，我们需要获取对话节点和话语节点的ID
        dialogue_node_id = 0  # 对话节点的ID
        utterance_node_ids = list(range(num_utterances))  # 话语节点的IDs，从0开始，因为异构图的不同类型的节点编号是分开编号的
        emotion_node_ids = list(range(num_utterances))  # 情绪节点的IDs
        cause_node_ids = list(range(num_utterances))  # 原因节点的IDs 

        # 从话语节点到对话节点
        G.add_edges(utterance_node_ids, [dialogue_node_id] * num_utterances, etype=('utterance','global','conversation'))
        # 从对话节点到话语节点
        G.add_edges([dialogue_node_id], utterance_node_ids, etype=('conversation','global', 'utterance'))

        # 从话语节点到情绪节点和原因节点
        G.add_edges(utterance_node_ids, emotion_node_ids, etype=('utterance', 'emotional_link', 'emotion'))
        G.add_edges(utterance_node_ids, cause_node_ids, etype=('utterance', 'causal_link', 'cause'))

        # 遍历每个话语节点，根据说话人关系添加边
        for i in range(num_utterances):
            count = 0  # 初始化计数器
            # 遍历前面的话语节点
            for j in range(i-1, -1, -1):
                if uttr_speaker_ids[j] == uttr_speaker_ids[i]:
                    # 同一说话人
                    G.add_edges(j, i, etype=('utterance', 'same_speaker', 'utterance'))
                    count += 1
                    if count == K:
                        break
                else:
                    # 不同说话人
                    G.add_edges(j, i, etype=('utterance', 'different_speaker', 'utterance'))
        if visualize:
            visualize_dgl_graph(G)

        return G


   

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

# 从数据集的视频文件中提取视频特征
def prepare_video_features(processor, visual_model, video_path, video_dir_list, data_type='train', num_frames = 8):
    def sample_or_pad_frames(frames, target_num_frames=8):
        """
        从帧列表中均匀采样或补充到指定数量的帧数
        :param frames: 帧列表
        :param target_num_frames: 目标帧数
        :return: 处理后的帧列表，数量为 target_num_frames
        """
        num_frames = len(frames)

        if num_frames >= target_num_frames:
            # 如果帧数多于目标帧数，则均匀采样
            indices = np.linspace(0, num_frames - 1, target_num_frames).astype(int)
            sampled_frames = [frames[i] for i in indices]
        else:
            # 如果帧数少于目标帧数，则通过重复或插值补充
            repeat_factor = target_num_frames // num_frames
            remainder = target_num_frames % num_frames
            sampled_frames = frames * repeat_factor + frames[:remainder]

        return sampled_frames
    def visualize_frames(frames, cols=4, title="Sampled Frames"):
        """
        可视化处理后的帧图像
        :param frames: 处理后的帧列表（PIL图像或NumPy数组）
        :param cols: 每行显示的帧数量
        :param title: 图像的标题
        """
        rows = len(frames) // cols + int(len(frames) % cols != 0)
        plt.figure(figsize=(cols * 3, rows * 3))
        plt.suptitle(title, fontsize=16)
        
        for i, frame in enumerate(frames):
            plt.subplot(rows, cols, i + 1)
            if isinstance(frame, np.ndarray):
                plt.imshow(frame)
            elif isinstance(frame, Image.Image):
                plt.imshow(np.asarray(frame))
            plt.axis('off')
        
        plt.savefig("sampled_frames.png")
    video_features = []
    for video_dir in video_dir_list:
        video_dir_path = os.path.join(video_path, data_type, video_dir)
        video_capture = cv2.VideoCapture(video_dir_path)
        frames = []
        success, frame = video_capture.read()
        while success:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
            frames.append(frame_rgb)
            success, frame = video_capture.read()
        video_capture.release()
        sampled_frames = sample_or_pad_frames(frames, target_num_frames = num_frames)
        # 将采样后的帧转换为 PIL 图像对象
        frames_pil = [Image.fromarray(frame) for frame in sampled_frames]
        # visualize_frames(frames_pil)
        video_inputs = processor(images=frames_pil, return_tensors="pt")
        with torch.no_grad():
            # 将 video_inputs 中的所有张量转移到 GPU 上
            video_inputs = {k: v.cuda() for k, v in video_inputs.items()}
            output_features = visual_model(pixel_values=video_inputs["pixel_values"].cuda()).last_hidden_state # shape: 1 * seq_len * feat_dim
            output_features = output_features.cpu()  # 移回 CPU
        video_features.append(output_features)
    return video_features



# 组合多个 WAV 文件，用于对话音频特征提取
def combine_wav_files(output_path, *input_paths):
    combined = AudioSegment.empty()  # 初始化一个空的音频片段
    
    for input_path in input_paths:
        sound = AudioSegment.from_wav(input_path)  # 加载每个 WAV 文件
        combined += sound  # 将音频文件首尾相接
    
    # 导出组合后的音频文件
    combined.export(output_path, format="wav")

# 序列截断处理
# input:
# tokens: 一段对话的词汇列表
# speaker_ids: 每个词汇对应的说话人列表
# mention_ids: 每个词汇对应的话语索引列表, 从1开始
# emotion_ids: 每个词汇对应的情绪类别列表
# max_seq_length: 最大序列长度
# cls_token: 分隔符
# 依次截断当前最长话语的最后一个token, 直到总长度小于max_seq_length
def truncate_sequences(tokens, speaker_ids, mention_ids, emotion_ids, max_seq_length, cls_token, sep_token = '</s>'):

    while len(tokens) > max_seq_length:
        # 将 tokens 通过结束标记进行分割，获取每个话语的长度
        utterance_lengths = [len(utterance.split())+1 for utterance in ' '.join(tokens).split(sep_token) if utterance] # 每个话语的长度, +1是因为要算上</s>
        # 1. 获取最大值及其索引
        max_length = max(utterance_lengths)
        max_index = utterance_lengths.index(max_length)
        # 2. 获取每个话语在 tokens 列表中的最后一个 token 的位置
        end_index = sum(utterance_lengths[:max_index + 1]) - 1
        # 3. 截断 tokens、speaker_ids、mention_ids、emotion_ids
        tokens.pop(end_index-1)
        speaker_ids.pop(end_index-1)
        mention_ids.pop(end_index-1)
        emotion_ids.pop(end_index-1)
    return tokens, speaker_ids, mention_ids, emotion_ids
    

def convert_examples_to_features(examples, tokenizer, max_seq_length, max_speaker_num, encoder_type = "BERT", data_type= 'train'):  
        logger.info("Converting {} examples to features.".format(len(examples)))
        features = [[]]
        for (idx, example) in enumerate(examples):
            uttr_speaker_ids = example.speaker_ids
            tokens, speaker_ids, mention_ids, emotion_ids, = tokenize(example.utterance_list, example.emotion_list, tokenizer, max_speaker_num)
            emotion_list = example.emotion_list
            utterance_list = example.utterance_list
            cause_list = example.cause_list
            uttr_len = example.uttr_len
            # 进行截断处理
            if len(tokens) > max_seq_length:
                tokens, speaker_ids, mention_ids, emotion_ids  = truncate_sequences(tokens, speaker_ids, mention_ids, emotion_ids, max_seq_length, tokenizer.cls_token, tokenizer.sep_token)
            assert len(tokens) == len(speaker_ids) == len(mention_ids) == len(emotion_ids)
            # 处理截断，直接将后面的
            assert len(tokens) <= max_seq_length
            uttr_indices = torch.LongTensor([i for i, x in enumerate(tokens) if x == tokenizer.cls_token]) # 用于定位每个utterance的起始位置

            # assert len(uttr_indices) == len(emotion_list)
            
            segment_ids = []
            segments_indices = [i for i, x in enumerate(tokens) if x == tokenizer.cls_token]
            segments_indices.append(len(tokens))
            for i in range(len(segments_indices)-1):
                semgent_len = segments_indices[i+1] - segments_indices[i]
                if i % 2 == 0:
                    segment_ids.extend([0] * semgent_len)
                else:
                    segment_ids.extend([1] * semgent_len)
            # segment_ids = [0] * len(tokens)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            ec_pairs = example.ec_pairs
            # Pad up to the sequence length
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                speaker_ids.append(0)
                mention_ids.append(0)
                emotion_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(speaker_ids) == max_seq_length
            assert len(mention_ids) == max_seq_length
            assert len(emotion_ids) == max_seq_length

            if idx < 2:
                logger.info("-------- Input Example --------")
                logger.info("guid: %s" % example.guid)
                logger.info("tokens: %s" % " ".join([x for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logger.info("speaker_ids: %s" % " ".join([str(x) for x in speaker_ids]))
                logger.info("mention_ids: %s" % " ".join([str(x) for x in mention_ids]))
                logger.info("emotion_ids: %s" % " ".join([str(x) for x in emotion_ids]))
            
            features[-1].append(
                        InputFeatures(
                            conversation_id = example.conversation_id,
                            input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            ec_pairs=ec_pairs,
                            speaker_ids=speaker_ids,
                            mention_ids=mention_ids,
                            emotion_ids=emotion_ids,
                            emotion_list = emotion_list,
                            cause_list = cause_list,
                            uttr_indices = uttr_indices,
                            uttr_len= uttr_len,
                            uttr_speaker_ids = uttr_speaker_ids,
                            uttr_idx = [uttr_idx.split('.')[0] for uttr_idx in example.video_dir_list]
                        )
                    )
            if len(features[-1]) == 1:
                features.append([])

        if len(features[-1]) == 0:
            features = features[:-1]
        
        logging.info("Feature: {}".format(len(features)))
        return features




def tokenize(utterance_list, emotions, tokenizer, max_speaker_num):
    speaker2id = {}
    for i in range(1, max_speaker_num + 1):
        token = "S{}".format(i)
        speaker2id[token] = i

    speaker_ids = []
    mention_ids = []
    emotion_ids = []
    speaker_id = 0
    mention_id = 0
    context = ''
    for i in range(len(utterance_list)):
        context += tokenizer.cls_token + utterance_list[i] + tokenizer.sep_token

    # tokens = tokenizer(context, return_tensors="pt", padding=True, truncation=True, add_special_tokens=True)
    tokens = tokenizer.tokenize(context)
    # 初始化说话人id
    # for token in tokens:
    #     if token in speaker2id.keys():
    #         speaker_id = speaker2id[token]
    #         break
    for token in tokens:
        if token in speaker2id.keys():
            speaker_id = speaker2id[token]
            mention_id += 1
        # if token == tokenizer.cls_token or token == tokenizer.sep_token:
        #     speaker_ids.append(0) # 记录每个token当前的说话人id
        #     mention_ids.append(0) # 每个token处在话语的index
        #     emotion_ids.append(0) # 每个token的情绪类别ID
        # else:
        speaker_ids.append(speaker_id) # 记录每个token当前的说话人id
        mention_ids.append(mention_id) # 每个token处在话语的index
        emotion_ids.append(emotions[mention_id - 1]) # 每个token的情绪类别ID

    return tokens, speaker_ids, mention_ids, emotion_ids

def mention2mask(mention_id, window=1):
    slen = len(mention_id) # max_seq_length
    mask = []
    turn_mention_ids = [i for i in range(1, np.max(mention_id) - 1)]
    for j in range(slen):
        if mention_id[j] not in turn_mention_ids:
            tmp = np.zeros(slen, dtype=bool)
            tmp[j] = 1
        else:
            start = mention_id[j]
            end = mention_id[j]
            if mention_id[j] - window in turn_mention_ids:
                start = mention_id[j] - window

            if mention_id[j] + window in turn_mention_ids:
                end = mention_id[j] + window
            tmp = (mention_id >= start) & (mention_id <= end)
        mask.append(tmp)
    mask = np.stack(mask)
    return mask