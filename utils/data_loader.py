import math
import torch
import random
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from utils.global_variables import DEVICE


class MECPECDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=False, max_length=512, modality = None):
        super(MECPECDataLoader, self).__init__(dataset, batch_size=batch_size, num_workers=1)
        self.shuffle = shuffle
        self.length = len(self.dataset)
        self.max_length = max_length
        self.order = list(range(self.length))
        self.modality = modality
        print("initialize {}".format(len(dataset)))
    
    def __iter__(self):
        self._prepare_data()
        return self._batch_iterator()

    def _prepare_data(self):
        if self.shuffle:
            random.shuffle(self.order)
            self.data = [self.dataset[idx] for idx in self.order]
        else:
            self.data = self.dataset

        self.batch_num = math.ceil(self.length / self.batch_size)
        self.batches = [self.data[idx * self.batch_size: min(self.length, (idx + 1) * self.batch_size)]
                        for idx in range(0, self.batch_num)]
        print("batch_num {} \t batches_num {}".format(self.batch_num, len(self.batches)))
    
    # 初始化张量
    # uttr_len: 对话话语数
    # audio_max_uttr_len: 音频最大话语长度
    # audio_dim: 音频特征维度
    def _initialize_tensors(self, uttr_len, audio_dims, video_dims):   
        # self.input_ids = torch.LongTensor(self.batch_size, self.max_length).cpu()
        # self.segment_ids = torch.LongTensor(self.batch_size, self.max_length).cpu()
        # self.mention_ids = torch.LongTensor(self.batch_size, self.max_length).cpu()
        # self.emotion_ids = torch.LongTensor(self.batch_size, self.max_length).cpu()
        # self.speaker_ids = torch.LongTensor(self.batch_size, self.max_length).cpu()
        # self.input_masks = torch.LongTensor(self.batch_size, self.max_length).cpu()
        # self.turn_masks = torch.LongTensor(self.batch_size, self.max_length, self.max_length).cpu()
        # self.uttr_indices = torch.LongTensor(self.batch_size, uttr_len).cpu()
        # self.audio_features = torch.FloatTensor(self.batch_size, uttr_len, audio_dims).cpu()
        # self.video_features = torch.FloatTensor(self.batch_size, uttr_len, video_dims).cpu()
        self.input_ids = torch.zeros(self.batch_size, self.max_length, dtype=torch.long).cpu()
        self.segment_ids = torch.zeros(self.batch_size, self.max_length, dtype=torch.long).cpu()
        self.mention_ids = torch.zeros(self.batch_size, self.max_length, dtype=torch.long).cpu()
        self.emotion_ids = torch.zeros(self.batch_size, self.max_length, dtype=torch.long).cpu()
        self.speaker_ids = torch.zeros(self.batch_size, self.max_length, dtype=torch.long).cpu()
        self.input_masks = torch.zeros(self.batch_size, self.max_length, dtype=torch.long).cpu()
        self.turn_masks = torch.zeros(self.batch_size, self.max_length, self.max_length, dtype=torch.long).cpu()
        self.uttr_indices = torch.zeros(self.batch_size, uttr_len, dtype=torch.long).cpu()
        self.audio_features, self.video_features = None, None
        if 'audio' in self.modality:
            self.audio_features = torch.zeros(self.batch_size, uttr_len, audio_dims).cpu()
        if 'video' in self.modality:
            self.video_features = torch.zeros(self.batch_size, uttr_len, video_dims).cpu()

    def _zero_tensors(self):
        for item in [self.input_ids, self.segment_ids, self.mention_ids, self.emotion_ids, 
                    self.speaker_ids, self.input_masks, self.turn_masks, self.uttr_indices, self.audio_features]:
            if item is not None:
                item.zero_()

    def _batch_iterator(self):
        for batch in self.batches:
            batch_len = len(batch)
            uttr_indices = pad_sequence([example['uttr_indices'] for example in batch], batch_first=True, padding_value=0)
            audio_dim, video_dim = 0, 0
            if 'audio' in self.modality:
                audio_batch_data = [torch.tensor(example['audio_features'], dtype=float) for example in batch]
                audio_dim = audio_batch_data[0][0].shape[-1]
            if 'video' in self.modality:
                video_batch_data = [torch.tensor(example['video_features'], dtype=float) for example in batch]
                video_dim = video_batch_data[0][0].shape[-1]
            # audio_uttr_indices = [torch.tensor(example['audio_uttr_indices'], dtype=torch.long) for example in batch]

            # conv_audio_batch_data = [example['conversation_audio_features'] for example in batch]
            
            
            uttr_len = uttr_indices.size(1)
            # audio_seq_len = max([audio_feature.shape[-2] for audio_feature in audio_batch_data])
            self._initialize_tensors(uttr_len, audio_dim, video_dim)
            self._zero_tensors()
            graphs = []
            # audio_features = [example['audio_features']  for example in batch]
            # 去掉第一个维度 (1, utt_len, dim) -> (utt_len, dim)，因为我们只需要填充第二个维度
            # padded_audio_features = [torch.cat(feats) for feats in audio_features]  # 每个对话的多个utterances合并成一个tensor
            # padded_audio_features = pad_sequence(padded_audio_features, batch_first=True)


            # 生成 mask，1 表示有效帧，0 表示填充部分
            # audio_mask = torch.ones(padded_audio_features.size()[:-1])  # 先初始化为全1的mask
            # audio_mask[padded_audio_features.sum(dim=-1) == 0] = 0  # 找到填充部分并将其置为0
            for i, example in enumerate(batch):
                in_id, seg_id, mt_id, em_id, sp_id, in_mask, tr_mask, uttr_ind, uttr_len, graph = \
                    example['input_id'], example['segment_id'], example['mention_id'], \
                    example['emotion_id'], example['speaker_id'], example['input_mask'], \
                    example['turn_mask'], example['uttr_indices'], example['uttr_len'], example['graph']
                audio_uttr_idx = 0
                word_num = in_id.shape[0] # S:max_seq_length
                uttr_num = uttr_ind.shape[0] # N:utterance length
                graphs.append(graph.to(DEVICE))
                self.input_ids[i, :word_num].copy_(torch.from_numpy(in_id))
                self.segment_ids[i, :word_num].copy_(torch.from_numpy(seg_id))
                self.input_masks[i, :word_num].copy_(torch.from_numpy(in_mask))
                self.mention_ids[i, :word_num].copy_(torch.from_numpy(mt_id))
                self.emotion_ids[i, :word_num].copy_(torch.from_numpy(em_id))
                self.speaker_ids[i, :word_num].copy_(torch.from_numpy(sp_id))
                self.turn_masks[i, :word_num, :word_num].copy_(torch.from_numpy(tr_mask))
                self.uttr_indices[i, :uttr_num].copy_(uttr_ind)
                if 'audio' in self.modality:
                    self.audio_features[i, :audio_batch_data[i].shape[0], :].copy_(audio_batch_data[i][:, :].squeeze(0))
                if 'video' in self.modality:
                    self.video_features[i, :video_batch_data[i].shape[0], :].copy_(video_batch_data[i][:, :].squeeze(0))
                # self.audio_uttr_indices[i, :uttr_num].copy_(torch.tensor(audio_uttr_ind, dtype=torch.long))
                # self.audio_mentions_ids[i, :len(audio_mt_id)].copy_(torch.tensor(audio_mt_id, dtype=torch.long))
                # self.conv_audio_features[i, :].copy_(conv_audio_batch_data[i].squeeze(0))
                # self.audio_mask[i, :].copy_(audio_mask[i, :])
                audio_uttr_idx += uttr_num
                

            context_word_mask = self.input_masks > 0 # 这里之前是self.input_ids > 0， 有问题
            context_word_len = context_word_mask.sum(dim=1)
            batch_max_len = context_word_len.max() # 
            max_length = max(len(dialog['emotion_list']) for dialog in batch) # 最大话语长度
            padded_emotion_list = [example['emotion_list'] + [0] * (max_length - len(example['emotion_list'])) for example in batch]
            padded_cause_list = [example['cause_list'] + [-1] * (max_length - len(example['cause_list'])) for example in batch]
            ec_pairs = [example['ec_pairs'] for example in batch]
            uttr_len = tuple([example['uttr_len'] for example in batch])
            yield {
                'input_ids': get_cuda(self.input_ids[:batch_len, :batch_max_len].contiguous()),
                'segment_ids': get_cuda(self.segment_ids[:batch_len, :batch_max_len].contiguous()),
                'mention_ids': get_cuda(self.mention_ids[:batch_len, :batch_max_len].contiguous()),
                'emotion_ids': get_cuda(self.emotion_ids[:batch_len, :batch_max_len].contiguous()),
                'speaker_ids': get_cuda(self.speaker_ids[:batch_len, :batch_max_len].contiguous()),
                'input_masks': get_cuda(self.input_masks[:batch_len, :batch_max_len].contiguous()),
                'turn_masks': get_cuda(self.turn_masks[:batch_len, :batch_max_len, :batch_max_len].contiguous()),
                'emotion_list': get_cuda(torch.LongTensor(padded_emotion_list)),
                'cause_list': get_cuda(torch.LongTensor(padded_cause_list)),
                'ec_pair': ec_pairs,
                'uttr_indices': get_cuda(self.uttr_indices[:batch_len, :max_length].contiguous()),
                'uttr_len': uttr_len,
                'graphs': graphs,
                #'audio_features': get_cuda(self.audio_features[:batch_len, :, :].contiguous()),
                #'video_features': get_cuda(self.video_features[:batch_len, :, :].contiguous())
                # 'audio_uttr_indices': get_cuda(self.audio_uttr_indices[:batch_len, :max_length].contiguous()),
                # 'audio_mentions_ids': get_cuda(self.audio_mentions_ids[:batch_len, :].contiguous())
                #'audio_mask': get_cuda(self.audio_mask[:batch_len, :].contiguous()),
                # 'conv_audio_features': get_cuda(self.conv_audio_features[:batch_len, :].contiguous())
            }

# 假设 get_cuda 是一个将张量移动到 CUDA 设备的函数
def get_cuda(tensor):
    return tensor.cuda() if torch.cuda.is_available() else tensor


