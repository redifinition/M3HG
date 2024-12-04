import json
import logging
import os
import random
import numpy as np
import scipy.sparse as sp
import torch
import wandb
import argparse
from transformers import RobertaTokenizer
from torch.utils.data import DataLoader
import yaml
from model.baselines.rankcp.rank_cp import RankCP
from model.model import MECPECModel
from model.model_mecad import M3HG
from pretrained_models.textual.RoBERTa.configuration_roberta import RobertaConfig
from transformers import BertConfig,BertTokenizer
from utils.data_loader import MECPECDataLoader
from utils.dataset import MECPECDataset
# from transformers import AdamW
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import shutil
from tqdm import tqdm, trange
from transformers import Wav2Vec2Processor, Wav2Vec2Config, Wav2Vec2Model
from transformers import VideoMAEConfig, VideoMAEImageProcessor, VideoMAEModel
from utils.eval_func import calc_eval_result, get_pred_result
from utils.global_variables import DEVICE
from dataclasses import dataclass, field

# 设置wandb为离线模式
os.environ["WANDB_MODE"] = "offline"
wandb.init(  
    # set the wanZ5EE8@i$3ji@db project where this run will be logged
    project="DAG-MECPEC-Roberta-large-a",
    name='Roberta-GAT-GRU-MLP',
    
)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(filename)s[line:%(lineno)d] - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


from dataclasses import dataclass
import yaml

@dataclass
class Config:
    data_dir: str = "data"
    network: str = "MECPEC"
    textual_pretrain_model_dir: str = "pretrained_models/textual/RoBERTa"
    # audio_pretrain_model_dir: str = "pretrained_models/audio/wav2vec2-base-960h"
    video_pretrain_model_dir: str = "pretrained_models/video/XCLIP"
    video_dir: str = "data/ECF/videos"
    output_dir: str = "output"
    pretrain_config_file: str = "config.json"
    vocab_file: str = "vocab.txt"
    data_name: str = "ECF"
    pretrain_encoder_type: str = "RoBERTa"
    init_pretrain_checkpoint: str = "pytorch_model.bin"
    do_lower_case: bool = False
    do_train: bool = True
    do_eval: bool = True
    max_speaker_num: int = 6
    max_seq_length: int = 512
    num_train_epochs: int = 50
    train_batch_size: int = 64
    eval_batch_size: int = 64
    learning_rate: float = 5e-5
    gradient_accumulation_steps: int = 2
    warmup_proportion: float = 0.1
    no_cuda: bool = False
    local_rank: int = -1
    save_checkpoints_steps: int = 1000
    seed: list = field(default_factory=lambda: [])
    optimize_on_cpu: bool = False
    fp16: bool = False
    loss_scale: float = 128
    resume: bool = False
    f1eval: bool = True
    mlp_layers: int = 2
    dropout: float = 0.1
    hidden_dim: int = 512
    model_metric: str = 'weighted_avg_f1_6'
    K: int = 1
    gat_layers: int = 2
    num_graph_attention_heads: int = 1
    pos_emb_dim: int = 50
    rel_pos_k : int = 15
    modality: list = field(default_factory=lambda: ['textual', 'audio'])
    # audio_extractor: str = 'wav2vec2'
    audio_feature_dim: int = 6373
    rnn_drop: float =  0.15
    rnn_n_layers: int = 2
    rnn_type: str = 'gru'
    use_vanilla: bool = False
    use_rnnpack: bool = False
    cross_num_head: int = 8
    cross_drop: float = 0.4
    cross_n_layers: int = 5
    audio_features_pkl_path: str = 'data/ECF/AudioFeatures.pkl'
    video_features_pkl_path: str = 'data/ECF/VideoFeatures.pkl'



    @classmethod
    def from_yaml(cls, config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return cls(**config)


        
def set_random_seed_all(seed):
    from transformers import set_seed
    # 设置随机种子
    set_seed(seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    os.environ['PYTHONHASHSEED'] =str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # 或者使用 ':16:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(False)


def set_optimizer_params_grad(named_params_optimizer, named_params_model, test_nan=False):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the gradient of the GPU parameters to the CPU/RAMM copy of the model
    """
    is_nan = False
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        if test_nan and torch.isnan(param_model.grad).sum() > 0:
            is_nan = True
        if param_opti.grad is None:
            param_opti.grad = torch.nn.Parameter(param_opti.data.new().resize_(*param_opti.data.size()))
        param_opti.grad.data.copy_(param_model.grad.data)
    return is_nan

def copy_optimizer_params_to_model(named_params_model, named_params_optimizer):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the parameters optimized on CPU/RAM back to the model on GPU
    """
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        param_model.data.copy_(param_opti.data)

def get_speaker_dict(dataset, data_dir):
    speaker2id = {}
    id = 1
    with open(os.path.join(data_dir,dataset,'all_data.json'),'r') as f:
        data = json.load(f)
    # ECF数据集的说话人数量
    for conversation in data:
        for utterance in conversation['conversation']:
            speaker = utterance['speaker']
            speaker2id[speaker] = id
            id+=1


def run(config, seed):
    from tensorboardX import SummaryWriter

    # 初始化 SummaryWriter
    writer = SummaryWriter(logdir='logs')  # 日志目录
    set_random_seed_all(seed)
    if config.local_rank == -1 or config.no_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() and not config.no_cuda else "cpu")
            n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", config.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')
        if config.fp16:
            logger.info("16-bits training currently not supported in distributed training")
            config.fp16 = False

    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(config.local_rank != -1))

    if config.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            config.gradient_accumulation_steps))

    config.train_batch_size = int(config.train_batch_size / config.gradient_accumulation_steps)

    if not config.do_train and not config.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")
    
    if config.pretrain_encoder_type == 'RoBERTa':
        lm_config = RobertaConfig.from_pretrained(config.textual_pretrain_model_dir)
    elif config.pretrain_encoder_type == 'BERT':
        lm_config = BertConfig.from_pretrained(config.textual_pretrain_model_dir)
    else:
        raise ValueError("The encoder type is invalid.")
    
    # wav2vec2_model, visual_model = None, None
    # am_config = None
    # if 'audio' in config.modality:
    #     am_config = Wav2Vec2Config.from_pretrained(config.audio_pretrain_model_dir)
    #     wav2vec2_model = Wav2Vec2Model.from_pretrained(config.audio_pretrain_model_dir)
    #     wav2vec2_model = wav2vec2_model.to('cuda')  # 将模型移动到 GPU 上
    visual_model =  None
    if 'video' in config.modality:
        visual_model = VideoMAEModel.from_pretrained(config.video_pretrain_model_dir)
        visual_model = visual_model.to('cuda')  # 将模型移动到 GPU 上
    
    # if config.max_seq_length > lm_config.max_position_embeddings:
    if config.max_seq_length > 512:
        raise ValueError(
            "Cannot use sequence length {} because the pretrained model was "
            "only trained up to sequence length {}".format(config.max_seq_length, lm_config.max_position_embeddings))

    output_dir = os.path.join(config.output_dir, config.data_name + '_' + config.pretrain_encoder_type)
    if os.path.exists(output_dir) and 'model.pt' in os.listdir(output_dir):
        if config.do_train and not config.resume:
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)
    else:
        os.makedirs(output_dir, exist_ok=True)

    if config.pretrain_encoder_type == 'RoBERTa':
        tokenizer = RobertaTokenizer.from_pretrained(config.textual_pretrain_model_dir, do_lower_case=False)
    elif config.pretrain_encoder_type == 'BERT':
        tokenizer = BertTokenizer.from_pretrained(config.textual_pretrain_model_dir, do_lower_case=False)
    else:
        raise ValueError("The pretrained tokenizer has not been initialized.")

    # 创建speaker2id字典
    speaker2id = {}
    for i in range(1, config.max_speaker_num + 1):
        token = "S{}".format(i)
        speaker2id[token] = i

    # 增加额外的特殊tokens
    special_tokens_dict = {'additional_special_tokens': list(speaker2id.keys())}
    tokenizer.add_special_tokens(special_tokens_dict)


    
    num_train_steps = None

    train_set, train_loader, valid_set, valid_loader, test_set, test_loader = None, None, None, None, None, None
    saved_dir = os.path.join("processed", config.data_name + '_' + config.pretrain_encoder_type)
    data_dir = os.path.join(config.data_dir, config.data_name)
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    
    video_processor = None, None
    # # 加载音频预处理器
    # if 'audio' in config.modality:
    #     audio_processor = Wav2Vec2Processor.from_pretrained(config.audio_pretrain_model_dir)
    if 'video' in config.modality:
        video_processor = VideoMAEImageProcessor.from_pretrained(config.video_pretrain_model_dir)

    if config.do_train:
        train_set = MECPECDataset(input_dir = data_dir, saved_file = os.path.join(saved_dir, 'train.pkl'), 
                                max_seq_length = config.max_seq_length, tokenizer = tokenizer, audio_features_path = config.audio_features_pkl_path, video_features_pkl_path = config.video_features_pkl_path,
                                visual_model = visual_model , video_processor = video_processor, video_path = config.video_dir, encoder_type = config.pretrain_encoder_type,
                                max_speaker_num = config.max_speaker_num, data_name = config.data_name, data_type='train', K = config.K)
        num_train_steps = int(
            len(train_set) / config.train_batch_size / config.gradient_accumulation_steps * config.num_train_epochs)
        train_loader = MECPECDataLoader(dataset=train_set, batch_size=config.train_batch_size, shuffle=True,
                                        max_length=config.max_seq_length, modality = config.modality)
        dev_set = MECPECDataset(input_dir = data_dir, saved_file = os.path.join(saved_dir, 'dev.pkl'), 
                                max_seq_length = config.max_seq_length, tokenizer = tokenizer, audio_features_path = config.audio_features_pkl_path, video_features_pkl_path = config.video_features_pkl_path,
                                visual_model = visual_model , video_processor = video_processor, video_path = config.video_dir, encoder_type = config.pretrain_encoder_type,
                                max_speaker_num = config.max_speaker_num, data_name = config.data_name, data_type='valid', K = config.K)
        dev_loader = MECPECDataLoader(dataset=dev_set, batch_size=config.eval_batch_size, shuffle=False,
                                    max_length=config.max_seq_length, modality = config.modality)
    if config.do_eval:
        test_set = MECPECDataset(input_dir = data_dir, saved_file = os.path.join(saved_dir, 'test.pkl'), 
                                max_seq_length = config.max_seq_length, tokenizer = tokenizer, audio_features_path = config.audio_features_pkl_path, video_features_pkl_path = config.video_features_pkl_path,
                                  visual_model = visual_model , video_processor = video_processor, video_path = config.video_dir, encoder_type = config.pretrain_encoder_type,
                                max_speaker_num = config.max_speaker_num, data_name = config.data_name, data_type='test', K = config.K)
        test_loader = MECPECDataLoader(dataset=test_set, batch_size=config.eval_batch_size, shuffle=False,
                                    max_length=config.max_seq_length, modality = config.modality)
    

    # Todo: 模型初始化
    if config.pretrain_encoder_type == 'RoBERTa':
        model = MECPECModel(lm_config, config, data_name=config.data_name, modality = config.modality).to(device)
        model_path = os.path.join(config.textual_pretrain_model_dir, config.init_pretrain_checkpoint)
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
        else:
            raise FileNotFoundError("The pre-trained model path does not exist: {}.".format(model_path))
        if 'textual' in config.modality:
            model.roberta.resize_token_embeddings(len(tokenizer))
    elif config.pretrain_encoder_type == 'BERT':
        if config.network == 'MECPEC':
            if config.data_name == 'ECF':
                model = MECPECModel(lm_config, config, data_name=config.data_name, modality = config.modality).to(device)
            else:
                model = M3HG(lm_config, config, data_name=config.data_name, modality = config.modality).to(device)
        elif config.network == 'RankCP':
            # RankCP的配置项
            config.feat_dim = 768
            config.gnn_dims = '192'
            config.att_heads = '4'
            config.K = 12
            config.pos_emb_dim = 50
            config.pairwise_loss = False
            model = RankCP(config, data_name=config.data_name).to(device)
        if 'textual' in config.modality:
            model.bert.resize_token_embeddings(len(tokenizer))
    else:
        raise ValueError("The pretrained model has not been initialized.")  


    if config.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.local_rank],
                                                        output_device=config.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if config.fp16:
        param_optimizer = [(n, param.clone().detach().to('cpu').float().requires_grad_())
                        for n, param in model.named_parameters()]
    elif config.optimize_on_cpu:
        param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_())
                        for n, param in model.named_parameters()]
    else:
        param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if n not in no_decay], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if n in no_decay], 'weight_decay_rate': 0.0}
    ]
    if config.do_train:
        optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(config.warmup_proportion * num_train_steps),
                                                num_training_steps=num_train_steps)
    

    global_step = 0
    if config.resume:
        model.load_state_dict(torch.load(os.path.join(output_dir, "model_best.pt")))

    if config.do_train:
        best_metric = 0
        logger.info("-------- Start training --------")
        logger.info("\tExample Size: {}".format(len(train_set)))
        logger.info("\tBatch Size: {}".format(config.train_batch_size))
        logger.info("\tStep Number: {}".format(num_train_steps))

        # 定义epoch的横坐标
        wandb.define_metric("train/epoch/step")
        wandb.define_metric("train/epoch/*", step_metric="train/epoch/step")
        wandb.define_metric("valid/epoch/step")
        wandb.define_metric("valid/epoch/*", step_metric="valid/epoch/step")
        # 定义batch的横坐标
        wandb.define_metric("train/batch/step")
        wandb.define_metric("train/batch/*", step_metric="train/batch/step")
        wandb.watch(model, log="all")
        wandb_batch_step = 0
        best_epoch = 0 # 记录效果最好的epoch
        for cur_epoch in trange(int(config.num_train_epochs), desc="Epoch"):
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_loader, desc=f"Training | Epoch {cur_epoch} | {config.num_train_epochs}")):
                input_ids = batch['input_ids'].to(device)
                segment_ids = batch['segment_ids'].to(device)
                mention_ids = batch['mention_ids'].to(device)
                emotion_ids = batch['emotion_ids'].to(device)
                speaker_ids = batch['speaker_ids'].to(device)
                input_masks = batch['input_masks'].to(device)
                turn_masks = batch['turn_masks'].to(device)
                emotion_list = batch['emotion_list'].to(device) # 这里的标签为1-7，0表示无效
                cause_list = batch['cause_list'].to(device)
                uttr_indices = batch['uttr_indices'].to(device)
                uttr_len = batch['uttr_len']
                ec_pair = batch['ec_pair']
                graphs = batch['graphs']
                modality = config.modality
                audio_features, video_features = None, None
                if 'audio' in modality:
                    audio_features = batch['audio_features']
                if 'video' in modality:
                    video_features = batch['video_features']
                # audio_uttr_indices = batch['audio_uttr_indices']
                # audio_mentions_ids = batch['audio_mentions_ids']
                uttr_mask = emotion_list > 0
                if config.network == 'MECPEC':
                    em_logits, ca_logits, couples_pred, emo_caus_pos = model(input_ids = input_ids, token_type_ids = segment_ids, 
                                    attention_masks=input_masks, speaker_ids=speaker_ids, mention_ids=mention_ids,
                                    emotion_ids=emotion_ids, turn_masks=turn_masks, uttr_indices = uttr_indices, graphs = graphs, uttr_len = uttr_len, 
                                    audio_features = audio_features, video_features = video_features, modality = modality)
                    if config.data_name == 'ECF':
                        loss_e = model.loss_pre_emo(em_logits, emotion_list)
                        loss_ec = model.loss_pre_ec(couples_pred, emo_caus_pos, ec_pair, uttr_mask)
                        loss_c = model.loss_pre_cau(ca_logits, cause_list)
                    else:
                        loss_e, loss_c = model.loss_pre(em_logits, ca_logits, emotion_list, cause_list)
                        loss_ec  = model.loss_rank(couples_pred, emo_caus_pos, ec_pair, uttr_mask)
                    loss = loss_e + loss_ec + loss_c
                elif config.network == 'RankCP':
                    # adj
                    adj = pad_matrices(uttr_len)
                    adj = np.array(adj)
                    couples_pred, emo_cau_pos, pred_e, pred_c = model(input_ids, segment_ids, input_masks, uttr_indices, np.array(uttr_len), adj)

                    loss_e, loss_c = model.loss_pre(pred_e, pred_c, emotion_list, cause_list)
                    loss_ec  = model.loss_rank(couples_pred, emo_cau_pos, ec_pair, uttr_mask)
                    loss = loss_e + loss_c + loss_ec
                if n_gpu > 1:
                    loss = loss.mean()
                if config.fp16 and config.loss_scale != 1.0:
                    loss = loss * config.loss_scale
                if config.gradient_accumulation_steps > 1:
                    loss = loss / config.gradient_accumulation_steps
                loss.backward()
                tr_loss += loss.item()
                wandb.log({"train/batch/loss": loss.item(),"train/batch/loss_emotion": loss_e.item(),"train/batch/loss_c": loss_c.item(),
                        "train/batch/loss_ec":loss_ec.item(),"train/batch/step": wandb_batch_step})
                writer.add_scalar("train/batch/loss", loss.item(), wandb_batch_step)
                writer.add_scalar("train/batch/loss_emotion", loss_e.item(), wandb_batch_step)
                writer.add_scalar("train/batch/loss_c", loss_c.item(), wandb_batch_step)
                writer.add_scalar("train/batch/loss_ec", loss_ec.item(), wandb_batch_step)
                wandb_batch_step+=1
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % config.gradient_accumulation_steps == 0:
                    if config.fp16 or config.optimize_on_cpu:
                        if config.fp16 and config.loss_scale != 1.0:
                            # scale down gradients for fp16 training
                            for param in model.parameters():
                                param.grad.data = param.grad.data / config.loss_scale
                        is_nan = set_optimizer_params_grad(param_optimizer, model.named_parameters(), test_nan=True)   
                        if is_nan:
                            logger.info("FP16 TRAINING: Nan in gradients, reducing loss scaling")
                            config.loss_scale = config.loss_scale / 2
                            model.zero_grad()
                            continue
                        optimizer.step()
                        scheduler.step()
                        copy_optimizer_params_to_model(model.named_parameters(), param_optimizer)
                    else:
                        optimizer.step()
                        scheduler.step()
                    optimizer.zero_grad()
                    model.zero_grad()
                    global_step += 1
            wandb.log({"train/epoch/loss": tr_loss/nb_tr_steps, "train/epoch/step":cur_epoch})
            logger.info("-------- Start Evaluating: {}--------".format(cur_epoch)) 
            if cur_epoch == 26:
                a = 1 # 调试
            emotion_pred_all, emotion_list_all, couples_pred_all, ec_pair_all, eval_loss = evaluate(model, dev_loader, 'valid' , config.data_name, config.network, config.modality)
            eval_result = calc_eval_result(emotion_pred_all, emotion_list_all, couples_pred_all, ec_pair_all)
            eval_f1 = eval_result[config.model_metric] #使用w-avg-f1-6作为最佳模型选择
            wandb.log({"valid/epoch/loss": eval_loss, "valid/epoch/ec_f1": eval_result['f1_ec'], 
                    "valid/epoch/emo_f1": eval_result['f1_emo'],
                    "valid/epoch/w-avg-6-f1":eval_result["weighted_avg_f1_6"], "valid/epoch/step":cur_epoch})
            writer.add_scalar("valid/epoch/loss", eval_loss, cur_epoch)
            writer.add_scalar("valid/epoch/ec_f1", eval_result['f1_ec'], cur_epoch)
            writer.add_scalar("valid/epoch/emo_f1", eval_result['f1_emo'], cur_epoch)
            writer.add_scalar("valid/epoch/w-avg-6-f1", eval_result["weighted_avg_f1_6"], cur_epoch)

            torch.save(model.state_dict(), os.path.join(output_dir, "model_{}.pt".format(cur_epoch)))
            logger.info("-------- Eval Results --------")
            log_result(eval_result)
            if eval_f1 >= best_metric:
                torch.save(model.state_dict(), os.path.join(output_dir, "model_best.pt".format(cur_epoch)))
                best_metric = eval_f1
                best_epoch = cur_epoch
                logger.info("-------- Current Best Results --------")
                logger.info('In epoch {} get best performance:'.format(best_epoch))
                for key in sorted(eval_result.keys()):
                    logger.info("{}: {}".format(key, str(eval_result[key])))

        # model.load_state_dict(torch.load(os.path.join(output_dir, "model_best.pt")))
        # torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))

    if config.do_eval:
        model.load_state_dict(torch.load(os.path.join(output_dir, "model_best.pt")))

        logger.info("-------- Start Testing --------")
        logger.info("\tExample Size: {}".format(len(test_set)))
        logger.info("\tBatch Size: {}".format(config.eval_batch_size))
        emotion_pred_all, emotion_list_all, couples_pred_all, ec_pair_all, eval_loss = evaluate(model, test_loader, 'test', config.data_name, config.network, config.modality)
        eval_result = calc_eval_result(emotion_pred_all, emotion_list_all, couples_pred_all, ec_pair_all)
        output_file =  os.path.join(output_dir, "test_result_{}.txt".format(seed))
        log_result(eval_result)
        with open(output_file, 'w') as f:
            f.write("Emotion Predictions:\n")
            for i, emotions in enumerate(emotion_pred_all):
                f.write(f"Dialogue {i+1}: {emotions}\n")
            
            f.write("\nCouples Predictions:\n")
            for i, couples in enumerate(couples_pred_all):
                couples_str = " ".join([f"({c[0]}, {c[1]})" for c in couples])
                f.write(f"Dialogue {i+1}: {couples_str}\n")
    return eval_result

def log_result(eval_result):
    for key in sorted(eval_result.keys()):
        logger.info("{}: {}".format(key, str(eval_result[key])))  



# validate and test:
def evaluate(model, loader, mode='valid', data_name = 'ECF', network='MECPEC',  modality = []):
    model.eval()
    eval_loss, eval_f1 = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0 # number of evaluation steps and examples
    # em_logits_all = [] # 所有的情绪预测
    emotion_list_all = [] # 所有的情绪预测实际值
    emotion_pred_all = []
    couples_pred_all = []
    ec_pair_all = []
    # uttr_mask_all = []
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        input_ids = batch['input_ids'].to(DEVICE)
        segment_ids = batch['segment_ids'].to(DEVICE)
        mention_ids = batch['mention_ids'].to(DEVICE)
        emotion_ids = batch['emotion_ids'].to(DEVICE)
        speaker_ids = batch['speaker_ids'].to(DEVICE)
        input_masks = batch['input_masks'].to(DEVICE)
        turn_masks = batch['turn_masks'].to(DEVICE)
        emotion_list = batch['emotion_list'].to(DEVICE) # 这里的标签为1-7，0表示无效
        cause_list = batch['cause_list'].to(DEVICE)
        uttr_indices = batch['uttr_indices'].to(DEVICE)
        uttr_len = batch['uttr_len']
        ec_pair = batch['ec_pair']
        graphs = batch['graphs']
        audio_features, video_features = None, None
        if 'audio' in modality:
            audio_features = batch['audio_features']
        if 'video' in modality:
            video_features = batch['video_features']
        # audio_uttr_indices = batch['audio_uttr_indices']
        # audio_mention_ids = batch['audio_mentions_ids']
        uttr_mask = emotion_list > 0
        with torch.no_grad():
            if network == 'MECPEC':
                em_logits, ca_logits, couples_pred, emo_caus_pos = model(input_ids = input_ids, token_type_ids = segment_ids, 
                                attention_masks=input_masks, speaker_ids=speaker_ids,mention_ids=mention_ids,
                                emotion_ids=emotion_ids, turn_masks=turn_masks, uttr_indices = uttr_indices, graphs = graphs, uttr_len = uttr_len, 
                                audio_features = audio_features, video_features = video_features, modality = modality)
                if data_name == 'ECF':
                    loss_e = model.loss_pre_emo(em_logits, emotion_list)
                    loss_c = model.loss_pre_cau(ca_logits, cause_list)
                    loss_ec = model.loss_pre_ec(couples_pred, emo_caus_pos, ec_pair, uttr_mask, test=True)   
                else:
                    loss_e, loss_c = model.loss_pre(em_logits, ca_logits, emotion_list, cause_list)
                    loss_ec = model.loss_rank(couples_pred, emo_caus_pos, ec_pair, uttr_mask)
                loss = loss_e + loss_ec + loss_c
            elif network == 'RankCP':
                # adj
                adj = pad_matrices(uttr_len)
                adj = np.array(adj)
                couples_pred, emo_caus_pos, pred_e, pred_c = model(input_ids, segment_ids, input_masks, uttr_indices, np.array(uttr_len), adj)
                em_logits = pred_e
                loss_e, loss_c = model.loss_pre(pred_e, pred_c, emotion_list, cause_list)
                loss_ec  = model.loss_rank(couples_pred, emo_caus_pos, ec_pair, uttr_mask)
                loss = loss_e + loss_c + loss_ec
            eval_loss += loss.mean().item()
            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1
            predicted_emotions_batch, pred_ec_pairs_batch = get_pred_result(em_logits, couples_pred, emo_caus_pos, uttr_mask)
            emotion_list_all.extend(emotion_list.cpu().numpy().tolist())
            emotion_pred_all.extend(predicted_emotions_batch.tolist())
            couples_pred_all.extend(pred_ec_pairs_batch)
            ec_pair_all.extend(ec_pair)
    eval_loss = eval_loss / nb_eval_steps
    return emotion_pred_all, emotion_list_all, couples_pred_all, ec_pair_all, eval_loss

def pad_matrices(doc_len_b):
    N = max(doc_len_b)
    adj_b = []
    for doc_len in doc_len_b:
        adj = np.ones((doc_len, doc_len))
        adj = sp.coo_matrix(adj)
        adj = sp.coo_matrix((adj.data, (adj.row, adj.col)),
                            shape=(N, N), dtype=np.float32)
        adj_b.append(adj.toarray())
    return adj_b

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default="config.yaml", type=str, help="Path to the YAML config file.")
    args = parser.parse_args()

    # 使用Config类加载配置
    config = Config.from_yaml(args.config_file)
    wandb.config.update(config.__dict__)
    # 初始化存储结果的数据结构
    f1_results = {
        'f1_per_class': [],
        'weighted_avg_f1_4': [],
        'weighted_avg_f1_6': []
    }
    for seed in config.seed:
        eval_result = run(config, seed)
        f1_results['f1_per_class'].append(eval_result['f1_per_class'])
        f1_results['weighted_avg_f1_4'].append(eval_result['weighted_avg_f1_4'])
        f1_results['weighted_avg_f1_6'].append(eval_result['weighted_avg_f1_6'])

    # 计算每个指标的平均值和标准偏差
    summary_results = {}
    summary_results['f1_per_class_mean'] = np.mean(f1_results['f1_per_class'], axis=0)
    summary_results['f1_per_class_std'] = np.std(f1_results['f1_per_class'], axis=0, ddof=1)
    summary_results['weighted_avg_f1_4_mean'] = np.mean(f1_results['weighted_avg_f1_4'])
    summary_results['weighted_avg_f1_4_std'] = np.std(f1_results['weighted_avg_f1_4'], ddof=1)
    summary_results['weighted_avg_f1_6_mean'] = np.mean(f1_results['weighted_avg_f1_6'])
    summary_results['weighted_avg_f1_6_std'] = np.std(f1_results['weighted_avg_f1_6'], ddof=1)

    # 打印结果
    logger.info("-------- Final Results --------")
    log_result(summary_results)

if __name__ == "__main__":
    main()