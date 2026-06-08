import os
import signal
import time
import csv
import sys
import warnings
import random
import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import numpy as np
import time
import pprint
from loguru import logger
import smplx
from torch.utils.tensorboard import SummaryWriter
import wandb
import matplotlib.pyplot as plt
from utils import config, logger_tools, other_tools_hf, metric, data_transfer, other_tools
from utils.joints import upper_body_mask, hands_body_mask, lower_body_mask
from dataloaders import data_tools
from dataloaders.build_vocab import Vocab
from optimizers.optim_factory import create_optimizer
from optimizers.scheduler_factory import create_scheduler
from optimizers.loss_factory import get_loss_func
from dataloaders.data_tools import joints_list
from utils import rotation_conversions as rc
import soundfile as sf
import librosa 
import subprocess
from transformers import pipeline
from models.vq.model import RVQVAE

device = "cuda:0" if torch.cuda.is_available() else "cpu"

import platform
if platform.system() == "Linux":
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

pipe = pipeline(
  "automatic-speech-recognition",
  model="openai/whisper-tiny.en",
  chunk_length_s=30,
  device=device,
)       

debug = False

class BaseTrainer(object):
    def __init__(self, args, cfg, ap):
        hf_dir = "hf"
        time_local = time.localtime()
        time_name_expend = "%02d%02d_%02d%02d%02d_"%(time_local[1], time_local[2],time_local[3], time_local[4], time_local[5])
        self.time_name_expend = time_name_expend

        # Change this line to force the path to be relative to your current folder
        base_out = args.out_path.lstrip("/") # Remove any leading slash
        tmp_dir = "./" + base_out + "custom/" + time_name_expend + hf_dir
        
        # Now makedirs will work because it's inside your user directory
        if not os.path.exists(tmp_dir + "/"):
            os.makedirs(tmp_dir + "/", exist_ok=True)


        # tmp_dir = args.out_path + "custom/"+ time_name_expend + hf_dir
        # if not os.path.exists(tmp_dir + "/"):
        #     os.makedirs(tmp_dir + "/")
        self.audio_path = tmp_dir + "/tmp.wav"
        sf.write(self.audio_path, ap[1], ap[0])
        
        
        audio, ssr = librosa.load(self.audio_path,sr=args.audio_sr)
        

        # use asr model to get corresponding text transcripts
        file_path = tmp_dir+"/tmp.lab"
        self.textgrid_path = tmp_dir + "/tmp.TextGrid"
        if not debug:
            text = pipe(audio, batch_size=8)["text"]
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(text)
            
            # use montreal forced aligner to get textgrid
            
            command = ["mfa", "align", tmp_dir, "english_us_arpa", "english_us_arpa", tmp_dir]
            # result = subprocess.run(command, capture_output=True, text=True)
            # We add 'errors="ignore"' or specify 'encoding="utf-8"' to bypass the ASCII crash
            result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="ignore")
            print(result)
            

        ap = (ssr, audio)
        self.args = args
        self.rank = 0 # dist.get_rank()
       
        args.textgrid_file_path = self.textgrid_path
        args.audio_file_path = self.audio_path
    
    
        self.rank = 0 # dist.get_rank()
        print("args.dataset",args.dataset)
        # exit()
        self.checkpoint_path = tmp_dir
        args.tmp_dir = tmp_dir
        if self.rank == 0:
            self.test_data = __import__(f"dataloaders.{args.dataset}", fromlist=["something"]).CustomDataset(args, "test")
            self.test_loader = torch.utils.data.DataLoader(
                self.test_data, 
                batch_size=1,  
                shuffle=False,  
                num_workers=args.loader_workers,
                drop_last=False,
            )
        logger.info(f"Init test dataloader success")
        model_module = __import__(f"models.{cfg.model.model_name}", fromlist=["something"])
        
        self.model = torch.nn.DataParallel(getattr(model_module, cfg.model.g_name)(cfg), args.gpus).cuda()
        
        if self.rank == 0:
            logger.info(self.model)
            logger.info(f"init {cfg.model.g_name} success")

        self.smplx = smplx.create(
        self.args.data_path_1+"smplx_models/", 
            model_type='smplx',
            gender='NEUTRAL_2020', 
            use_face_contour=False,
            num_betas=300,
            num_expression_coeffs=100, 
            ext='npz',
            use_pca=False,
        ).to(self.rank).eval()    

        self.args = args
        self.ori_joint_list = joints_list[self.args.ori_joints]
        self.tar_joint_list_face = joints_list["beat_smplx_face"]
        self.tar_joint_list_upper = joints_list["beat_smplx_upper"]
        self.tar_joint_list_hands = joints_list["beat_smplx_hands"]
        self.tar_joint_list_lower = joints_list["beat_smplx_lower"]

        #
        total_dim = 165
        
        self.joint_mask_face = np.zeros(total_dim)
        self.joint_mask_upper = np.zeros(total_dim)
        self.joint_mask_hands = np.zeros(total_dim)
        self.joint_mask_lower = np.zeros(total_dim)
        #
        # self.joint_mask_face = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        self.joints = 55
        # for joint_name in self.tar_joint_list_face:
        #     self.joint_mask_face[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        #
        for joint_name in self.tar_joint_list_face:
            if joint_name in self.ori_joint_list:
                start = self.ori_joint_list[joint_name][0]
                end = self.ori_joint_list[joint_name][1]
                self.joint_mask_face[start:end] = 1
            else:
                print(f"Warning: Joint {joint_name} not found in ori_joint_list. Skipping.")
        #
        # self.joint_mask_upper = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        # for joint_name in self.tar_joint_list_upper:
        #     self.joint_mask_upper[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        #
        for joint_name in self.tar_joint_list_upper:
            if joint_name in self.ori_joint_list:
                self.joint_mask_upper[self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        #
        # self.joint_mask_hands = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        # for joint_name in self.tar_joint_list_hands:
        #     self.joint_mask_hands[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        #
        for joint_name in self.tar_joint_list_hands:
            if joint_name in self.ori_joint_list:
                self.joint_mask_hands[self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        #
        # self.joint_mask_lower = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        # for joint_name in self.tar_joint_list_lower:
        #     self.joint_mask_lower[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        #
        for joint_name in self.tar_joint_list_lower:
            if joint_name in self.ori_joint_list:
                self.joint_mask_lower[self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        #

        # After the mask-building loops, add:
        if self.joint_mask_upper.sum() == 0:
            self.joint_mask_upper[36:75] = 1
        if self.joint_mask_lower.sum() == 0:
            self.joint_mask_lower[9:36] = 1
        if self.joint_mask_hands.sum() == 0:
            self.joint_mask_hands[75:165] = 1
        
        
        self.tracker = other_tools.EpochTracker(["fid", "l1div", "bc", "rec", "trans", "vel", "transv", 'dis', 'gen', 'acc', 'transa', 'exp', 'lvd', 'mse', "cls", "rec_face", "latent", "cls_full", "cls_self", "cls_word", "latent_word","latent_self","predict_x0_loss"], [False,True,True, False, False, False, False, False, False, False, False, False, False, False, False, False, False,False, False, False,False,False,False])

        
        ##### VQ-VAE models #####
        """Initialize and load VQ-VAE models for different body parts."""
        # Face VQ model
        vq_model_module = __import__("models.motion_representation", fromlist=["something"])
        self.vq_model_face = self._create_face_vq_model(vq_model_module)
        
        # Body part VQ models
        self.vq_models = self._create_body_vq_models()
        
        # Set all VQ models to eval mode
        self.vq_model_face.eval().to(self.rank)
        for model in self.vq_models.values():
            model.eval().to(self.rank)
        self.vq_model_upper, self.vq_model_hands, self.vq_model_lower = self.vq_models.values()
        self.vqvae_latent_scale = self.args.vqvae_latent_scale 


        self.args.vae_length = 240
        
        ##### Loss functions #####
        self.reclatent_loss = nn.MSELoss().to(self.rank)
        self.vel_loss = torch.nn.L1Loss(reduction='mean').to(self.rank)
        



        ###
        print(f"Mask sums — upper: {int(self.joint_mask_upper.sum())}, lower: {int(self.joint_mask_lower.sum())}, hands: {int(self.joint_mask_hands.sum())}")
        ###
        ##### Normalization #####
        self.use_trans = self.args.use_trans
        self.mean = np.load(args.mean_pose_path)
        self.std = np.load(args.std_pose_path)
        
        # Extract body part specific normalizations
        for part in ['upper', 'hands', 'lower']:
            mask = globals()[f'{part}_body_mask']
            setattr(self, f'mean_{part}', torch.from_numpy(self.mean[mask]).cuda())
            setattr(self, f'std_{part}', torch.from_numpy(self.std[mask]).cuda())
        
        # Translation normalization if needed
        if self.args.use_trans:
            self.trans_mean = torch.from_numpy(np.load(self.args.mean_trans_path)).cuda()
            self.trans_std = torch.from_numpy(np.load(self.args.std_trans_path)).cuda()
    
    def _create_face_vq_model(self, module):
        """Create and initialize face VQ model."""
        self.args.vae_layer = 2
        self.args.vae_length = 256
        self.args.vae_test_dim = 106
        #
        self.args.vae_codebook_size = 256
        #
        model = getattr(module, "VQVAEConvZero")(self.args).to(self.rank)
        other_tools.load_checkpoints(model, "./datasets/hub/pretrained_vq/face_vertex_1layer_790.bin", 
                                   self.args.e_name)
        return model
    
    def _create_body_vq_models(self):
        """Create VQ-VAE models for body parts."""
        vq_configs = {
            'upper': {'dim_pose': 78},
            'hands': {'dim_pose': 180},
            'lower': {'dim_pose': 54 if not self.args.use_trans else 57}
        }

        vq_models = {}
        for part, config in vq_configs.items():
            model = self._create_rvqvae_model(config['dim_pose'], part)
            vq_models[part] = model
            
        return vq_models
    
    def _create_rvqvae_model(self, dim_pose: int, body_part: str) -> RVQVAE:
        """Create a single RVQVAE model with specified configuration."""
        args = self.args
        model = RVQVAE(
            args, dim_pose, args.nb_code, args.code_dim, args.code_dim,
            args.down_t, args.stride_t, args.width, args.depth,
            args.dilation_growth_rate, args.vq_act, args.vq_norm
        )
        
        # Load pretrained weights
        checkpoint_path = getattr(args, f'vqvae_{body_part}_path')
        model.load_state_dict(torch.load(checkpoint_path)['net'])
        return model
      
    
    def inverse_selection(self, filtered_t, selection_array, n):
        original_shape_t = np.zeros((n, selection_array.size))
        selected_indices = np.where(selection_array == 1)[0]
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
        return original_shape_t
    
    def inverse_selection_tensor(self, filtered_t, selection_array, n):
        selection_array = torch.from_numpy(selection_array).cuda()
        original_shape_t = torch.zeros((n, 165)).cuda()
        selected_indices = torch.where(selection_array == 1)[0]
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
        return original_shape_t
    
    def _load_data(self, dict_data):

        # tar_pose = dict_data["pose"].to(self.rank) # The [1, 1680, 165] tensor
        
        # # Ensure masks are torch tensors and match the device
        # mask_u = torch.from_numpy(self.joint_mask_upper).bool().to(self.rank)
        
        # # Now the indexing will work because both are 165 dims
        # tar_pose_upper = tar_pose[:, :, mask_u]

        tar_pose_raw = dict_data["pose"]
        tar_pose = tar_pose_raw[:, :, :165].to(self.rank)
        tar_contact = tar_pose_raw[:, :, 165:169].to(self.rank)
        tar_trans = dict_data["trans"].to(self.rank)
        tar_trans_v = dict_data["trans_v"].to(self.rank)
        tar_exps = dict_data["facial"].to(self.rank)
        # in_audio = dict_data["audio"].to(self.rank)
        #
        if "audio" in dict_data:
            in_audio = dict_data["audio"].to(self.rank)
        elif "audio_onset" in dict_data:
            # Many versions of this dataloader use 'audio_onset' as the main audio key
            in_audio = dict_data["audio_onset"].to(self.rank)
        else:
            # Fallback: find the first key that looks like audio
            audio_key = [k for k in dict_data.keys() if 'audio' in k][0]
            in_audio = dict_data[audio_key].to(self.rank)
        #
        if 'wavlm' in dict_data:
            wavlm = dict_data["wavlm"].to(self.rank)
        else:
            wavlm = None
        in_word = dict_data["word"].to(self.rank)
        tar_beta = dict_data["beta"].to(self.rank)
        tar_id = dict_data["id"].to(self.rank).long()
        bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints

        # tar_pose_hands = tar_pose[:, :, 25*3:55*3]
        # tar_pose_hands = rc.axis_angle_to_matrix(tar_pose_hands.reshape(bs, n, 30, 3))
        # tar_pose_hands = rc.matrix_to_rotation_6d(tar_pose_hands).reshape(bs, n, 30*6)

        # tar_pose_upper = tar_pose[:, :, self.joint_mask_upper.astype(bool)]
        # tar_pose_upper = rc.axis_angle_to_matrix(tar_pose_upper.reshape(bs, n, 13, 3))
        # tar_pose_upper = rc.matrix_to_rotation_6d(tar_pose_upper).reshape(bs, n, 13*6)

        # tar_pose_leg = tar_pose[:, :, self.joint_mask_lower.astype(bool)]
        # tar_pose_leg = rc.axis_angle_to_matrix(tar_pose_leg.reshape(bs, n, 9, 3))
        # tar_pose_leg = rc.matrix_to_rotation_6d(tar_pose_leg).reshape(bs, n, 9*6)

        # --- Body Part Slicing ---
        # Hands: 30 joints * 3 = 90 dims
        tar_pose_hands = tar_pose[:, :, 75:165] 
        tar_pose_hands = rc.axis_angle_to_matrix(tar_pose_hands.reshape(bs, n, 30, 3))
        tar_pose_hands = rc.matrix_to_rotation_6d(tar_pose_hands).reshape(bs, n, 30*6)

        # Upper: 13 joints * 3 = 39 dims
        mask_u = torch.from_numpy(self.joint_mask_upper).bool().to(self.rank)
        tar_pose_upper = tar_pose[:, :, mask_u]
        # If mask failed, force it to indices used in BEAT2 (12-25)
        if tar_pose_upper.shape[-1] != 39:
            tar_pose_upper = tar_pose[:, :, 36:75] 
        tar_pose_upper = rc.axis_angle_to_matrix(tar_pose_upper.reshape(bs, n, 13, 3))
        tar_pose_upper = rc.matrix_to_rotation_6d(tar_pose_upper).reshape(bs, n, 13*6)

        # Lower: 9 joints * 3 = 27 dims
        mask_l = torch.from_numpy(self.joint_mask_lower).bool().to(self.rank)
        tar_pose_leg = tar_pose[:, :, mask_l]
        if tar_pose_leg.shape[-1] != 27:
            tar_pose_leg = tar_pose[:, :, 9:36] # Standard BEAT2 legs
        tar_pose_leg = rc.axis_angle_to_matrix(tar_pose_leg.reshape(bs, n, 9, 3))
        tar_pose_leg = rc.matrix_to_rotation_6d(tar_pose_leg).reshape(bs, n, 9*6)

        tar_pose_lower = tar_pose_leg

        if self.args.pose_norm:
            tar_pose_upper = (tar_pose_upper - self.mean_upper) / self.std_upper
            tar_pose_hands = (tar_pose_hands - self.mean_hands) / self.std_hands
            tar_pose_lower = (tar_pose_lower - self.mean_lower) / self.std_lower
        
        
        if self.use_trans:
            tar_trans_v = (tar_trans_v - self.trans_mean)/self.trans_std
            tar_pose_lower = torch.cat([tar_pose_lower,tar_trans_v], dim=-1)
      

        latent_upper_top = self.vq_model_upper.map2latent(tar_pose_upper)
        latent_hands_top = self.vq_model_hands.map2latent(tar_pose_hands)
        latent_lower_top = self.vq_model_lower.map2latent(tar_pose_lower)
        
        latent_in = torch.cat([latent_upper_top, latent_hands_top, latent_lower_top], dim=2)/self.args.vqvae_latent_scale
        
        style_feature = None
        
        return {
            "in_audio": in_audio,
            "wavlm": wavlm,
            "in_word": in_word,
            "tar_trans": tar_trans,
            "tar_exps": tar_exps,
            "tar_beta": tar_beta,
            "tar_pose": tar_pose,
            "latent_in":  latent_in,
            "tar_id": tar_id,
            "tar_contact": tar_contact,
            "style_feature":style_feature,
        }
    
    def _g_test(self, loaded_data):
        
        mode = 'test'
        bs, n, j = loaded_data["tar_pose"].shape[0], loaded_data["tar_pose"].shape[1], self.joints 
        tar_pose = loaded_data["tar_pose"]
        tar_beta = loaded_data["tar_beta"]
        tar_exps = loaded_data["tar_exps"]
        tar_contact = loaded_data["tar_contact"]
        tar_trans = loaded_data["tar_trans"]
        in_word = loaded_data["in_word"]
        in_audio = loaded_data["in_audio"]
        in_x0 = loaded_data['latent_in']
        in_seed = loaded_data['latent_in']
        
        remain = n%8
        if remain != 0:
            tar_pose = tar_pose[:, :-remain, :]
            tar_beta = tar_beta[:, :-remain, :]
            tar_trans = tar_trans[:, :-remain, :]
            in_word = in_word[:, :-remain]
            tar_exps = tar_exps[:, :-remain, :]
            tar_contact = tar_contact[:, :-remain, :]
            in_x0 = in_x0[:, :in_x0.shape[1]-(remain//self.args.vqvae_squeeze_scale), :]
            in_seed = in_seed[:, :in_x0.shape[1]-(remain//self.args.vqvae_squeeze_scale), :]
            n = n - remain

        tar_pose_jaw = tar_pose[:, :, 66:69]
        tar_pose_jaw = rc.axis_angle_to_matrix(tar_pose_jaw.reshape(bs, n, 1, 3))
        tar_pose_jaw = rc.matrix_to_rotation_6d(tar_pose_jaw).reshape(bs, n, 1*6)
        tar_pose_face = torch.cat([tar_pose_jaw, tar_exps], dim=2)

        tar_pose_hands = tar_pose[:, :, 25*3:55*3]
        tar_pose_hands = rc.axis_angle_to_matrix(tar_pose_hands.reshape(bs, n, 30, 3))
        tar_pose_hands = rc.matrix_to_rotation_6d(tar_pose_hands).reshape(bs, n, 30*6)

        # tar_pose_upper = tar_pose[:, :, self.joint_mask_upper.astype(bool)]
        # tar_pose_upper = rc.axis_angle_to_matrix(tar_pose_upper.reshape(bs, n, 13, 3))
        # tar_pose_upper = rc.matrix_to_rotation_6d(tar_pose_upper).reshape(bs, n, 13*6)

        # tar_pose_leg = tar_pose[:, :, self.joint_mask_lower.astype(bool)]
        # tar_pose_leg = rc.axis_angle_to_matrix(tar_pose_leg.reshape(bs, n, 9, 3))
        # tar_pose_leg = rc.matrix_to_rotation_6d(tar_pose_leg).reshape(bs, n, 9*6)
        # --- FIX FOR UPPER BODY SLICING ---
        tar_pose_upper = tar_pose[:, :, self.joint_mask_upper.astype(bool)]
        # If the mask resulted in 0 dimensions, force the standard BEAT2 upper body range
        if tar_pose_upper.shape[-1] == 0:
            # Standard BEAT2 upper body is usually 13 joints (39 dims) 
            # adjust indices 36:75 if your skeleton layout differs
            tar_pose_upper = tar_pose[:, :, 36:75] 
        
        tar_pose_upper = rc.axis_angle_to_matrix(tar_pose_upper.reshape(bs, n, 13, 3))
        tar_pose_upper = rc.matrix_to_rotation_6d(tar_pose_upper).reshape(bs, n, 13*6)

        # --- FIX FOR LOWER BODY SLICING ---
        tar_pose_leg = tar_pose[:, :, self.joint_mask_lower.astype(bool)]
        # If the mask resulted in 0 dimensions, force the standard BEAT2 lower body range
        if tar_pose_leg.shape[-1] == 0:
            # Standard BEAT2 legs are usually 9 joints (27 dims)
            tar_pose_leg = tar_pose[:, :, 9:36]

        tar_pose_leg = rc.axis_angle_to_matrix(tar_pose_leg.reshape(bs, n, 9, 3))
        tar_pose_leg = rc.matrix_to_rotation_6d(tar_pose_leg).reshape(bs, n, 9*6)
        #
        tar_pose_lower = torch.cat([tar_pose_leg, tar_trans, tar_contact], dim=2)
        
        tar_pose_6d = rc.axis_angle_to_matrix(tar_pose.reshape(bs, n, 55, 3))
        tar_pose_6d = rc.matrix_to_rotation_6d(tar_pose_6d).reshape(bs, n, 55*6)
        latent_all = torch.cat([tar_pose_6d, tar_trans, tar_contact], dim=-1)
        
        rec_all_face = []
        rec_all_upper = []
        rec_all_lower = []
        rec_all_hands = []
        vqvae_squeeze_scale = self.args.vqvae_squeeze_scale
        roundt = (n - self.args.pre_frames * vqvae_squeeze_scale) // (self.args.pose_length - self.args.pre_frames * vqvae_squeeze_scale)
        remain = (n - self.args.pre_frames * vqvae_squeeze_scale) % (self.args.pose_length - self.args.pre_frames * vqvae_squeeze_scale)
        round_l = self.args.pose_length - self.args.pre_frames * vqvae_squeeze_scale
         

        for i in range(0, roundt):
            in_word_tmp = in_word[:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames * vqvae_squeeze_scale]

            in_audio_tmp = in_audio[:, i*(16000//30*round_l):(i+1)*(16000//30*round_l)+16000//30*self.args.pre_frames * vqvae_squeeze_scale]
            in_id_tmp = loaded_data['tar_id'][:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames]
            in_seed_tmp = in_seed[:, i*(round_l)//vqvae_squeeze_scale:(i+1)*(round_l)//vqvae_squeeze_scale+self.args.pre_frames]
            in_x0_tmp = in_x0[:, i*(round_l)//vqvae_squeeze_scale:(i+1)*(round_l)//vqvae_squeeze_scale+self.args.pre_frames]
            mask_val = torch.ones(bs, self.args.pose_length, self.args.pose_dims+3+4).float().cuda()
            mask_val[:, :self.args.pre_frames, :] = 0.0
            if i == 0:
                in_seed_tmp = in_seed_tmp[:, :self.args.pre_frames, :]
            else:
                in_seed_tmp = last_sample[:, -self.args.pre_frames:, :]

            cond_ = {'y':{}}
            cond_['y']['audio'] = in_audio_tmp
            cond_['y']['audio_onset'] = in_audio_tmp
            cond_['y']['word'] = in_word_tmp
            cond_['y']['id'] = in_id_tmp
            cond_['y']['seed'] =in_seed_tmp
            # cond_['y']['mask'] = (torch.zeros([self.args.batch_size, 1, 1, self.args.pose_length]) < 1).cuda()
            cond_['y']['mask'] = (torch.zeros([bs, 1, 1, self.args.pose_length]) < 1).cuda()
            cond_['y']['style_feature'] = torch.zeros([bs, 512]).cuda()

            shape_ = (bs, 3*128, 1, 32)
            sample = self.model(cond_)['latents']
            print(f"LATENT MIN/MAX: {sample.min().item():.2f} to {sample.max().item():.2f}")
            sample = sample.squeeze().permute(1,0).unsqueeze(0)

            last_sample = sample.clone()
            
            rec_latent_upper = sample[...,:128]
            rec_latent_hands = sample[...,128:2*128]
            rec_latent_lower = sample[...,2*128:]
            
           

            if i == 0:
                rec_all_upper.append(rec_latent_upper)
                rec_all_hands.append(rec_latent_hands)
                rec_all_lower.append(rec_latent_lower)
            else:
                rec_all_upper.append(rec_latent_upper[:, self.args.pre_frames:])
                rec_all_hands.append(rec_latent_hands[:, self.args.pre_frames:])
                rec_all_lower.append(rec_latent_lower[:, self.args.pre_frames:])

        rec_all_upper = torch.cat(rec_all_upper, dim=1) * self.vqvae_latent_scale
        rec_all_hands = torch.cat(rec_all_hands, dim=1) * self.vqvae_latent_scale
        rec_all_lower = torch.cat(rec_all_lower, dim=1) * self.vqvae_latent_scale

        rec_upper = self.vq_model_upper.latent2origin(rec_all_upper)[0]
        print(f"DEBUG: rec_upper max value: {rec_upper.max()}")
        rec_hands = self.vq_model_hands.latent2origin(rec_all_hands)[0]
        rec_lower = self.vq_model_lower.latent2origin(rec_all_lower)[0]
        print(f"DECODED UPPER MIN/MAX: {rec_upper.min().item():.2f} to {rec_upper.max().item():.2f}")
        print(f"DECODED HANDS MIN/MAX: {rec_hands.min().item():.2f} to {rec_hands.max().item():.2f}")
        
        if self.use_trans:
            rec_trans_v = rec_lower[...,-3:]
            rec_trans_v = rec_trans_v * self.trans_std + self.trans_mean
            rec_trans = torch.zeros_like(rec_trans_v)
            rec_trans = torch.cumsum(rec_trans_v, dim=-2)
            rec_trans[...,1]=rec_trans_v[...,1]
            rec_lower = rec_lower[...,:-3]
        
        if self.args.pose_norm:
            rec_upper = rec_upper * self.std_upper + self.mean_upper
            rec_hands = rec_hands * self.std_hands + self.mean_hands
            rec_lower = rec_lower * self.std_lower + self.mean_lower




        n = n - remain
        tar_pose = tar_pose[:, :n, :]
        tar_exps = tar_exps[:, :n, :]
        tar_trans = tar_trans[:, :n, :]
        tar_beta = tar_beta[:, :n, :]


        rec_exps = tar_exps
        #rec_pose_jaw = rec_face[:, :, :6]
        rec_pose_legs = rec_lower[:, :, :54]
        bs, n = rec_pose_legs.shape[0], rec_pose_legs.shape[1]
        rec_pose_upper = rec_upper.reshape(bs, n, 13, 6)
        rec_pose_upper = rc.rotation_6d_to_matrix(rec_pose_upper)#
        rec_pose_upper = rc.matrix_to_axis_angle(rec_pose_upper).reshape(bs*n, 13*3)
        # rec_pose_upper_recover = self.inverse_selection_tensor(rec_pose_upper, self.joint_mask_upper, bs*n)
        rec_pose_lower = rec_pose_legs.reshape(bs, n, 9, 6)
        rec_pose_lower = rc.rotation_6d_to_matrix(rec_pose_lower)
        rec_lower2global = rc.matrix_to_rotation_6d(rec_pose_lower.clone()).reshape(bs, n, 9*6)
        rec_pose_lower = rc.matrix_to_axis_angle(rec_pose_lower).reshape(bs*n, 9*3)
        # rec_pose_lower_recover = self.inverse_selection_tensor(rec_pose_lower, self.joint_mask_lower, bs*n)
        rec_pose_hands = rec_hands.reshape(bs, n, 30, 6)
        rec_pose_hands = rc.rotation_6d_to_matrix(rec_pose_hands)
        rec_pose_hands = rc.matrix_to_axis_angle(rec_pose_hands).reshape(bs*n, 30*3)
        # rec_pose_hands_recover = self.inverse_selection_tensor(rec_pose_hands, self.joint_mask_hands, bs*n)
        # rec_pose = rec_pose_upper_recover + rec_pose_lower_recover + rec_pose_hands_recover
        # --- FIX FOR RECOVERY (INVERSE MAPPING) ---
        
        # 1. Create a temporary mask for upper body recovery if the original is empty
        mask_u_rec = self.joint_mask_upper.copy()
        if mask_u_rec.sum() == 0:
            # Match the indices we used in the slicing step (36:75)
            mask_u_rec[36:75] = 1 
        rec_pose_upper_recover = self.inverse_selection_tensor(rec_pose_upper, mask_u_rec, bs*n)

        # 2. Same for lower body/legs
        mask_l_rec = self.joint_mask_lower.copy()
        if mask_l_rec.sum() == 0:
            # Match the indices we used in the slicing step (9:36)
            mask_l_rec[9:36] = 1
        rec_pose_lower_recover = self.inverse_selection_tensor(rec_pose_lower, mask_l_rec, bs*n)

        # 3. Same for hands
        mask_h_rec = self.joint_mask_hands.copy()
        if mask_h_rec.sum() == 0:
            # Match the indices we used in the slicing step (75:165)
            mask_h_rec[75:165] = 1
        rec_pose_hands_recover = self.inverse_selection_tensor(rec_pose_hands, mask_h_rec, bs*n)

        # The final pose assembly
        rec_pose = rec_pose_upper_recover + rec_pose_lower_recover + rec_pose_hands_recover
        #
        rec_pose[:, 66:69] = tar_pose.reshape(bs*n, 55*3)[:, 66:69]

        rec_pose = rc.axis_angle_to_matrix(rec_pose.reshape(bs*n, j, 3))
        rec_pose = rc.matrix_to_rotation_6d(rec_pose).reshape(bs, n, j*6)
        tar_pose = rc.axis_angle_to_matrix(tar_pose.reshape(bs*n, j, 3))
        tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(bs, n, j*6)

        ##
        # At the end of _g_test, before the return dict:
        # Both are (bs, n, 55*6) rotation_6d at this point
        rec_aa = rc.rotation_6d_to_matrix(rec_pose.reshape(bs*n, 55, 6))
        rec_aa = rc.matrix_to_axis_angle(rec_aa).reshape(bs, n, 55*3)
        tar_aa = rc.rotation_6d_to_matrix(tar_pose.reshape(bs*n, 55, 6))
        tar_aa = rc.matrix_to_axis_angle(tar_aa).reshape(bs, n, 55*3)

        joint_l2 = torch.mean((rec_aa - tar_aa) ** 2).item()
        joint_mse = torch.mean((rec_aa - tar_aa) ** 2, dim=[0,1])  # per-joint MSE
        ##


        
        return {
            'rec_pose': rec_pose,
            'rec_trans': rec_trans,
            'tar_pose': tar_pose,
            'tar_exps': tar_exps,
            'tar_beta': tar_beta,
            'tar_trans': tar_trans,
            'rec_exps': rec_exps,
        }
    # def _g_test(self, loaded_data):
    #     bs, n, j = loaded_data["tar_pose"].shape[0], loaded_data["tar_pose"].shape[1], self.joints
    #     tar_pose = loaded_data["tar_pose"]
    #     tar_beta = loaded_data["tar_beta"]
    #     tar_exps = loaded_data["tar_exps"]
    #     tar_contact = loaded_data["tar_contact"]
    #     tar_trans = loaded_data["tar_trans"]
    #     in_word = loaded_data["in_word"]
    #     in_audio = loaded_data["in_audio"]
    #     in_seed = loaded_data['latent_in']

    #     remain = n % 8
    #     if remain != 0:
    #         tar_pose = tar_pose[:, :-remain, :]
    #         tar_beta = tar_beta[:, :-remain, :]
    #         tar_trans = tar_trans[:, :-remain, :]
    #         in_word = in_word[:, :-remain]
    #         tar_exps = tar_exps[:, :-remain, :]
    #         tar_contact = tar_contact[:, :-remain, :]
    #         in_seed = in_seed[:, :in_seed.shape[1]-(remain//self.args.vqvae_squeeze_scale), :]
    #         n = n - remain

    #     rec_all_upper, rec_all_lower, rec_all_hands = [], [], []
    #     vqvae_squeeze_scale = self.args.vqvae_squeeze_scale
    #     pre_frames_scaled = self.args.pre_frames * vqvae_squeeze_scale
    #     roundt = (n - pre_frames_scaled) // (self.args.pose_length - pre_frames_scaled)
    #     remain = (n - pre_frames_scaled) % (self.args.pose_length - pre_frames_scaled)
    #     round_l = self.args.pose_length - pre_frames_scaled

    #     for i in range(0, roundt):
    #         in_audio_tmp = in_audio[
    #             :, i*(16000//30*round_l):(i+1)*(16000//30*round_l) + 16000//30*pre_frames_scaled
    #         ]
    #         in_word_tmp = in_word[:, i*round_l:(i+1)*round_l + pre_frames_scaled]
    #         in_id_tmp = loaded_data['tar_id'][:, i*round_l:(i+1)*round_l + self.args.pre_frames]
    #         in_seed_tmp = in_seed[
    #             :, i*round_l//vqvae_squeeze_scale:(i+1)*round_l//vqvae_squeeze_scale + self.args.pre_frames
    #         ]

    #         if i == 0:
    #             in_seed_tmp = in_seed_tmp[:, :self.args.pre_frames, :]
    #         else:
    #             in_seed_tmp = last_sample[:, -self.args.pre_frames:, :]

    #         # ← NO mask key, NO audio key, correct bs for style_feature
    #         cond_ = {'y': {}}
    #         cond_['y']['audio_onset'] = in_audio_tmp
    #         cond_['y']['word'] = in_word_tmp
    #         cond_['y']['id'] = in_id_tmp
    #         cond_['y']['seed'] = in_seed_tmp
    #         cond_['y']['style_feature'] = torch.zeros([bs, 512]).cuda()

    #         sample = self.model(cond_)['latents']
    #         sample = sample.squeeze(2).permute(0, 2, 1)  # ← fixed permute

    #         last_sample = sample.clone()

    #         code_dim = self.vq_model_upper.code_dim
    #         rec_latent_upper = sample[..., :code_dim]
    #         rec_latent_hands = sample[..., code_dim:code_dim*2]
    #         rec_latent_lower = sample[..., code_dim*2:code_dim*3]

    #         if i == 0:
    #             rec_all_upper.append(rec_latent_upper)
    #             rec_all_hands.append(rec_latent_hands)
    #             rec_all_lower.append(rec_latent_lower)
    #         else:
    #             rec_all_upper.append(rec_latent_upper[:, self.args.pre_frames:])
    #             rec_all_hands.append(rec_latent_hands[:, self.args.pre_frames:])
    #             rec_all_lower.append(rec_latent_lower[:, self.args.pre_frames:])

    #     rec_all_upper = torch.cat(rec_all_upper, dim=1) * self.vqvae_latent_scale
    #     rec_all_hands = torch.cat(rec_all_hands, dim=1) * self.vqvae_latent_scale
    #     rec_all_lower = torch.cat(rec_all_lower, dim=1) * self.vqvae_latent_scale

    #     rec_upper = self.vq_model_upper.latent2origin(rec_all_upper)[0]
    #     rec_hands = self.vq_model_hands.latent2origin(rec_all_hands)[0]
    #     rec_lower = self.vq_model_lower.latent2origin(rec_all_lower)[0]

    #     if self.use_trans:
    #         rec_trans_v = rec_lower[..., -3:]
    #         rec_trans_v = rec_trans_v * self.trans_std + self.trans_mean
    #         rec_trans = torch.cumsum(rec_trans_v, dim=-2)
    #         rec_trans[..., 1] = rec_trans_v[..., 1]
    #         rec_lower = rec_lower[..., :-3]

    #     if self.args.pose_norm:
    #         rec_upper = rec_upper * self.std_upper + self.mean_upper
    #         rec_hands = rec_hands * self.std_hands + self.mean_hands
    #         rec_lower = rec_lower * self.std_lower + self.mean_lower

    #     n = n - remain
    #     tar_pose = tar_pose[:, :n, :]
    #     tar_exps = tar_exps[:, :n, :]
    #     tar_trans = tar_trans[:, :n, :]
    #     tar_beta = tar_beta[:, :n, :]

    #     rec_exps = tar_exps
    #     rec_pose_legs = rec_lower[:, :, :54]
    #     bs, n = rec_pose_legs.shape[0], rec_pose_legs.shape[1]

    #     rec_pose_upper = rc.rotation_6d_to_matrix(rec_upper.reshape(bs, n, 13, 6))
    #     rec_pose_upper = rc.matrix_to_axis_angle(rec_pose_upper).reshape(bs*n, 13*3)
    #     rec_pose_upper_recover = self.inverse_selection_tensor(rec_pose_upper, self.joint_mask_upper, bs*n)

    #     rec_pose_lower = rc.rotation_6d_to_matrix(rec_pose_legs.reshape(bs, n, 9, 6))
    #     rec_pose_lower = rc.matrix_to_axis_angle(rec_pose_lower).reshape(bs*n, 9*3)
    #     rec_pose_lower_recover = self.inverse_selection_tensor(rec_pose_lower, self.joint_mask_lower, bs*n)

    #     rec_pose_hands = rc.rotation_6d_to_matrix(rec_hands.reshape(bs, n, 30, 6))
    #     rec_pose_hands = rc.matrix_to_axis_angle(rec_pose_hands).reshape(bs*n, 30*3)
    #     rec_pose_hands_recover = self.inverse_selection_tensor(rec_pose_hands, self.joint_mask_hands, bs*n)

    #     rec_pose = rec_pose_upper_recover + rec_pose_lower_recover + rec_pose_hands_recover
    #     rec_pose[:, 66:69] = tar_pose.reshape(bs*n, 55*3)[:, 66:69]

    #     rec_pose = rc.axis_angle_to_matrix(rec_pose.reshape(bs*n, j, 3))
    #     rec_pose = rc.matrix_to_rotation_6d(rec_pose).reshape(bs, n, j*6)
    #     tar_pose = rc.axis_angle_to_matrix(tar_pose.reshape(bs*n, j, 3))
    #     tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(bs, n, j*6)

    #     return {
    #         'rec_pose': rec_pose,
    #         'rec_trans': rec_trans,
    #         'tar_pose': tar_pose,
    #         'tar_exps': tar_exps,
    #         'tar_beta': tar_beta,
    #         'tar_trans': tar_trans,
    #         'rec_exps': rec_exps,
    #     }

    def test_demo(self, epoch):
        '''
        input audio and text, output motion
        do not calculate loss and metric
        save video
        '''
        results_save_path = self.checkpoint_path + f"/{epoch}/"
        if os.path.exists(results_save_path): 
            import shutil
            shutil.rmtree(results_save_path)
        os.makedirs(results_save_path)
        start_time = time.time()
        total_length = 0
        self.model.eval()
        self.smplx.eval()
        # print("AAA")
        # print("self.test_loader",self.test_loader)
        # print("len(self.test_loader:)",len(self.test_loader))
        # self.eval_copy.eval()
        render_vid_path = None
        with torch.no_grad():
            for its, batch_data in enumerate(self.test_loader):
                # print("bbbb")
                ####
                # if 'pose' in batch_data:
                #     batch_data['pose'] = torch.zeros_like(batch_data['pose'])
                print(f"AUDIO MAX: {torch.max(batch_data['audio_onset'])}")
                print(f"AUDIO MIN: {torch.min(batch_data['audio_onset'])}")
                print(f"FAKE POSE SEED MAX: {torch.max(batch_data.get('pose', torch.tensor([0])))}")
                ####
                loaded_data = self._load_data(batch_data)    
                net_out = self._g_test(loaded_data)
                # After net_out = self._g_test(loaded_data), add:
                joint_l2_loss = net_out.get('joint_l2', 0)
                logger.info(f"Test Joint L2 Loss: {joint_l2_loss:.6f}")
                np.save(results_save_path + "test_joint_l2.npy", np.array([joint_l2_loss]))
                ##
                tar_pose = net_out['tar_pose']
                rec_pose = net_out['rec_pose']
                tar_exps = net_out['tar_exps']
                tar_beta = net_out['tar_beta']
                rec_trans = net_out['rec_trans']
                tar_trans = net_out['tar_trans']
                rec_exps = net_out['rec_exps']
                bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints
                if (30/self.args.pose_fps) != 1:
                    assert 30%self.args.pose_fps == 0
                    n *= int(30/self.args.pose_fps)
                    tar_pose = torch.nn.functional.interpolate(tar_pose.permute(0, 2, 1), scale_factor=30/self.args.pose_fps, mode='linear').permute(0,2,1)
                    rec_pose = torch.nn.functional.interpolate(rec_pose.permute(0, 2, 1), scale_factor=30/self.args.pose_fps, mode='linear').permute(0,2,1)
                

                rec_pose = rc.rotation_6d_to_matrix(rec_pose.reshape(bs*n, j, 6))
                rec_pose = rc.matrix_to_rotation_6d(rec_pose).reshape(bs, n, j*6)
                tar_pose = rc.rotation_6d_to_matrix(tar_pose.reshape(bs*n, j, 6))
                tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(bs, n, j*6)

                rec_pose = rc.rotation_6d_to_matrix(rec_pose.reshape(bs*n, j, 6))
                rec_pose = rc.matrix_to_axis_angle(rec_pose).reshape(bs*n, j*3)
                tar_pose = rc.rotation_6d_to_matrix(tar_pose.reshape(bs*n, j, 6))
                tar_pose = rc.matrix_to_axis_angle(tar_pose).reshape(bs*n, j*3)
                

                tar_pose_np = tar_pose.detach().cpu().numpy()
                rec_pose_np = rec_pose.detach().cpu().numpy()
                rec_trans_np = rec_trans.detach().cpu().numpy().reshape(bs*n, 3)
                rec_exp_np = rec_exps.detach().cpu().numpy().reshape(bs*n, 100) 
                tar_exp_np = tar_exps.detach().cpu().numpy().reshape(bs*n, 100)
                tar_trans_np = tar_trans.detach().cpu().numpy().reshape(bs*n, 3)
                gt_npz = np.load("./demo/examples/2_scott_0_1_1.npz", allow_pickle=True)

                results_npz_file_save_path = results_save_path+f"result_{self.time_name_expend}"+'.npz'
                np.savez(results_npz_file_save_path,
                    betas=gt_npz["betas"],
                    poses=rec_pose_np,
                    expressions=rec_exp_np,
                    trans=rec_trans_np,
                    model='smplx2020',
                    gender='neutral',
                    mocap_frame_rate = 30,
                )
                total_length += n
                render_vid_path = other_tools_hf.render_one_sequence_no_gt(
                    results_npz_file_save_path, 
                    #results_save_path+"gt_"+test_seq_list.iloc[its]['id']+'.npz', 
                    results_save_path,
                    self.audio_path,
                    self.args.data_path_1+"smplx_models/",
                    use_matplotlib = False,
                    args = self.args,
                    )

        # result = [
        #     gr.Video(value=render_vid_path, visible=True),
        #     gr.File(value=results_npz_file_save_path, label="download motion and visualize in blender"),
        #     ]
        #
        # if render_vid_path is None:
        #     logger.error("Rendering failed - render_vid_path is None")
        #     return [None, None]
        if render_vid_path is None:
            print("ERROR: Dataloader is empty! Audio was filtered out.")
            return gr.Video(visible=False), None
        # print("cccc")
        result = [
        gr.Video(value=render_vid_path, visible=True),
        gr.File(value=results_npz_file_save_path, label="download motion and visualize in blender"),
        ]
        #
        end_time = time.time() - start_time
        logger.info(f"total inference time: {int(end_time)} s for {int(total_length/self.args.pose_fps)} s motion")
        return result
       
# @logger.catch
# def gesturelsm(audio_path,sample_stratege):
#     # args, cfg = config.parse_args()
#     args, cfg = config.parse_args(config_path="configs_new/meanflow_rvqvae_128.yaml")
    
#     # Force the checkpoint to your best one
#     args.test_ckpt = "./outputs/weights/meanflow_rvqvae_128/best_fgd/ckpt.pth"
    
#     print(sample_stratege) nnn 
# @logger.catch
# def gesturelsm(audio_path, sample_stratege):
#     # 1. Setup the Configs
#     config_file = "configs_new/meanflow_rvqvae_128.yaml"
#     args, cfg = config.parse_args(config_path=config_file)
#     # args.test_ckpt = "./outputs/weights_25pct/meanflow_rvqvae_128/best_fgd/ckpt.pth"
#     # These match the names in your screenshot
#     args.mean_trans_path = "./mean_std/beatx_2_trans_mean.npy"
#     args.std_trans_path = "./mean_std/beatx_2_trans_std.npy"

#     # These are usually also required for the body pose normalization
#     args.mean_path = "./mean_std/beatx_2_330_mean.npy"
#     args.std_path = "./mean_std/beatx_2_330_std.npy"
#     # --------------------------------------
#     args.mean_pose_path = "./mean_std/beatx_2_330_mean.npy"
#     args.std_pose_path  = "./mean_std/beatx_2_330_std.npy"
#     ###
#     #args.emo_rep = None
#     ###
#     args.test_clip = False  # The dataloader is looking for this
#     args.onset_rep = True   # Common missing attribute in this dataset
#     args.use_trans = True
#     # 2. Fix the VQ-VAE config error and point to your trained weights
#     args.cfg = config_file 
#     args.test_ckpt = "./outputs/weights/meanflow_rvqvae_128/best_fgd/ckpt.pth"
#     args.dataset = "beat_sep_lower"
#     print(f"Strategy selected: {sample_stratege}")

#     # 3. CONVERT AUDIO PATH TO DATA TUPLE (Fixes the IndexError)
#     # Gradio gives a string path; BaseTrainer wants (sample_rate, numpy_array)
#     import librosa
#     audio_data, sr = librosa.load(audio_path, sr=16000) 
#     ap_tuple = (sr, audio_data) 

#     # 4. Environment & Seed setup
#     if not sys.warnoptions:
#         warnings.simplefilter("ignore")
#     other_tools_hf.set_random_seed(args)
#     other_tools_hf.print_exp_info(args)

#     # 5. Initialize Trainer with the tuple 'ap_tuple'
#     trainer = BaseTrainer(args, cfg, ap = ap_tuple)
    
#     # # 6. Load your trained weights with a key-fix
#     # checkpoint = torch.load(args.test_ckpt, map_location='cuda')
    
#     # # If your checkpoint uses 'model_state_dict' instead of 'model_state', we swap it
#     # if 'model_state_dict' in checkpoint and 'model_state' not in checkpoint:
#     #     checkpoint['model_state'] = checkpoint.pop('model_state_dict')
    
#     # # Now we save it to a temporary location or pass the modified dict if load_checkpoints allows
#     # # But the easiest way is to modify the loader call:
#     # other_tools.load_checkpoints(trainer.model, args.test_ckpt, args.g_name)

#     # 6. Load your trained weights (BULLETPROOF METHOD)
#     logger.info(f"Loading weights from: {args.test_ckpt}")
#     checkpoint = torch.load(args.test_ckpt, map_location='cuda')
    
#     # Find the exact key where the brain is stored
#     if 'model_state_dict' in checkpoint:
#         state_dict = checkpoint['model_state_dict']
#     elif 'model_state' in checkpoint:
#         state_dict = checkpoint['model_state']
#     elif 'net' in checkpoint:
#         state_dict = checkpoint['net']
#     else:
#         state_dict = checkpoint # Fallback
        
#     # --- ADDED THIS TO SCRUB THE 'module.' PREFIX ---
#     from collections import OrderedDict
#     clean_state_dict = OrderedDict()
#     for k, v in state_dict.items():
#         name = k[7:] if k.startswith('module.') else k
#         clean_state_dict[name] = v
#     # ------------------------------------------------
        
#     # Force the weights into the DataParallel model
#     if hasattr(trainer.model, 'module'):
#         trainer.model.module.load_state_dict(clean_state_dict, strict=True)
#     else:
#         trainer.model.load_state_dict(clean_state_dict, strict=True)
        
#     logger.info("Weights successfully loaded into the model!")

#     # # 6. Load your specific trained weights into the model
#     # other_tools.load_checkpoints(trainer.model, args.test_ckpt, args.g_name)
    
#     # 7. Run the inference and return
#     result = trainer.test_demo(999)
#     return result
@logger.catch
def gesturelsm(audio_path, sample_stratege):
    # 1. HARDCODED Configs
    config_file = "configs_new/shortcut_rvqvae_128.yaml"
    args, cfg = config.parse_args(config_path=config_file)
    
    # --- SAFETY NET: Force it to read the model_type so it doesn't crash ---
    if hasattr(cfg, 'data'):
        for k, v in cfg.data.items():
            setattr(args, k, v)
    if not hasattr(args, 'model_type') or not args.model_type:
        args.model_type = 'smplx'
    # -----------------------------------------------------------------------
    
    # 2. HARDCODED Paths
    args.mean_trans_path = "./mean_std/beatx_2_trans_mean.npy"
    args.std_trans_path = "./mean_std/beatx_2_trans_std.npy"
    args.mean_path = "./mean_std/beatx_2_330_mean.npy"
    args.std_path = "./mean_std/beatx_2_330_std.npy"
    args.mean_pose_path = "./mean_std/beatx_2_330_mean.npy"
    args.std_pose_path  = "./mean_std/beatx_2_330_std.npy"
    
    args.test_clip = False  
    args.onset_rep = True   
    args.use_trans = True
    args.dataset = "beat_sep_lower"
    
    # 3. HARDCODED Checkpoint to your 50% model
    args.cfg = config_file 
    args.test_ckpt = "./outputs/weights_50pct/shortcut_rvqvae_128/best_fgd/ckpt.pth"
    
    print(f"Strategy selected: {sample_stratege}")

    # 4. CONVERT AUDIO PATH TO DATA TUPLE
    import librosa
    audio_data, sr = librosa.load(audio_path, sr=16000) 
    ap_tuple = (sr, audio_data) 

    # 5. Environment & Seed setup
    import sys
    import warnings
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    other_tools_hf.set_random_seed(args)
    other_tools_hf.print_exp_info(args)

    # 6. Initialize Trainer
    trainer = BaseTrainer(args, cfg, ap=ap_tuple)
    
    # 7. BULLETPROOF LOADER
    logger.info(f"Loading weights from: {args.test_ckpt}")
    checkpoint = torch.load(args.test_ckpt, map_location='cuda')
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    elif 'net' in checkpoint:
        state_dict = checkpoint['net']
    else:
        state_dict = checkpoint # Fallback
        
    from collections import OrderedDict
    clean_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        clean_state_dict[name] = v
        
    if hasattr(trainer.model, 'module'):
        trainer.model.module.load_state_dict(clean_state_dict, strict=True)
    else:
        trainer.model.load_state_dict(clean_state_dict, strict=True)
        
    logger.info("Weights successfully loaded into the model!")
    
    # 8. Run the inference and return
    result = trainer.test_demo(999)
    return result


    # # Use your main config for everything
    # config_file = "configs_new/meanflow_rvqvae_128.yaml"
    # args, cfg = config.parse_args(config_path=config_file)
    
    # # CRITICAL: Tell the VQ-VAE initialization to use the same config file
    # args.cfg = config_file 
    
    # # Point to your best weights
    # args.test_ckpt = "./outputs/weights/meanflow_rvqvae_128/best_fgd/ckpt.pth"
    # print(sample_stratege)
    # trainer = BaseTrainer(args, cfg, ap = audio_path)
    # #os.environ['TRANSFORMERS_CACHE'] = args.data_path_1 + "hub/"
    # if not sys.warnoptions:
    #     warnings.simplefilter("ignore")
    # # dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    # #logger_tools.set_args_and_logger(args, rank)
    # other_tools_hf.set_random_seed(args)
    # other_tools_hf.print_exp_info(args)

    # # return one intance of trainer
    # #trainer = BaseTrainer(args, cfg, ap = audio_path)
    # other_tools.load_checkpoints(trainer.model, args.test_ckpt, args.g_name)
    
    # result = trainer.test_demo(999)
    # return result

examples = [
    ["demo/examples/2_scott_0_1_1.wav"],
    ["demo/examples/2_scott_0_2_2.wav"],
    ["demo/examples/2_scott_0_3_3.wav"],
    ["demo/examples/2_scott_0_4_4.wav"],
    ["demo/examples/2_scott_0_5_5.wav"],
]

# demo = gr.Interface(
#     gesturelsm,  # function
#     inputs=[
#         gr.Audio(),
#     ],  # input type
#     outputs=[
#         gr.Video(format="mp4", visible=True),
#         gr.File(label="download motion and visualize in blender")
#     ],
#     title='GestureLSM: Latent Shortcut based Co-Speech Gesture Generation with Spatial-Temporal Modeling',
#     description="1. Upload your audio.  <br/>\
#         2. Then, sit back and wait for the rendering to happen! This may take a while (e.g. 1-4 minutes) <br/>\
#         3. After, you can view the videos.  <br/>\
#         4. Notice that we use a fix face animation, our method only produce body motion. <br/>\
#         5. Use DDPM sample strategy will generate a better result, while it will take more inference time.  \
#             ",
#     article="Project links: [GestureLSM](https://github.com/andypinxinliu/GestureLSM). <br/>\
#              Reference links: [EMAGE](https://pantomatrix.github.io/EMAGE/). ", 
#     examples=examples,
# )


demo = gr.Interface(
    gesturelsm, 
    inputs=[
        gr.Audio(type="filepath"),
        gr.Radio(["fast", "ddpm"], value="fast", label="Sample Strategy")
    ],
    outputs=[
        gr.Video(format="mp4", visible=True),
        gr.File(label="download motion and visualize in blender")
    ],
    title='GestureLSM Demo',
    description="Upload audio to generate gestures.",
    examples=examples,
)

            
if __name__ == "__main__":
    os.environ["MASTER_ADDR"]='127.0.0.3'
    os.environ["MASTER_PORT"]='8678'
    #os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    demo.launch(server_name="0.0.0.0",share=True)