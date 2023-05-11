import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from lang_module.config import reward_key
from lang_module.utils import gen_embedding
from collections import deque
import numpy as np
from array2gif import write_gif
from torchvision import models


class Lan_Encoder(nn.Module):
    # nature paper architecture
    def __init__(self, seq_len=4, hidden_size=32, num_layers=2, embedding_dim=50):
        super(Lan_Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(seq_len*hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            # nn.ReLU(),
        )
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = torch.flatten(x, 1, 2)
        # print(f"x shape is {x.shape}")
        x = self.mlp(x)
        return x

# method 1 cnn + lstm , 2 3d卷积
class Clip_Encoder(nn.Module): 
    def __init__(self, in_channels=1):
        super(Clip_Encoder, self).__init__()
        # self.image_conv = nn.Sequential(
        #     nn.Conv2d(3, 16, (2, 2)),
        #     nn.ReLU(),
        #     nn.MaxPool2d((2, 2)),
        #     nn.Conv2d(16, 32, (2, 2)),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, (2, 2)),
        #     nn.ReLU(),
        #     nn.Flatten(1, -1)
        # )
        self.image_conv = models.resnet18(pretrained=True)
        del self.image_conv.fc
        # self.image_conv.fc = torch.nn.Linear(256, 256)
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=2)
        self.fc1 = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128)
        )


    def forward(self, x_3d):
        hidden = None
        # x_3d = x_3d.unsqueeze(2)
        # print(f"x_3d shape is {x_3d.shape}")
        features = []
        for i in range(x_3d.size(1)):
        
            with torch.no_grad():
                x = self.image_conv(x_3d[:, i, :, :, :])
                # print(f"cnn out shape is {x.shape}")
                features.append(x)
                # print(f"features shape is {x.shape}")
                # out, hidden = self.lstm(x.unsqueeze(0), hidden)
        # features = torch.stack(features)
        # out, (h_n, c_n) = self.lstm(features)
        # out = out.permute(1, 0, 2)
        # out = torch.flatten(out, 1, 2)
        x = torch.mean(torch.stack(features), dim=0)
        # x = self.fc1(out[-1, : , :])
        # print(x.shape)
        x = self.fc1(x)
        # x = F.relu(x)
        return x


class Contras_Model(nn.Module):
    def __init__(self, ):
        super(Contras_Model, self).__init__()

        self.lang_encoder = Lan_Encoder()
        self.clip_encoder = Clip_Encoder()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, x_lang, x_clip):
        lang_feature = self.lang_encoder(x_lang)
        clip_feature = self.clip_encoder(x_clip)

        # normalized features
        lang_feature = lang_feature / lang_feature.norm(dim=1, keepdim=True)
        clip_feature = clip_feature / clip_feature.norm(dim=1, keepdim=True)

        # consine similarity
        logits_per_clip = self.logit_scale * clip_feature @ lang_feature.t()
        logits_per_lang = logits_per_clip.t()

        return logits_per_clip, logits_per_lang

    def lang_code(self, x_lang):
        return self.lang_encoder(x_lang)

    def clip_encode(self, x_clip):
        return self.clip_encoder(x_clip)
    

class Reward_Model():
    def __init__(self, lang_path, threshold=0.95, glove_path='glove.6B.50d.txt', device="cuda:0"):
        self.device = device
        self.encoder = Contras_Model()
        self.encoder.load_state_dict(torch.load(lang_path)['model'])
        self.encoder.to(device)
        self.encoder.eval()
        self.sim_reward = []
        self.instructions = []
        self.used_ins = []

        self.lang_features = []
        for key in reward_key:
            emb = gen_embedding(glove_path, key).to(device).unsqueeze(0)
            print(f"emb shape is {emb.shape}")
            with torch.no_grad():
                feature = self.encoder.lang_code(emb)
            self.lang_features.append(feature.squeeze(0))
            self.sim_reward.append(reward_key[key])
            self.instructions.append(key)
        print(self.instructions)
        self.lang_features = torch.stack(self.lang_features)
        self.lang_features = self.lang_features / self.lang_features.norm(dim=1, keepdim=True )
        self.clip = deque(maxlen=2)
        self.threshold = threshold

        self.visual_clip = deque(maxlen=2)

    def add_obs(self, obs):
        self.clip.append(obs)
        # if frame:
        #     self.visual_clip.append(np.moveaxis(frame, 2, 0))

    def add_frame(self, frame):
        self.visual_clip.append(np.moveaxis(frame, 2, 0))

    def cal_reward(self, done=False, reward=None):
        shape_reward = 0
        val = 0
        if len(self.clip) >= 2:
            seqs = torch.tensor(np.asarray(self.clip), dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                    clip_feature = self.encoder.clip_encode(seqs).squeeze(0) # cal consine similarity
            clip_feature = clip_feature / clip_feature.norm()
            logits_per_lang = self.lang_features @ clip_feature.t()
            val = torch.max(logits_per_lang)
            # val = logits_per_lang[-1]
            # print(f"consin similarity is {val}")
            if val > self.threshold:
                # self.visual_frames()
                # print('reach goal')
                idx = torch.argmax(logits_per_lang)
                # if idx not in self.used_ins:
                #     if idx != 5:
                #         self.used_ins.append(idx)
                    # if reward > 0:
                    #     print(f"get true instruction : {self.instructions[idx]}")
                    # else:
                    #     print(f"get false instruction : {self.instructions[idx]} , reward {reward}")
                shape_reward =  self.sim_reward[idx]
        if done:
            print(f"game done, reward is {reward} , val is {val}, clip length is {len(self.clip)}, shape_reward is {shape_reward}")

            if reward > 0 and shape_reward < 1:
                np.save("biclip/nomatch.npy", self.clip)
                print("--------------------------------------- match wrong--------------------------------------------------")
            # self.used_ins.clear()
            # self.clip.clear()
        return shape_reward
            
    def visual_frames(self,):
        write_gif(np.array(self.visual_clip), f"figures/{np.random.randint(1, 10000)}.gif", fps=1)


class Parallel_Reward_Models():
    def __init__(self, n_procs, lang_path = "checkpoints/new_lang_model/encoder_epoch_280.pth") -> None:
        self.reward_models = []
        self.n_procs = n_procs
        for i in range(n_procs):
            self.reward_models.append(Reward_Model(lang_path))

    def add_obss(self, obss):
        # print(obss)

        assert len(obss) == self.n_procs
        for i in range(self.n_procs):
            self.reward_models[i].add_obs(obss[i]['image'])
    
    def add_frames(self, frames):
        assert len(frames) == self.n_procs
        for i in range(self.n_procs):
            self.reward_models[i].add_frame(frames[i])
            
    def get_rewards(self, done, reward=None):
        rewards = []
        for i in range(self.n_procs):
            rewards.append(self.reward_models[i].cal_reward(done[i], reward[i]))
        return rewards






        
