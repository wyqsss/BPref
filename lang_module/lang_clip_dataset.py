import torch
from torch.utils.data import Dataset, DataLoader
import imageio
import cv2
import numpy as np
import random


raw_data = "np_data/episode_0.npy"
raw_data1 = "np_data/episode_2.npy"
data = {
    "Go up when the key is up to you": 0,
    "Pick up the key when the key is near you": 6,
    "Open the door with key when the door is near you": 9,
    "Go down indoor when the goal is down to you": 14,
    "Reach the Goal": 17,
    "Go right indoor when the goal is right to you": 12,
    #no train
    # "Go right indoor when the goal is right to you": 11,
    # "Go up with key when the door is up to you": 7,
    # "Go down when the key is down to you": ,
}

class Lang_Clip_DataSet(Dataset):
    def __init__(self, glove_path, csv_path):
        super(Lang_Clip_DataSet, self).__init__()

        self.dictionary = {}
        with open(glove_path) as f:
            for line in f.readlines():
                line = line.strip()
                parts = line.split()
                self.dictionary[parts[0]] = np.array(list(map(eval, parts[1:])))

        self.annotation = np.loadtxt(open(csv_path, 'rb'), delimiter=',', dtype=str)

        self.data = []
        for item in self.annotation:
            if item[1] == "others of the door":
                if random.random() > 0.25:
                    continue
            sent_emb = self.generate_embeddings(item[1])
            seq = self.generate_images(item[0])
            self.data.append((sent_emb, seq))



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1]   #sentence embedding, image sequence, label

    def generate_embeddings(self, sentence): # [sent_len, emb_size]
        sent_emb = []
        for w in sentence.split():
            try:
                sent_emb.append(self.dictionary[w])
            except KeyError:
                print(f"do not exist key {w}")
                sent_emb.append(np.zeros(50))
        for i in range(len(sent_emb), 4):
            sent_emb.append(np.zeros(50))
        
        return torch.tensor(np.asarray(sent_emb), dtype=torch.float32)
    
    def generate_images(self, video_path):
        # vid = imageio.get_reader(video_path, "ffmpeg")
        # print(video_path)
        images = []
        capture = cv2.VideoCapture(video_path)

        while True:
            ret,img=capture.read() # img 就是一帧图片  
            if not ret:
                break # 当获取完最后一帧就结束   
            # print(img.shape)       
            # 可以用 cv2.imshow() 查看这一帧，也可以逐帧保存
            images.append(img)

        images = np.array(images).transpose(0, 3, 1, 2)
        # print(f"sequence shape is {images.shape}")
        return torch.tensor(images, dtype=torch.float32)  # [B, C, H, W]，


    
    # def load_annotations_file(self, filename):
    #     clips = []
    #     sentences = []
    #     # translator = str.maketrans('', '', string.punctuation)
    #     with open(filename) as f:
    #         for line in f.readlines():
    #             line = line.strip()
    #             sentence, video_path = line.split(';')
    #             sentence = sentence.lower()
    #             # sentence = sentence.translate(translator)
    #             st_embd = self.generate_embeddings(sentence)
    #             # print(f"sent embbed shape is {st_embd.shape}")
    #             seq = self.generate_images(video_path)
    #             self.pos_samples.append((sentence, st_embd, seq)) #  positive sample

    # def create_neg_samples(self):
    #     for i in range(len(self.pos_samples)):
    #         self.data.append((self.pos_samples[i][1], self.pos_samples[i][2]))  # positive samples
    #         self.labels.append(torch.tensor([1, 0], dtype=torch.float32))





if __name__ == "__main__":
    data_set = Lang_Clip_DataSet(glove_path="glove.6B.50d.txt", csv_path="p2anotation.csv")
    print(len(data_set))