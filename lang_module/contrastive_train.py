from encoder import Contras_Model
from lang_clip_dataset import Lang_Clip_DataSet
import argparse
import os
import torch
import numpy as np

def main(args):
    full_dataset = Lang_Clip_DataSet(args.glove, csv_path="p2anotation.csv")
    train_loader = torch.utils.data.DataLoader(full_dataset, batch_size=args.batch_size, shuffle=True)

    device = args.device
    model = Contras_Model()
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)
    criterion = torch.nn.CrossEntropyLoss()


    for i in range(args.epochs):

        avg_loss = 0
        for idx, batch in enumerate(train_loader):
            sent_embs, seqs = batch
            sent_embs = sent_embs.to(device)
            seqs = seqs.to(device)
            labels = torch.tensor(np.arange(len(seqs))).to(device)
            logits_per_clip , logits_per_lang = model(sent_embs, seqs)
            # print(labels.shape)
            # print(logits_per_clip.shape)

            loss_c = criterion(logits_per_clip, labels)
            loss_l = criterion(logits_per_lang, labels)

            loss = (loss_c + loss_l) / 2

            avg_loss += loss.item()
            # print(f"Training epoch {i}, step {idx} loss is {loss.item()}")
            optim.zero_grad()
            loss.backward()
            optim.step()
        print(f"Training epoch {i} average loss is : {avg_loss / len(train_loader)}")
        if i % args.save_internal == 0:
                torch.save({'model': model.state_dict()}, os.path.join(args.save_path, f"encoder_epoch_{i}.pth"))
    
def evaluate(args):
    device = args.device
    model = Contras_Model()
    model.load_state_dict(torch.load(args.load_model)['model'])
    model = model.to(device)

    seqs = []
    for i in range(10):
        arrs = np.load("8x8episode_0.npy")
        seq = arrs[-2:]
        seqs.append(seq)
    goal8 = torch.tensor(seqs, dtype=torch.float32).to(device)
    print(goal8.shape)

    # sent_emb = gen_embedding("./glove.6B.50d.txt", "")
    full_dataset = Lang_Clip_DataSet(args.glove)
    val_loader = torch.utils.data.DataLoader(full_dataset, batch_size=args.batch_size)
    for idx, batch in enumerate(val_loader):
        sent_embs, seqs = batch
        sent_embs = sent_embs.to(device)
        seqs = seqs.to(device)
        with torch.no_grad():
            logits_per_clip , logits_per_lang = model(sent_embs, seqs)
        # print(logits_per_clip)
        for lang in logits_per_lang:
            print(lang.cpu())

        # with torch.no_grad():
        #     logits_per_clip , logits_per_lang = model(sent_embs, goal8)
        # # print(logits_per_clip)
        # for lang in logits_per_lang:
        #     print(lang.cpu())

        

# def gen_embedding(glove_path, sentence):

#     self.dictionary = {}
#         with open(glove_path) as f:
#             for line in f.readlines():
#                 line = line.strip()
#                 parts = line.split()
#                 self.dictionary[parts[0]] = np.array(list(map(eval, parts[1:])))

#     sent_emb = []
#         for w in sentence.split():
#             try:
#                 sent_emb.append(self.dictionary[w])
#             except KeyError:
#                 print(f"do not exist key {w}")
#                 sent_emb.append(np.zeros(50))
#         for i in range(len(sent_emb), 14):
#             sent_emb.append(np.zeros(50))
#     return torch.tensor(np.asarray(sent_emb), dtype=torch.float32)




if __name__ == "__main__":
    
    args = argparse.ArgumentParser()
    
    # set hyperparameter
    args.add_argument('--annotation', type=str, default='./annotation.txt')
    args.add_argument('--glove', type=str, default='./glove.6B.50d.txt')
    args.add_argument('--batch_size', type=int, default=64)
    args.add_argument('--lr', type=float, default=0.05)
    args.add_argument('--epochs', type=int, default=101)   
    args.add_argument('--eval_internal', type=int, default=5) 
    args.add_argument('--save_internal', type=int, default=20) 
    
    args.add_argument('--save_path', type=str, default='./checkpoints/goal_lang_model')
    args.add_argument('--device', type=str, default='cuda:0')

    args.add_argument('--load_model', type=str, default=None)

    pars = args.parse_args()


    if pars.load_model:
        evaluate(pars)
    else:
        main(pars)

