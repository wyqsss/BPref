import torch
from byol_pytorch import BYOL
from torchvision import models
from lang_clip_dataset import Image_DataSet

resnet = models.resnet18(pretrained=True)

learner = BYOL(
    resnet,
    image_size = 256,
    hidden_layer = 'avgpool'
)

opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

def sample_unlabelled_images():
    return torch.randn(20, 3, 256, 256)

image_dateset = Image_DataSet("p2anotation.csv")
print("load dataset")
image_loader = torch.utils.data.DataLoader(image_dateset, batch_size=32, shuffle=True, num_workers=4 )
print(f"load dataloader length {len(image_loader)}")
for i in range(100):
    # images = sample_unlabelled_images()
    avg_loss = 0
    for idx, batch in enumerate(image_loader):
        loss = learner(batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        learner.update_moving_average() # update moving average of target encoder
        avg_loss += loss.item()
    print(f"Training epoch {i} average loss is : {avg_loss / len(image_loader)}")
    

# save your improved network
torch.save(resnet.state_dict(), './improved-net.pt')