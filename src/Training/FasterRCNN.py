import __init__
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchsummary import summary
# from torchinfo import summary
import torchvision
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
import random
from tqdm import tqdm
import torch
from matplotlib import pyplot as plt
import visualize_image_mask as vim
import datetime
import pandas as pd
import csv


class DeepLesion(Dataset):
    def __init__(self, images_filepaths, transform=None):
        self.images_filepaths = images_filepaths
        self.transform = transform

    def __len__(self):
        return len(self.images_filepaths)

    def __getitem__(self, idx):
        image_filepath = self.images_filepaths[idx]
        image_filename = image_filepath.split('/')[-1][:-4]
        image = np.load(image_filepath)

        image = np.swapaxes(image, 0, 2)
        image = np.swapaxes(image, 1, 2)
        image = np.clip(image, -1024, 3071)
        image = ((image+1024)/(3071+1024))*255.0

        metaData = image_filename.split('_')
        lesionType = int(metaData[5])

        # bboxes = [
        #             [int(metaData[9]), int(metaData[7]), int(metaData[13]), int(metaData[11])],
        #         ]
        bboxes = [
                    [int(metaData[7]), int(metaData[9]), int(metaData[11]), int(metaData[13])],
                ]
        class_label = [int(metaData[5])]
        area = (int(metaData[11]) - int(metaData[7]))* (int(metaData[13]) - int(metaData[9]))

        # image2 = image[1,:,:]
        # boxes = bboxes.copy()
        # mask_file_value = 255
        # mask = np.zeros(shape = image2.shape[:2], dtype = image.dtype)
        # mask[boxes[0][0]-3:boxes[0][0], boxes[0][1]:boxes[0][3]] = mask_file_value
        # mask[boxes[0][2]:boxes[0][2]+3, boxes[0][1]:boxes[0][3]] = mask_file_value
        # mask[boxes[0][0]:boxes[0][2], boxes[0][1]-3:boxes[0][1]] = mask_file_value
        # mask[boxes[0][0]:boxes[0][2], boxes[0][3]:boxes[0][3]+3] = mask_file_value
        # plt.imshow(image2 + mask, cmap = 'gray', vmin=0, vmax = 255)
        # plt.show()


        # image2 = np.swapaxes(image, 1, 2)
        # image2 = np.swapaxes(image2, 0, 2)
        # boxes = bboxes.copy()
        # mask_file_value = 1400
        # mask = np.zeros(shape = image2.shape, dtype = image.dtype)
        # mask[boxes[0][0]-3:boxes[0][0], boxes[0][1]:boxes[0][3], 1] = mask_file_value
        # mask[boxes[0][2]:boxes[0][2]+3, boxes[0][1]:boxes[0][3], 1] = mask_file_value
        # mask[boxes[0][0]:boxes[0][2], boxes[0][1]-3:boxes[0][1], 1] = mask_file_value
        # mask[boxes[0][0]:boxes[0][2], boxes[0][3]:boxes[0][3]+3, 1] = mask_file_value
        # vim.visualize(image2, mask)

        
   

        if self.transform is not None:
            transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_label)
            image = transformed["image"]
            bboxes = transformed["bboxes"]
            class_label = transformed['class_labels']

        target = {"boxes": torch.tensor(bboxes),
                  "labels": torch.tensor([1]),
                  "area": torch.tensor([area])}
        
        # print(target["boxes"].numpy())
        # print(target["boxes"].numpy()[0][0], target["boxes"].numpy()[0][1])
        # idx1, idx2 = target["boxes"].numpy()[0][0], target["boxes"].numpy()[0][1]
        # print(idx1, idx2)
        # print(image[1,idx1,idx2])
        # input('......')
        
        return image, target, image_filename, idx

def custom_model():
    backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features
    # backbone = torchvision.models.mobilenet_v2().features

    # ``FasterRCNN`` needs to know the number of
    # output channels in a backbone. For mobilenet_v2, it's 1280
    # so we need to add it here
    backbone.out_channels = 1280

    # let's make the RPN generate 5 x 3 anchors per spatial
    # location, with 5 different sizes and 3 different aspect
    # ratios. We have a Tuple[Tuple[int]] because each feature
    # map could potentially have different sizes and
    # aspect ratios
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    # let's define what are the feature maps that we will
    # use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling.
    # if your backbone returns a Tensor, featmap_names is expected to
    # be [0]. More generally, the backbone should return an
    # ``OrderedDict[Tensor]``, and in ``featmap_names`` you can choose which
    # feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2,
    )

    # put the pieces together inside a Faster-RCNN model
    model = FasterRCNN(
        backbone,
        num_classes=2,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
    )

    return model

def set_hyperParam():
    params = {}
    params["batch_size"] = 4
    params["num_workers"] = 16
    params["device"] = "cuda:4"
    # params["device"] = "cpu"
    params["epochs"] = 10
    params["lr"] = 0.00000001
    return params

def validate(val_loader, model, params):
    model.train()
    val_losses = 0

    stream = tqdm(val_loader, total=len(val_loader))
    with torch.no_grad():
        for i, (images, targets, _, _) in enumerate(stream, start=1):
            images = images.to(params["device"], non_blocking=True).float()
            targets = {k:v.to(params["device"], non_blocking=True) for k, v in targets.items()}
            boxes_list = targets["boxes"]
            labels_list = targets["labels"]
            num_samples = len(boxes_list)
            list_of_dics = []
            for j in range(num_samples):
                sample_dict = {"boxes": boxes_list[j], "labels": labels_list[j]}
                list_of_dics.append(sample_dict)
            targets = list_of_dics

            loss_dict = model(images, targets)
            val_losses += sum(loss for loss in loss_dict.values()).item()
    val_losses = val_losses/len(val_loader)
    print('validation loss is: ' + str(val_losses))

def test(test_loader, model, log_dir, params):
    print('Testing ...')
    with torch.no_grad():
        model.eval()
        with open(os.path.join(log_dir,'detection.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['uid', 'x_min', 'y_min', 'x_max', 'y_max', 'scores'])
            for (images, targets, filename, idx) in test_loader:
                images = images.to(params["device"], non_blocking=True).float()
                targets = {k:v.to(params["device"], non_blocking=True) for k, v in targets.items()}
                output = model(images)
                
                for i in range(len(idx)):
                    image_names = []
                    x_min, y_min, x_max, y_max = [], [], [], []
                    scores = []
                    bbox = output[i]['boxes'].cpu().numpy()
                    score = output[i]['scores'].cpu().numpy()
                    if bbox.shape[0] == 0:
                        continue
                    uid = filename[i].split("_")
                    uid = uid[0] +"_"+ uid[1] +"_"+ uid[2] +"_"+ uid[3]
                    for b in range(bbox.shape[0]):
                        image_names.append(uid)
                        x_min.append(bbox[b][0])
                        y_min.append(bbox[b][1])
                        x_max.append(bbox[b][2])
                        y_max.append(bbox[b][3])
                        scores.append(score[b])

                    rows = zip(image_names, x_min, y_min, x_max, y_max, scores)
                    for row in rows:
                        writer.writerow(row)

    print('Testing Completed ...')


def train(train_loader, model, params, epoch):
    model.train()
    stream = tqdm(train_loader, total=len(train_loader))
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

    for i, (images, targets, _, _) in enumerate(stream, start=1):
        images = images.to(params["device"], non_blocking=True).float()
        targets = {k:v.to(params["device"], non_blocking=True) for k, v in targets.items()}

        boxes_list = targets["boxes"]
        labels_list = targets["labels"]
        num_samples = len(boxes_list)
        list_of_dics = []
        for j in range(num_samples):
            sample_dict = {"boxes": boxes_list[j], "labels": labels_list[j]}
            list_of_dics.append(sample_dict)
        targets = list_of_dics


        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses_comp = [loss.item() for loss in loss_dict.values()]
        
        optimizer.zero_grad()
        loss_value = losses.item()
        losses.backward()
        optimizer.step()
        
        description = f"Epoch: {epoch}/{params['epochs']}, loss:{loss_value:.4f}, class_loss:{losses_comp[0]:.4f}, Reg_loss:{losses_comp[1]:.4f}"

        stream.set_description(description)




if __name__ == '__main__':
    log_dir = os.path.join('/gpfs_projects/mohammadmeh.farhangi/DiskStation/DeepLesion/Results/Dummy',str(datetime.datetime.now()).replace(" ", "-"))
    # writer = SummaryWriter(log_dir)
    params = set_hyperParam()
    model = custom_model()
    model = model.to(params["device"])
    train_dir = '/gpfs_projects/mohammadmeh.farhangi/DiskStation/DeepLesion/NIH/numpy_dataset/3_consecutive_slice/Train'
    images_filepaths = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
    # images_filepaths = [images_filepaths[i] for i in range(len(images_filepaths)) if i%500 == 0]
    
    random.shuffle(images_filepaths)
    train_dataset  = DeepLesion(images_filepaths, transform=None)
    # vim.visualize_dataset(train_dataset)
    


    val_dir = '/gpfs_projects/mohammadmeh.farhangi/DiskStation/DeepLesion/NIH/numpy_dataset/3_consecutive_slice/Tune'
    images_filepaths = [os.path.join(val_dir, f) for f in os.listdir(val_dir)]
    # images_filepaths = [images_filepaths[i] for i in range(len(images_filepaths)) if i%50 == 0]
    val_dataset = DeepLesion(images_filepaths=images_filepaths, transform=None)

    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True, num_workers=params["num_workers"], pin_memory=True,)
    val_loader = DataLoader(val_dataset, batch_size=params["batch_size"], shuffle=False, num_workers=params["num_workers"], pin_memory=True,)

    w_images, w_labels, w_filename, w_idx = next(iter(train_loader))
    # writer.add_graph(model, w_images.to(params["device"], non_blocking=True).float())

    for epoch in range(1, params["epochs"] + 1):
        train(train_loader, model, params, epoch)
        validate(val_loader, model, params)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    torch.save(model.state_dict(), log_dir+'.pth')
    test(val_loader, model, log_dir, params)
