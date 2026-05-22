import torch
import os
import datetime
import glob
from natsort import natsorted
from Datamanage.load_data import LeNet_load_data as load_data
import numpy as np
from tqdm import tqdm
from model.VAE import *
from args.args import args
import torch.nn.functional as F
from Datamanage.load_data import get_vgg16_conv_kernels
from Datamanage.load_data import get_conv_weight_dataloader


# 加载概念向量
concept_path = '/mnt/data/code/concept_vector'
concepts = []
for file_name in os.listdir(concept_path):
    concept = np.load(os.path.join(concept_path, file_name))
    concepts.append(concept)
concept_tensor = torch.cat([torch.tensor(concept) for concept in concepts], dim=0) # [n_concept, 328]
# 定义概念向量的数量
n_concept = concept_tensor.shape[0]



kernel_h = 3
kernel_w = 3

# 定义训练参数
num_epochs = 10
batch_size = 64
learning_rate = 0.001
# 检查 CUDA 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = 'sava_model'



# 定义损失函数
def my_loss_function(input_data, out_data, image_data, labels, concept_model, concept=concept_tensor):
    # input_shape:[B, H, W]
    # out_shape: [B, n_concept+1, H, W]

    # 损失函数1：重构损失
    total_kernel = out_data[:, len(concept), :, :]
    total_kernel = total_kernel.view(total_kernel.size(0), -1)  # 展平
    input_kernel = input_data.view(input_data.size(0), -1)  # 展平
    manhatten_distance = torch.sum(torch.abs(total_kernel - input_kernel), dim=1) # 计算每个样本的曼哈顿距离
    loss_1 = 1 / (1 + manhatten_distance)  # 曼哈顿距离越小，损失越大


    # 重构损失
    output_selected = out_data[:, len(concept), :, :]
    input_data_flat = input_data.view(input_data.size(0), -1)
    output_selected_flat = output_selected.view(output_selected.size(0), -1)
    cosine_sim = F.cosine_similarity(input_data_flat, output_selected_flat, dim=1)
    loss_1 = 1 - cosine_sim.mean()

    # 独立性损失
    ker_batch = input_data.size(0)
    loss_2 = 0
    for i in range(ker_batch): # 循环的是卷积核
        ker = out_data[i][:len(concept), :, :]  # shape: [Concept_n, H, W]
        kernels = ker.unsqueeze(1).repeat(1, 3, 1, 1) # shape: [Concept_n, 3, H, W]
        conv_layer = nn.Conv2d(in_channels=3, out_channels=30, kernel_size=(kernel_h, kernel_w), padding=1, bias=False).cuda()
        with torch.no_grad():
            conv_layer.weight.copy_(kernels)
        feature = conv_layer(image_data)
        feature = F.adaptive_avg_pool2d(feature, output_size=(4, 4))  # shape: [8, 30, 4, 4]
        feature = feature.view(feature.size(0), feature.size(1), -1)  # shape: [8, 30, 16]
        concepts = torch.tensor(np.stack(concept), dtype=torch.float32).cuda()
        feature_map = F.normalize(feature, dim=-1)
        concepts = F.normalize(concepts, dim=-1)
        tep_loss = 0
        for j in range(feature_map.size(0)): # 循环的是图
            feature_image = feature_map[j]  # shape: [30, 16]
            cosine_sim_image = F.cosine_similarity(feature_image, concepts, dim=1)
            label = labels[j].item()

            # 正类概念
            pos_ids = torch.tensor([label * 3, label * 3 + 1, label * 3 + 2])
            pos_sim = cosine_sim_image[pos_ids].mean()

            # 负类概念
            all_idx = torch.arange(len(concepts))
            neg_idx =  all_idx[~((all_idx // 3) == label)]

            neg_sim = cosine_sim_image[neg_idx].view(-1)
            # hinge-based loss: 负类不能超过正类 - margin
            neg_loss = torch.logsumexp(neg_sim - pos_sim + 0.1, dim=0)

            pos_loss = -pos_sim
            tep_loss += pos_loss + neg_loss
        loss_2 += tep_loss / feature_map.size(0)

    loss_2 = loss_2 / ker_batch
    loss = loss_1 + loss_2
    return  loss, loss_1, loss_2


def train_model(model, ker_dataloader, image_dataloader):
    # 加载模型
    print("Loading model...")
    concept_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)
    concept_model.eval()

    epochs = args.epochs
    device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = args.save_dir

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        print(f"\nEpoch [{epoch+1}/{epochs}]")

        ker_loop = tqdm(ker_dataloader, desc="Kernels", leave=False)
        for x in ker_loop:
            x = x.to(device)

            image_loop = tqdm(image_dataloader, desc="Images", leave=False)
            for image_data, labels in image_loop:
                image_data = image_data.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                out = model(x)

                loss, loss1, loss2 = my_loss_function(x, out, image_data, labels, concept_model)
                loss.backward()
                optimizer.step()

                batch_loss = loss.item()
                batch_loss1 = loss1.item()
                batch_loss2 = loss2.item()
                total_loss += batch_loss
                image_loop.set_postfix(
                    loss=f"{batch_loss:.4f}",
                    loss1=f"{batch_loss1:.4f}",
                    loss2=f"{batch_loss2:.4f}"
                )

        print(f"Total Loss: {total_loss:.4f}")

    # 保存模型
    os.makedirs(model_path, exist_ok=True)
    model_init_filename = f"Concept_model_{timestamp}.pth"
    final_model_path = os.path.join(model_path, model_init_filename)
    torch.save(model.state_dict(), final_model_path)
    print(f"\n✅ Saved final model to: {final_model_path}")


def train_main(args):
    device = args.device
    print(f"Using device: {device}")

    # LeNet定义数据集
    # ker_dataloader = get_conv_weight_dataloader(
    #     pth_path='/mnt/data/code/sava_model/lenet_20250513_153941.pth',
    #     conv_key='conv2.weight',
    #     batch_size=args.batch_size
    # )

    # 定义数据集
    ker_dataloader = get_vgg16_conv_kernels(args.model_path)
    image_trainloader, image_val_loader, class_names = load_data(args.data_dir, args.batch_size, args.val_split)

    model = Concept_VAE(kernel_h=kernel_h, kernel_w=kernel_w, n_concept=n_concept)

    train_model(model, ker_dataloader, image_trainloader)




# 定义训练代码
# def train_model(model, ker_dataloader,  image_dataloader, epochs=10, device='cuda'):
#     model = model.cuda()
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#
#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0
#         for x, _ in ker_dataloader:
#             x = x.cuda()
#             for image_data, labels in image_dataloader:
#                 optimizer.zero_grad()
#                 # 前向传播
#                 out = model(x)
#                 image_data = image_data.cuda()
#                 labels = labels.cuda()
#                 loss = my_loss_function(x, out, image_data, labels)
#                 # 反向传播
#                 loss.backward()
#                 optimizer.step()
#                 total_loss += loss.item()
#         print("total_loss:{}", total_loss)
#
#     # 保存最终模型
#     final_model_path = os.path.join(model_path, 'final_model.pth')
#     torch.save(model.state_dict(), final_model_path)
#     print(f"\nSaved final model to {final_model_path}")





# 定义模型
# model = Concept_VAE(kernel_h, kernel_w, n_concept).cuda()
# # 定义优化器
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# 定义训练函数

# def train(model, ker_dataloader, image_dataloader, epochs=10, device='cuda'):
#     model.train()
#     for epoch in range(epochs):
#         total_loss = 0
#
#         # 使用tqdm包装dataloader，显示进度条
#         progress_bar = tqdm(enumerate(ker_dataloader), total=len(ker_dataloader),
#                             desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
#         for batch_idx, (x, _) in progress_bar:
#             x = x.cuda() # 输入形状应为[batch, 3, 3]
#             optimizer.zero_grad()
#             # 前向传播
#             out = model(x)
#             # 计算损失
#             pro_bar = tqdm(enumerate(image_dataloader), total=len(image_dataloader),
#                            desc=f"Batch {batch_idx + 1}/{len(ker_dataloader)}", leave=False)
#             for img_idx, (image_data, labels) in pro_bar:
#                 image_data = image_data.cuda()
#                 loss = my_loss_function(x, out, image_data, labels)
#                 # 反向传播
#                 loss.backward()
#                 optimizer.step()
#
#                 # 更新进度条
#                 pro_bar.set_postfix({
#                     'loss': f'{loss.item() / image_data.size(0):.4f}'
#                 })
#             # 更新进度条
#             progress_bar.set_postfix({
#                 'loss': f'{loss.item() / x.size(0):.4f}'
#             })
#             total_loss += loss.item()
#
#         print(f'Epoch {epoch + 1} | Avg Loss: {total_loss / len(ker_dataloader.dataset):.4f}')

# conv_layer = nn.Conv2d(in_channels=3, out_channels=len(concept), kernel_size=(kernel_h, kernel_w), bias=False)

# conv_layer.weight = nn.Parameter(conv_kernels)
# conv_kernels = out_data[:, :len(concept), :, :]  # shape: [B, n_concept, H, W]
# feature_map = conv_layer(image_data)  # shape: [B, n_concept, H', W']

# conv_kernels = out_data[:, :len(concept), :, :]  # shape: [B, n_concept, H, W]
#     class ConvConceptExtractor(nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.conv = nn.Conv2d(in_channels=3, out_channels=len(concept), kernel_size=(kernel_h, kernel_w),stride=1, padding=1, bias=False)
#             self.conv.weight = nn.Parameter(conv_kernels)
#             self.pool = nn.Sequential(
#                 nn.MaxPool2d(kernel_size=2, stride=2),  # [B, 30, 112, 112]
#                 nn.MaxPool2d(kernel_size=2, stride=2),  # [B, 30, 56, 56]
#                 nn.MaxPool2d(kernel_size=2, stride=2),  # [B, 30, 28, 28]
#                 nn.MaxPool2d(kernel_size=2, stride=2),  # [B, 30, 14, 14]
#                 nn.MaxPool2d(kernel_size=2, stride=2),  # [B, 30, 7, 7]
#                 nn.AdaptiveMaxPool2d((4, 4))  # [B, 30, 4, 4]
#             )
#
#         def forward(self, x):
#             x = self.conv(x)  # 卷积
#             x = self.pool(x)  # 多次 MaxPool 下采样
#             return x
#     conv_extractor = ConvConceptExtractor()
#     feature_map = conv_extractor(image_data)  # shape: [B, n_concept, 4, 4]
#     feature_map = feature_map.view(feature_map.size(0), feature_map.size(1), -1)
#     feature_map = F.normalize(feature_map, dim=-1)  # 归一化特征图
#     concepts = torch.tensor(np.stack(concept), dtype=torch.float32)
#     concepts = F.normalize(concepts, dim=-1)  # 归一化概念向量
#     concepts_all = concepts.view(-1, 16)  # [n_concept*3, 16]
#     sim = torch.matmul(feature_map, concepts_all.T)
#
#     loss_2 = 0.0
#     for i in range(feature_map.size(0)):
#         label = labels[i].item()
#         pos_ids = list(range(label * 3, (label + 1) * 3))
#
#         # 取出这 B 个样本中正类的通道
#         for pid in pos_ids:
#             # 该通道提取的特征与其正类概念 pid 相似度越高越好（越接近1）
#             pos_sim = sim[i, pid, pid]
#             loss_2 += (1 - pos_sim)
#
#             # 该通道提取的特征与其他类的概念越不相似越好（越接近0）
#             neg_ids = list(set(range(concepts_all.size(0))) - set(pos_ids))
#             neg_sim = sim[i, pid, neg_ids]
#             loss_2 += (neg_sim ** 2).mean()
# # 加载概念向量
# # 加载 .pth 文件
# files = glob.glob('/mnt/data/code/concept_vector/*.pth')
# files = natsorted(files)
# path = os.path.basename(files[-1])
# concept_path = os.path.join(args.save_concept_vector, path)
# concept_vector = torch.load(concept_path)
# key = concept_vector.keys()
# concept_list = []
# for i in key:
#     concept_list.append(concept_vector[i])
# concept = np.concatenate(concept_list, axis=0)
# n_concept = len(concept)