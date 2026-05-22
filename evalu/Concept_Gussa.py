import matplotlib.pyplot as plt

from model.VAE import *
from args.args import args
from torchcam.methods import GradCAM
from model.VAE import simpleconvnet
from Datamanage.load_data import get_vgg16_conv_kernels
from Datamanage.load_data import LeNet_load_data as load_data


kernel_h = 3
kernel_w = 3
n_concept = 21
model_path = '/mnt/data/code/sava_model/Concept_model_20250611_164104.pth'

# 定义数据集
ker_dataloader = get_vgg16_conv_kernels(args.model_path,batch_size=1)
image_trainloader, image_val_loader, class_names = load_data(args.data_dir, 8, args.val_split)

# 定义模型
model = Concept_VAE(kernel_h=kernel_h, kernel_w=kernel_w, n_concept=n_concept)
model.load_state_dict(torch.load(model_path))
model.cuda()
sim_model = simpleconvnet().cuda()

for ker in ker_dataloader:
    for image_data, labels in image_val_loader:
        ker = ker.cuda()
        out_data = model(ker)
        for batch_id in range(1):
            concept_kernel = out_data[batch_id][:21, :, :]  # shape: [Concept_n, H, W]

            for ker_id in range(concept_kernel.size(0)):  # 循环每个卷积核
                selected_kernel = concept_kernel[ker_id]
                selected_kernel = selected_kernel.unsqueeze(0).unsqueeze(1).repeat(3, 3, 1, 1)

                sim_model.conv1.weight.data = selected_kernel
                sim_model.conv2.weight.data = selected_kernel
                sim_model.conv2.weight.data = selected_kernel

                cam = GradCAM(sim_model, target_layer=sim_model.conv3)
                activation_map = cam(image_data)

                plt.imshow(activation_map[0].cpu().detach().numpy(), cmap='jet')
                plt.colorbar()
                plt.show()








kernel = model()

model.eval()




