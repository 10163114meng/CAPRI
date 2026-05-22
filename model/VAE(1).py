import torch
import torch.nn as nn
import torch.nn.functional as F


class KernelEncoder(nn.Module):
    def __init__(self, kernel_h, kernel_w, n_concept=4):
        super().__init__()
        input_dim = kernel_h * kernel_w
        self.n_concept = n_concept

        self.encoder = nn.Sequential(
            nn.Flatten(start_dim=1),  #
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_concept * 4)
        )

    def forward(self, x):  # x: [B, h, w]
        out = self.encoder(x)  # [B, n_concept * 4]
        out = out.view(-1, self.n_concept, 4)  # [B, n_concept, 4]


        weight  = torch.tanh(out[:, :, 0])            # [-1, 1]
        mu      = torch.tanh(out[:, :, 1]) * 1.0       # [-5, 5]
        sigma   = F.softplus(out[:, :, 2]) + 1e-5        # > 0
        amp     = F.softplus(out[:, :, 3])             # > 0

        return torch.stack([weight, mu, sigma, amp], dim=-1)  # [B, n_concept, 4]


def sampling(params, kernel_h, kernel_w):

    B, C, _ = params.shape
    device =  params.device


    weight = torch.tanh(params[:, :, 0])             # [-1, 1]
    mu     = torch.tanh(params[:, :, 1])            # [-1, 1]
    sigma  = F.softplus(params[:, :, 2])               # >0
    amp    = F.softplus(params[:, :, 3])              # >0


    y, x_grid = torch.meshgrid(
        torch.linspace(-1, 1, kernel_h, device=device),
        torch.linspace(-1, 1, kernel_w, device=device),
        indexing='ij'
    )
    coords = x_grid.reshape(1, 1, -1)  # [1, 1, H*W]


    mu = mu.unsqueeze(-1)        # [B, C, 1]
    sigma = sigma.unsqueeze(-1)
    amp = amp.unsqueeze(-1)


    responses = amp * torch.exp(-0.5 * ((coords - mu) / sigma) ** 2)  # [B, C, H*W]

    weight = weight.unsqueeze(-1)   # [B, C, 1]
    combined = (weight * responses).sum(dim=1, keepdim=True)  # [B, 1, H*W]

    return torch.cat([responses, combined], dim=1)  # [B, C+1, H*W]



class ConceptDecoder(nn.Module):
    def __init__(self, kernel_h, kernel_w, hidden_dim=128):
        super().__init__()
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w
        self.kernel_size = kernel_h * kernel_w

        self.decoder = nn.Sequential(
            nn.Linear(self.kernel_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.kernel_size)
        )

    def forward(self, x):  # x: [B, C+1, H*W]
        B, C, HW = x.shape

        x = x.view(B * C, HW)
        x = self.decoder(x)  # [B*C, H*W]
        x = x.view(B, C, self.kernel_h, self.kernel_w)  # [B, C, H, W]
        return x


class Concept_VAE(nn.Module):
    def __init__(self, kernel_h, kernel_w, n_concept):
        super().__init__()
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w
        self.n_concept = n_concept

        self.encoder = KernelEncoder(kernel_h, kernel_w, n_concept)
        self.decoder = ConceptDecoder(kernel_h, kernel_w)

    def forward(self, x):
        out = self.encoder(x)  # shape: [B, n_concept, 2], [B, n_concept, 2], [B, n_concept, 1]
        z = sampling(out, self.kernel_h, self.kernel_w)  # shape: [B, n_concept+1, n_concept]
        out = self.decoder(z)  # shape: [B, n_concept+1, H, W]
        return out


class simpleconvnet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=3):
        super(simpleconvnet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride= 1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride= 1, padding=1,  bias=False)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride= 1, padding=1,  bias=False)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 第一层卷积
        x = F.relu(self.conv2(x))
        x = self.conv3(x)  # 第二层卷积
        return x


# class KernelEncoder(nn.Module):
#     def __init__(self, kernel_h, kernel_w, n_concept):
#         super().__init__()
#         self.n_concept = n_concept
#         input_dim = kernel_h * kernel_w
#
#         self.flatten = nn.Flatten(start_dim=1)
#
#         self.mlp = nn.Sequential(
#             nn.Linear(input_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#         )
#         # 输出 μ1, μ2, σ1, σ2, ρ（共 5 个参数 × 每个概念）
#         self.param_head = nn.Linear(64, self.n_concept * 5)
#
#     def forward(self, x):
#         """
#         输入: x.shape = [B, kernel_h, kernel_w]
#         输出: mu, var, rho with shape:
#             mu:  [B, n_concept, 2]
#             var: [B, n_concept, 2]  （正值）
#             rho: [B, n_concept, 1]  （[-1, 1]）
#         """
#         x = self.flatten(x)              # [B, kernel_h * kernel_w]
#         x = self.mlp(x)                  # [B, 64]
#         raw_params = self.param_head(x) # [B, n_concept * 5]
#         raw_params = raw_params.view(-1, self.n_concept, 5)  # [B, n_concept, 5]
#
#         mu  = torch.tanh(raw_params[..., 0:2]) * 5.0                   # [B, n, 2]
#         sigma = F.softplus(raw_params[..., 2:4])       # [B, n, 2], 正值
#         rho = torch.tanh(raw_params[..., 4:5])         # [B, n, 1], 限定在 [-1, 1]
#
#         return mu, sigma, rho

# def sampling(mu, var, rho):
#     # 循环batch
#     total_list  = []
#     for i in range(mu.shape[0]):
#         mu_i = mu[i]
#         var_i = torch.clamp(var[i], min=1e-3)  # 避免负数和 0 方差
#         rho_i = torch.clamp(rho[i], min=-0.99, max=0.99)
#         # 循环每个概念
#         n_Ga = []
#         for j in range(mu_i.shape[0]):
#             mu_ij = mu_i[j]
#             var_ij = var_i[j]
#             rho_ij = rho_i[j]
#             v1, v2 = var_ij[0], var_ij[1]
#             rho_val = rho_ij[0] if rho_ij.ndim > 0 else rho_ij
#             cov_matrix = torch.stack([
#                 torch.stack([v1 ** 2, rho_val * v1 * v2]),
#                 torch.stack([rho_val * v1 * v2, v2 ** 2])
#             ])
#             cov_matrix = cov_matrix.to(mu.device)
#             # 强制对称 & 稍微增加对角线，确保正定
#             epsilon = 1e-4
#             cov_matrix = (cov_matrix + cov_matrix.T) / 2
#             cov_matrix += epsilon * torch.eye(cov_matrix.size(0), device=cov_matrix.device)
#
#             mu_cpu = mu_ij.cpu() if mu_ij.is_cuda else mu_ij
#             cov_cpu = cov_matrix.cpu() if cov_matrix.is_cuda else cov_matrix
#             # 生成多元正态分布样本
#             dist = torch.distributions.MultivariateNormal(loc=mu_cpu, covariance_matrix=cov_cpu)
#             samples = dist.rsample((mu_i.shape[0],)).cpu()
#             probs = torch.exp(dist.log_prob(samples))
#             n_Ga.append(probs)
#         n_Ga.append(sum(n_Ga))
#         total_list.append(torch.stack(n_Ga, dim=0))
#     return torch.stack(total_list, dim=0).cuda()


# class ConceptDecoder(nn.Module):
#     def __init__(self, weight_h, weight_w, n_concept):
#         super().__init__()
#         self.weight_h = weight_h
#         self.weight_w = weight_w
#         output_dim = weight_h * weight_w
#
#         # 映射每个 [n_concept] 向量 → [weight_h * weight_w]
#         self.decoder = nn.Sequential(
#             nn.Linear(n_concept, 128),
#             nn.ReLU(),
#             nn.Linear(128, output_dim),
#         )
#
#     def forward(self, z):  # z: [B, n_concept+1, n_concept]
#         B, N, C = z.shape
#         z = z.view(-1, C)                       # [B * (n_concept+1), n_concept]
#         out = self.decoder(z)                   # → [B * (n_concept+1), H * W]
#         out = out.view(B, N, self.weight_h, self.weight_w)
#         return out  # shape: [B, n_concept+1, H, W]

# from args.args import args
# import torch.nn.functional as F
# import glob
# from natsort import natsorted
# import numpy as np
# import os

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
# concept_num = len(concept)
# filter_size = [3, 3]
# input_decoder = concept_num*(len(concept[0]) + 4)


# class Encoder(nn.Module):
#     def __init__(self, input_dim=filter_size, latent_dim=concept_num):
#         super(Encoder, self).__init__()
#         self.input_dim = input_dim
#         self.flatten_dim = input_dim[0] * input_dim[1]  # 3×3 → 9
#
#         # 全连接层
#         self.fc = nn.Sequential(
#             nn.Linear(self.flatten_dim, 16),
#             nn.ReLU(),
#             nn.Linear(16, 8),
#             nn.ReLU(),
#         )
#
#         # 输出x的分布参数
#         self.fc_mu_x = nn.Linear(8, latent_dim)
#         self.fc_logvar_x = nn.Linear(8, latent_dim)
#         # 输出y的分布参数
#         self.fc_mu_y = nn.Linear(8, latent_dim)
#         self.fc_logvar_y = nn.Linear(8, latent_dim)
#
#
#     def forward(self, x):
#         x = x.view(-1, self.flatten_dim)  # Flatten: (batch, 3, 3) → (batch, 9)
#         x = self.fc(x)
#         mu_x = self.fc_mu_x(x)  # x的均值μₓ
#         logvar_x = self.fc_logvar_x(x)  # x的对数方差logσₓ²
#         mu_y = self.fc_mu_y(x)  # y的均值μᵧ
#         logvar_y = self.fc_logvar_y(x)
#         combined_stack = torch.stack([mu_x, logvar_x, mu_y, logvar_y], dim=2)
#         return combined_stack
#
#
# # 定义解码器
# class Decoder(nn.Module):
#     def __init__(self, latent_dim=input_decoder, output_dim = filter_size):
#         super(Decoder, self).__init__()
#         self.input_dim = latent_dim
#         self.output_size = output_dim[0] * output_dim[1]
#         self.fc1 = nn.Linear(latent_dim, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, self.output_size)
#
#     def forward(self, z):
#         z = z.view(-1, self.input_dim)
#         z = F.relu(self.fc1(z))
#         z = F.relu(self.fc2(z))
#         x_hat = torch.sigmoid(self.fc3(z))  # 使用sigmoid以保证输出在[0,1]之间
#         out = x_hat.reshape(-1, *filter_size)# 还原为原始形状
#         return out
#
#
# ## 修改
# class VAE(nn.Module):
#     def __init__(self):
#         super(VAE, self).__init__()
#         self.encoder = Encoder()
#         self.decoder = Decoder()
#
#
#     def forward(self, x):
#         combined_stack = self.encoder(x)
#         concept_tensor = torch.from_numpy(concept).float().cuda()  # shape: [concept_num, 16]
#         batch_size = combined_stack.size(0)
#         concept_expanded = concept_tensor.expand(batch_size, -1, -1)
#         z = torch.cat([combined_stack, concept_expanded], dim=2)
#         x_hat = self.decoder(z)
#         return x_hat
