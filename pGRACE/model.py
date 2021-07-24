from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv


class Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation, base_model=GCNConv, k: int = 2, skip=False):
        super(Encoder, self).__init__()
        self.base_model = base_model
        print("jjjjj")
        assert k >= 2
        self.k = k
        self.skip = skip
        if not self.skip:
            self.conv = [base_model(in_channels, 2 * out_channels).jittable()]
            for _ in range(1, k - 1):
                self.conv.append(base_model(2 * out_channels, 2 * out_channels))
            self.conv.append(base_model(2 * out_channels, out_channels))
            self.conv = nn.ModuleList(self.conv)
            print(self.conv)
            self.activation = activation
        else:
            self.fc_skip = nn.Linear(in_channels, out_channels)
            self.conv = [base_model(in_channels, out_channels)]
            for _ in range(1, k):
                self.conv.append(base_model(out_channels, out_channels))
            self.conv = nn.ModuleList(self.conv)
            self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        if not self.skip:
            for i in range(self.k):
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                x = self.activation(self.conv[i](x, edge_index))
            return x
        else:
            h = self.activation(self.conv[0](x, edge_index))
            hs = [self.fc_skip(x), h]
            for i in range(1, self.k):
                u = sum(hs)
                hs.append(self.activation(self.conv[i](u, edge_index)))
            return hs[-1]


class GRACE(torch.nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int, tau: float = 0.5):
        super(GRACE, self).__init__()
        self.encoder: Encoder = encoder
        self.tau: float = tau
        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)
        self.num_hidden = num_hidden

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                                     / (refl_sim.sum(1) + between_sim.sum(1)
                                        - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    # def loss(self, z1: torch.Tensor, z2: torch.Tensor, mean: bool = True, batch_size: Optional[int] = None):
    #     h1 = self.projection(z1)
    #     h2 = self.projection(z2)
    #
    #     if batch_size is None:
    #         l1 = self.semi_loss(h1, h2)
    #         l2 = self.semi_loss(h2, h1)
    #     else:
    #         l1 = self.batched_semi_loss(h1, h2, batch_size)
    #         l2 = self.batched_semi_loss(h2, h1, batch_size)
    #
    #     ret = (l1 + l2) * 0.5
    #     ret = ret.mean() if mean else ret.sum()
    #
    #     return ret
    def cal_hard_negative_sample_part_1(self,edge_index,index):
        #计算困难负样本第一部分 某个样本的一阶邻居
        print(edge_index)
        hnsp1 = edge_index[:,edge_index[0, :] == index]
        hnsp1 = hnsp1[1,:]
        return hnsp1 #返回节点i的邻居节点下标
    def cal_hard_negative_sample_part_2(self,z,z_index,index,k,hnsp1):
         #计算困难负样本第二部分 根据特征表示的相似度 返回相似度高的前k个节点的特征表示
         device = z.device
         #z_index = z[index,:] #节点index的特征表示
         # print(z_index.size())
         #left是除了节点index本身和其一阶邻居之外的其他节点下标
         left = torch.tensor([i for i in range(z.size(0)) if i not in hnsp1 and i != index],dtype=int)
         # print(left.size())
         z_left = z[left,:]
         # print(z_left.size())
         # print(z)
         # print(z_left)
         # print(z_index - z_left)
         # print((z_index - z_left).size())
         dis = (z_index - z_left).pow(2).sum(1).sqrt()
         # print("z_left.size()",z_left.size()) #[13749,128]
         # print("dis.size()", dis.size()) #[13749]
         dis = dis.view(-1,1).to(device)
         left = left.view(-1,1).to(device)
         dis = torch.cat([dis, left], dim=1).to(device)
         dis = dis[dis.argsort(descending= True,dim=0)[:,0],:]
         dis_k_index = dis[0:k,1].int()
         # print("*",dis)
         # print("*", dis_k_index)
         hnsp2 = dis_k_index
         return hnsp2
    def generate_hard_negative_sample_1(self,hnsp1,hnsp2,z,z_index,alpha):
        # hnsp1 = hnsp1.view(-1,1)
        # hnsp2 = hnsp2.view(-1, 1)
        hnsp = torch.cat([hnsp1,hnsp2])
        z_hnsp = z[hnsp,:] #本身的困难负样本
        gene_hns = alpha * z_index + (1 - alpha) * z_hnsp #生成的困难负样本
        hns = torch.cat([z_hnsp, gene_hns], dim=0)
        return hns

    # def  generate_hard_sample_2(self,hnsp1,hnsp2,z,q,beta):
    #     hnsp = torch.cat([hnsp1, hnsp2])
    #     z_hnsp = z[hnsp, :]
    #     return beta * q + (1 - beta) * z_hnsp

    def loss(self,z1:torch.Tensor, z2: torch.Tensor, edge_index_1:torch.Tensor,  edge_index_2: torch.Tensor,k):
           l = 0
           device = z1.device
           for i in range(z1.size(0)):
               print(i)
               #遍历每一个节点
               z1_index = z1[i, :]
               intra_hnsp1 = self.cal_hard_negative_sample_part_1(edge_index_1,i)
               print(intra_hnsp1)
               # (self, z, z_index, index, k, hnsp1)
               intra_hnsp2 = self.cal_hard_negative_sample_part_2(z1, z1_index,i,k, intra_hnsp1 )
               print(intra_hnsp2)
               alpha = torch.rand(1).to(device)
               intra_negative = self.generate_hard_negative_sample_1(intra_hnsp1,intra_hnsp2,z1,z1_index,alpha)
               print(intra_negative.size())
               #intra_dis = torch.exp(torch.sqrt(torch.mm(z1_index.view(1,-1),intra_negative.t()))/self.tau).sum()
               intra_dis = torch.exp((z1_index - intra_negative).pow(2).sum(1).sqrt() / self.tau).sum()
               inter_hnsp1 = self.cal_hard_negative_sample_part_1(edge_index_2, i)
               print(inter_hnsp1)
               inter_hnsp2 = self.cal_hard_negative_sample_part_2(z2, z1_index,i, k, inter_hnsp1)
               print(inter_hnsp2)
               # beta = torch.rand()/2
               z2_index = z2[i, :]
               inter_negative = self.generate_hard_negative_sample_1(inter_hnsp1, inter_hnsp2, z2, z1_index, alpha)
               inter_dis = torch.exp((z1_index - inter_negative).pow(2).sum(1).sqrt() / self.tau).sum()
               # inter_dis = torch.exp(torch.sqrt(torch.mm(z2_index.view(1, -1), inter_negative.t())) / self.tau).sum()
               d = torch.exp((z1_index - z2_index).pow(2).sum().sqrt() / self.tau)
               l += torch.log(d / ( d + intra_dis + inter_dis))
           return l

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret
