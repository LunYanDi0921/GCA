import torch
# x=[1,2,3,4,5,6]
# s = torch.tensor([8,9])
# x = torch.tensor([[1,2],[34,3],[5,34],[8,9]])
# # x=x.argsort(descending= True,dim=0)
# # x = x.sum(1)
# x = x[x.argsort(descending= True,dim=0)[:,0],:]
# print(x)
# for i in range(13000):
#     print(i)
dis = torch.tensor(
    [[1,2,3],
     [10,2,19],
     [67,4,10]]
)
dis = dis.argsort(descending= True,dim=1)
print(dis)