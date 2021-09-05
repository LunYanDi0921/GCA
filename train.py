import numpy
import argparse
import os.path as osp
import random
import nni
import time

import torch
from torch_geometric.utils import dropout_adj, degree, to_undirected

from simple_param.sp import SimpleParam
from pGRACE.model import Encoder, GRACE
from pGRACE.functional import drop_feature, drop_edge_weighted, \
    degree_drop_weights, \
    evc_drop_weights, pr_drop_weights, \
    feature_drop_weights, drop_feature_weighted_2, feature_drop_weights_dense
from pGRACE.eval import log_regression, MulticlassEvaluator
from pGRACE.utils import get_base_model, get_activation, \
    generate_split, compute_pr, eigenvector_centrality
from pGRACE.dataset import get_dataset

print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
def train(epoch):
    model.train()
    optimizer.zero_grad()

    def drop_edge(idx: int):
        global drop_weights

        if param['drop_scheme'] == 'uniform':
            return dropout_adj(data.edge_index, p=param[f'drop_edge_rate_{idx}'])[0]
        elif param['drop_scheme'] in ['degree', 'evc', 'pr']:
            return drop_edge_weighted(data.edge_index, drop_weights, p=param[f'drop_edge_rate_{idx}'], threshold=0.7)
        else:
            raise Exception(f'undefined drop scheme: {param["drop_scheme"]}')

    edge_index_1 = drop_edge(1)
    print(edge_index_1.size())
    edge_index_2 = drop_edge(2)
    print(edge_index_2.size())
    x_1 = drop_feature(data.x, param['drop_feature_rate_1'])
    # print(x_1.size())
    x_2 = drop_feature(data.x, param['drop_feature_rate_2'])
    # print(x_2.size())
    if param['drop_scheme'] in ['pr', 'degree', 'evc']:
        x_1 = drop_feature_weighted_2(data.x, feature_weights, param['drop_feature_rate_1'])
        x_2 = drop_feature_weighted_2(data.x, feature_weights, param['drop_feature_rate_2'])
    # print(x_1.size())
    # print(x_2.size())
    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)
    # print(z1.size(0)) #torch.Size([13752, 128])
    # print(z2.size()) #torch.Size([13752, 128])

    #loss = model.loss(z1, z2, batch_size=1024 if args.dataset == 'Coauthor-Phy' else None)
    loss = model.loss(z1, z2,edge_index_1,edge_index_2,20,epoch,1000)
    print(loss)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(final=False):
    model.eval()
    z = model(data.x, data.edge_index)

    evaluator = MulticlassEvaluator()
    if args.dataset == 'WikiCS':
        accs = []
        for i in range(20):
            acc = log_regression(z, dataset, evaluator, split=f'wikics:{i}', num_epochs=800)['acc']
            accs.append(acc)
        acc = sum(accs) / len(accs)
    else:
        acc = log_regression(z, dataset, evaluator, split='rand:0.1', num_epochs=3000, preload_split=split)['acc']

    if final and use_nni:
        nni.report_final_result(acc)
    elif use_nni:
        nni.report_intermediate_result(acc)

    return acc


if __name__ == '__main__':
    # argparse是一个Python模块：命令行选项、参数和子命令解析器
    # 主要有三个步骤：
    # 创建ArgumentParser()对象
    # 调用add_argument()方法添加参数
    # 使用parse_args() 解析添加的参数
    #使用 argparse 的第一步是创建一个 ArgumentParser 对象 创建解析器
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--dataset', type=str, default='Amazon-Computers')
    parser.add_argument('--param', type=str, default='local:amazon_computers.json')
    parser.add_argument('--seed', type=int, default=39788)
    parser.add_argument('--verbose', type=str, default='train,eval,final')
    parser.add_argument('--save_split', type=str, nargs='?')
    parser.add_argument('--load_split', type=str, nargs='?')
    default_param = {
        #学习率
        'learning_rate': 0.1,
        'num_hidden': 256,
        'num_proj_hidden': 32,
        'activation': 'prelu',
        'base_model': 'GCNConv',
        'num_layers': 2,
        'drop_edge_rate_1': 0.3,
        'drop_edge_rate_2': 0.4,
        'drop_feature_rate_1': 0.1,
        'drop_feature_rate_2': 0.0,
        'tau': 0.4,
        'num_epochs': 1,
        'weight_decay': 1e-5,
        'drop_scheme': 'degree',
    }

    # add hyper-parameters into parser
    param_keys = default_param.keys()
    for key in param_keys:
        parser.add_argument(f'--{key}', type=type(default_param[key]), nargs='?')
    args = parser.parse_args()
    print(args)
    #print(args['dataset'])
    # parse param
    # print(default_param)
    sp = SimpleParam(default=default_param)
    # print(args.param)
    param = sp(source=args.param, preprocess='nni')

    print(param)
    # print(args)
    # merge cli arguments and parsed param
    for key in param_keys:
        if getattr(args, key) is not None:
            param[key] = getattr(args, key)
    print(param)
    use_nni = args.param == 'nni'
    if use_nni and args.device != 'cpu':
        args.device = 'cuda'

    torch_seed = args.seed
    torch.manual_seed(torch_seed)
    random.seed(12345)
    # print(args.device)
    device = torch.device(args.device)

    path = osp.expanduser('~/datasets')
    path = osp.join(path, args.dataset)
    # print(path)
    dataset = get_dataset(path, args.dataset)
    print(dataset)
    data = dataset[0]
    data = data.to(device)  #Data(edge_index=[2, 491722], x=[13752, 767], y=[13752])
    # print(data.num_nodes)
    # print(data.num_edges)

    # generate split
    #将节点划分为训练集 验证集 测试集 返回一个长为num_nodes的tensor 以训练集为例 tensor中训练样本所对应的下标位置的值是ture 否则为false 
    split = generate_split(data.num_nodes, train_ratio=0.1, val_ratio=0.1)
    # print(split)
    # print(args.save_split)
    # print(args.load_split)
    if args.save_split:
        torch.save(split, args.save_split)
    elif args.load_split:
        split = torch.load(args.load_split)
    #print(data.num_features)
    encoder = Encoder(dataset.num_features, param['num_hidden'], get_activation(param['activation']),
                      base_model=get_base_model(param['base_model']), k=param['num_layers']).to(device)
    model = GRACE(encoder, param['num_hidden'], param['num_proj_hidden'], param['tau']).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=param['learning_rate'],
        weight_decay=param['weight_decay'] #权重衰减(如L2惩罚)(默认: 0)
    )

    if param['drop_scheme'] == 'degree':
        drop_weights = degree_drop_weights(data.edge_index).to(device)
        print("drop_weights.size",drop_weights.size())
    elif param['drop_scheme'] == 'pr':
        drop_weights = pr_drop_weights(data.edge_index, aggr='sink', k=200).to(device)
    elif param['drop_scheme'] == 'evc':
        drop_weights = evc_drop_weights(data).to(device)
    else:
        drop_weights = None

    if param['drop_scheme'] == 'degree':
        edge_index_ = to_undirected(data.edge_index)
        node_deg = degree(edge_index_[1])
        if args.dataset == 'WikiCS':
            feature_weights = feature_drop_weights_dense(data.x, node_c=node_deg).to(device)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_deg).to(device)
    elif param['drop_scheme'] == 'pr':
        node_pr = compute_pr(data.edge_index)
        if args.dataset == 'WikiCS':
            feature_weights = feature_drop_weights_dense(data.x, node_c=node_pr).to(device)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_pr).to(device)
    elif param['drop_scheme'] == 'evc':
        node_evc = eigenvector_centrality(data)
        if args.dataset == 'WikiCS':
            feature_weights = feature_drop_weights_dense(data.x, node_c=node_evc).to(device)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_evc).to(device)
    else:
        feature_weights = torch.ones((data.x.size(1),)).to(device)
    print("feature_weights.size",feature_weights.size())

    log = args.verbose.split(',')
    param['num_epochs'] =3000
    for epoch in range(1, param['num_epochs'] + 1):
        time_start = time.time()
        loss = train(epoch)
        time_end = time.time()    #结束计时
        time_c= time_end - time_start   #运行所花时间
        print('time cost', time_c, 's')
        if 'train' in log:
            print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}')
        acc = test()

        if 'eval' in log:
            print(f'(E) | Epoch={epoch:04d}, avg_acc = {acc}')
        # if epoch % 100 == 0:
        #     acc = test()

        #     if 'eval' in log:
        #         print(f'(E) | Epoch={epoch:04d}, avg_acc = {acc}')

    acc = test(final=True)

    if 'final' in log:
        print(f'{acc}')
