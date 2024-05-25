import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import math
import argparse
import h5py
import time

#torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = True


parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-w","--n_way",type = int, default = 16)                      # way
parser.add_argument("-s","--n_shot",type = int, default = 1)       # support set per class
parser.add_argument("-b","--n_query",type = int, default = 4)       # query set per class
parser.add_argument("-e","--episode",type = int, default= 1000)
#-----------------------------------------------------------------------------------#
parser.add_argument("-l1","--learning_rate1", type = float, default = 0.001)
parser.add_argument("-l2","--learning_rate2", type = float, default = 0.009)

parser.add_argument("-g","--gpu",type=int, default=0)
args = parser.parse_args()



n_way = args.n_way
n_shot = args.n_shot
n_query = args.n_query
EPISODE = args.episode
LEARNING_RATE1 = args.learning_rate1
LEARNING_RATE2 = args.learning_rate2

GPU = args.gpu

n_examples = 5  # 训练数据集中每类200个样本
im_width, im_height, depth = 28, 28, 100 


class CNNEncoder(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv3d(1,16,kernel_size=3,padding=1),
                        nn.BatchNorm3d(16),
                        nn.ReLU(),
                        nn.MaxPool3d(kernel_size=(4, 2, 2),padding=1))

        self.layer2 = nn.Sequential(
                        nn.Conv3d(16,32,kernel_size=3,padding=1),
                        nn.BatchNorm3d(32),
                        nn.ReLU(),
                        nn.MaxPool3d(kernel_size=(4, 2, 2),padding=1))
        self.layer3 = nn.Sequential(
                        nn.Conv3d(32,64,kernel_size=3,padding=1),
                        nn.BatchNorm3d(64),
                        nn.ReLU(),
                        nn.MaxPool3d(kernel_size=(4, 2, 2),padding=1))
        self.layer4 = nn.Sequential(
                        nn.Conv3d(64,128,kernel_size=3,padding=1),
                        nn.BatchNorm3d(128),
                        nn.ReLU(),
                        nn.MaxPool3d(kernel_size=(4, 2, 2),padding=1))       
        self.layer5 = nn.Sequential(
                        nn.Conv3d(128,64,kernel_size=(1, 3, 3),padding=0),
                        nn.BatchNorm3d(64),
                        nn.ReLU()
        )




    def forward(self,x):
        out1 = self.layer1(x)
        # print("out1:",out1.size())
        out2 = self.layer2(out1)
        # print("out2:",out2.size())
        out3 = self.layer3(out2)
        # print("out3:",out3.size())
        out4 = self.layer4(out3)
        # print("out4:",out4.size())
        out = self.layer5(out4)
        # print("out5:",out.size())


        return out 


class PrototypeGenerator(nn.Module):
    def __init__(self):
        super(PrototypeGenerator, self).__init__()
        self.fc1 = nn.Linear(84, 64)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # 将输入展平为一维张量
        x = x.view(x.size(0), -1)
        # 第一个全连接层
        out = F.relu(self.fc1(x))

        out = self.dropout(out)

        return out


def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

def train(im_width, im_height, depth):


    feature_encoder = CNNEncoder()
    Prototype_network = PrototypeGenerator()

    feature_encoder.apply(weights_init)
    Prototype_network.apply(weights_init)

    feature_encoder.cuda(GPU)
    Prototype_network.cuda(GPU)

    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=LEARNING_RATE1)
    feature_encoder_scheduler = StepLR(optimizer=feature_encoder_optim, step_size=700, gamma=0.5)
    Prototype_network_optim = torch.optim.Adam(Prototype_network.parameters(), lr=LEARNING_RATE2)
    Prototype_network_scheduler = StepLR(Prototype_network_optim, step_size=700, gamma=0.5)

    feature_encoder.load_state_dict(torch.load(str("./model/meta_training_feature_encoder_20way_1shot_newmodel.pkl"), map_location='cuda:0'))
    print("load feature encoder success")

    Prototype_network.load_state_dict(torch.load(str("./model/meta_training_network_20way_1shot_newmodel.pkl"), map_location='cuda:0'))
    print("load Prototype_network success")

    feature_encoder.train()
    Prototype_network.train()


    # 训练数据集
    f = h5py.File('data/SA/SA_' + str(im_width) + '_' + str(im_height) + '_' + str(depth) + '_support' + str(n_examples) + '.h5', 'r')
    train_dataset = f['data_s'][:]
    f.close()
    train_dataset = train_dataset.reshape(-1, n_examples, im_width, im_height, depth)  
    train_dataset = train_dataset.transpose((0, 1, 4, 2, 3))[:, :, np.newaxis, :, :, :]
    #print(train_dataset)
    #print(train_dataset.shape) # (78, 200, 1, 100, 9, 9) #
    n_train_classes = train_dataset.shape[0]

    A = time.time()
    for episode in range(EPISODE):



        # start:每一个episode的采样过程##########################################################################################
        epi_classes = np.random.permutation(n_train_classes)[:n_way]  
        support = np.zeros([n_way, n_shot, 1, depth, im_height, im_width], dtype=np.float32)  
        query = np.zeros([n_way, n_query,  1, depth, im_height, im_width], dtype=np.float32)  
        # (N,C_in,D_in,H_in,W_in)

        for i, epi_cls in enumerate(epi_classes):
            selected = np.random.permutation(n_examples)[:n_shot + n_query] # 支撑集合
            support[i] = train_dataset[epi_cls, selected[:n_shot]]
            query[i] = train_dataset[epi_cls, selected[n_shot:]]

        support = support.reshape(n_way * n_shot, 1, depth, im_height, im_width)
        query = query.reshape(n_way * n_query, 1, depth, im_height, im_width)
        labels = np.tile(np.arange(n_way)[:, np.newaxis], (1, n_query)).astype(np.uint8).reshape(-1)
        # print("labels",labels)

        support_labels = np.arange(n_way)
        support_labels_tensor = torch.tensor(support_labels, dtype=torch.long, device='cuda')
        one_hot_labels = torch.nn.functional.one_hot(support_labels_tensor)
        # print("one_hot_labels1:",one_hot_labels.size())

        # # 创建新数组，填充零
        # # 创建一个形状为 [16, 20] 的零张量，位于 CUDA 设备上
        new_one_hot_labels = torch.zeros((16, 20), device=one_hot_labels.device)

        # 将原始张量的值复制到新张量的左侧部分
        new_one_hot_labels[:, :16] = one_hot_labels

        one_hot_labels = new_one_hot_labels.view(16, 20, 1, 1)
        # print("one_hot_labels",one_hot_labels.size())
        

        support_tensor = torch.from_numpy(support)
        query_tensor = torch.from_numpy(query)
        label_tensor = torch.tensor(labels, dtype=torch.long, device='cuda')

        # end:每一个episode的采样过程##########################################################################################
        # calculate features
        sample_features = feature_encoder(Variable(support_tensor).cuda(GPU))  
        # print("sample_features1",sample_features.size())
        sample_features = sample_features.view(n_way, n_shot, list(sample_features.size())[-4], list(sample_features.size())[-3],list(sample_features.size())[-2], list(sample_features.size())[ -1])
        # print("sample_features2",sample_features.size())
        sample_features = torch.mean(sample_features, 1).squeeze(1)  
        # print("sample_features3",sample_features.size())
        batch_features = feature_encoder(Variable(query_tensor).cuda(GPU))  

        ################################################################################################################
        sample_features = sample_features.view(n_way, list(sample_features.size())[1]*list(sample_features.size())[2],list(sample_features.size())[-2], list(sample_features.size())[-1])
        # print("sample_features后维度",sample_features.size())

        batch_features = batch_features.view(n_way*n_query, list(batch_features.size())[1] * list(batch_features.size())[2],
                                               list(batch_features.size())[-2], list(batch_features.size())[-1])

      
        sample_features_with_labels = torch.cat([sample_features, one_hot_labels], dim=1)
        prototypes = Prototype_network(sample_features_with_labels)
        batch_features = batch_features.squeeze(-1).squeeze(-1)

        crossEntropy = nn.CrossEntropyLoss().cuda(GPU)
        logits = euclidean_metric(batch_features, prototypes)
        loss = crossEntropy(logits, label_tensor.cuda().long())


        # training
        # 把模型中参数的梯度设为0
        feature_encoder.zero_grad()
        Prototype_network.zero_grad()
        loss.backward()


        # 进行优化
        feature_encoder_optim.step()
        Prototype_network_optim.step()

        feature_encoder_scheduler.step()
        Prototype_network_scheduler.step()


        if (episode+1)%100 == 0:
            print("episode:",episode+1,"loss",loss)
            #################调试#################
            _, predict_label = torch.max(logits, 1)
            predict_label = predict_label.cpu().numpy().tolist()
            #print(predict_label)
            #print(labels)
            rewards = [1 if predict_label[j] == labels[j] else 0 for j in range(labels.shape[0])]
            # print(rewards)
            total_rewards = np.sum(rewards)
            # print(total_rewards)

            accuracy = total_rewards*100.0 / labels.shape[0]
            print("accuracy:", accuracy)
    print(time.time()-A)

    torch.save(feature_encoder.state_dict(),str('./model/SA_feature_encoder_16way_1shot_newmodel_' + str(LEARNING_RATE2)+'_' + str(n_examples) + 'FT_' + str(EPISODE) + '.pkl'))
    torch.save(Prototype_network.state_dict(),str('./model/SA_network_16way_1shot_newmodel_' +str(LEARNING_RATE2)+'_' + str(n_examples) + 'FT_' + str(EPISODE) + '.pkl'))



if __name__ == '__main__':
    train(im_width, im_height, depth)



