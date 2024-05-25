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
from sklearn import metrics

torch.backends.cudnn.benchmark = True

def kappa(testData, k): #testData表示要计算的数据，k表示数据矩阵的是k*k的
    dataMat = np.mat(testData)
    s = dataMat.sum()
    #print(dataMat.shape)
    print(dataMat)
    P0 = 0.0
    for i in range(k):
        P0 += dataMat[i, i]*1.0
    xsum = np.sum(dataMat, axis=1)
    ysum = np.sum(dataMat, axis=0)
    Pe = float(ysum * xsum) / float(s ** 2)
    print("Pe = ", Pe)
    P0 = float(P0/float(s*1.0))
    #print("P0 = ", P0)
    cohens_coefficient = float((P0-Pe)/(1-Pe))

    a = []
    a = dataMat.sum(axis=0)
    a = np.float32(a)
    a = np.array(a)
    a = np.squeeze(a)

    print(a)

    for i in range(k):
        a[i] = float(dataMat[i, i]*1.0)/float(a[i]*1.0)
    print(a*100)
    print("AA: ", a.mean()*100)
    return cohens_coefficient, a.mean()*100, a*100


parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-w","--n_way",type = int, default = 9)                      # way
parser.add_argument("-s","--n_shot",type = int, default = 5)       # support set per class
#-----------------------------------------------------------------------------------#
parser.add_argument("-l","--learning_rate2", type = float, default = 0.006)#seed=1
parser.add_argument("-g","--gpu",type=int, default=0)
args = parser.parse_args()

LEARNING_RATE2 = args.learning_rate2
n_way = args.n_way
n_shot = args.n_shot
GPU = args.gpu

im_width, im_height, channels = 28, 28, 100


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
        out5 = self.layer5(out4)
        # print("out5:",out5.size())


        return out5

class PrototypeGenerator(nn.Module):
    def __init__(self):
        super(PrototypeGenerator, self).__init__()
        self.fc1 = nn.Linear(84, 64)
        self.dropout = nn.Dropout(p=0.5)

        

    def forward(self, x):
        # 将输入展平为一维张量
        x = x.view(x.size(0), -1)
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


feature_encoder = CNNEncoder()
Prototype_network = PrototypeGenerator()

feature_encoder.cuda(GPU)
Prototype_network.cuda(GPU)


feature_encoder.load_state_dict(torch.load(str("./model/UP_feature_encoder_9way_1shot_newmodel_"+ str(LEARNING_RATE2)+"_5FT_1000.pkl")))
print("load feature encoder success")

Prototype_network.load_state_dict(torch.load(str("./model/UP_network_9way_1shot_newmodel_"+ str(LEARNING_RATE2)+"_5FT_1000.pkl")))
print("load Prototype_network success")

feature_encoder.eval()
Prototype_network.eval()



def rn_predict(support_images, test_images, num):

    support_tensor = torch.from_numpy(support_images)
    query_tensor = torch.from_numpy(test_images)

    support_labels = np.arange(n_way)
    support_labels_tensor = torch.tensor(support_labels, dtype=torch.long, device='cuda')
    one_hot_labels = torch.nn.functional.one_hot(support_labels_tensor)

    # # 创建新数组，填充零
    # # 创建一个形状为 [16, 20] 的零张量，位于 CUDA 设备上
    new_one_hot_labels = torch.zeros((9, 20), device=one_hot_labels.device)

    # 将原始张量的值复制到新张量的左侧部分
    new_one_hot_labels[:, :9] = one_hot_labels


    one_hot_labels = new_one_hot_labels.view(9, 20, 1, 1)
    # print("one_hot_labels",one_hot_labels.size())



    # calculate features
    sample_features = feature_encoder(Variable(support_tensor).cuda(GPU))  # 数量*通道*高度*宽度  
    sample_features = sample_features.view(n_way, n_shot, list(sample_features.size())[-4],
                                           list(sample_features.size())[-3],
                                           list(sample_features.size())[-2], list(sample_features.size())[
                                               -1]) 
    sample_features = torch.mean(sample_features, 1).squeeze(1)  # 同类样本取平均
    batch_features = feature_encoder(Variable(query_tensor).cuda(GPU))  
  

    ################################################################################################################
    sample_features = sample_features.view(n_way, list(sample_features.size())[1] * list(sample_features.size())[2],
                                           list(sample_features.size())[-2], list(sample_features.size())[-1])
    batch_features = batch_features.view(num,
                                         list(batch_features.size())[1] * list(batch_features.size())[2],
                                         list(batch_features.size())[-2], list(batch_features.size())[-1])
      
    # 拼接标签与特征
    sample_features_with_labels = torch.cat([sample_features, one_hot_labels], dim=1)
    prototypes = Prototype_network(sample_features_with_labels)

    batch_features = batch_features.squeeze(-1).squeeze(-1)
    
    #欧氏距离
    logits = euclidean_metric(batch_features, prototypes)

    # 得到预测标签
    _, predict_label = torch.max(logits, 1)
    # print('predict_label', predict_label)

    return predict_label


def test(im_width, im_height, channels):

    #A = time.time()
    # 加载支撑数据
    f = h5py.File('data/UP/UP_' + str(im_width) + '_' + str(im_height) + '_' + str(channels) + '_support' + str(args.n_shot) + '.h5', 'r')
    support_images = np.array(f['data_s'])  
    support_images = support_images.reshape(-1, im_width, im_height, channels).transpose((0, 3, 1, 2))[:, np.newaxis, :, :, :]
    print('support_images = ', support_images.shape)  
    f.close()

    # 加载测试
    f = h5py.File(r'.\data\UP\UP_28_28_100_test.h5', 'r')  # 路径
    test_images = np.array(f['data'])  
    test_images = test_images.reshape(-1, im_width, im_height, channels).transpose((0, 3, 1, 2))[:, np.newaxis, :, :, :]
    print('test_images = ', test_images.shape) 
    test_labels = f['label'][:] 
    f.close()

    predict_labels = []  # 记录预测标签
    # S1
    for i in range(0, 4277):#10988 42776 10249 54129
        test_images_ = test_images[10 * i:10 * (i + 1), :, :, :, :]
        predict_label = rn_predict(support_images, test_images_, num = 10)
        predict_labels.extend(predict_label.cpu().numpy().tolist())

    # S2
    test_images_ = test_images[-6:, :, :, :, :]
    predict_label = rn_predict(support_images, test_images_, num = 6)
    predict_labels.extend(predict_label.cpu().numpy().tolist())

    # S3
    print(np.unique(predict_labels))
    rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(test_images.shape[0])]

    ##################### 混淆矩阵 #####################
    from sklearn import metrics
    matrix = metrics.confusion_matrix(test_labels, predict_labels)
    print(matrix)
    OA = np.sum(np.trace(matrix)) / 42776.0 * 100
    print('OA = ', round(OA, 2))

    ################################################################
    n = 42776
    matrix = np.zeros((9, 9), dtype=int)
    print(len(predict_labels))
    for j in range(n):
        matrix[test_labels[j], predict_labels[j]] += 1  # 构建混淆矩阵
    print("OA: ", np.sum(np.trace(matrix)) / float(n) * 100)

    from sklearn import metrics
    kappa_true = metrics.cohen_kappa_score(test_labels, predict_labels)
    print("kappa_ture",kappa_true * 100)


    kappa_temp, aa_temp, ca_temp = kappa(matrix, 9)
    print("kappa_temp",kappa_temp * 100)
    f = open(f"UP/UP_" + str(LEARNING_RATE2) + '_' + str(np.sum(np.trace(matrix)) / float(n) * 100) + ".txt", 'w')   
    for index in range(len(ca_temp)):
        f.write(str(ca_temp[index]) + '\n')
    f.write(str(np.sum(np.trace(matrix)) / float(n) * 100) + '\n')
    f.write(str(aa_temp) + '\n')
    f.write(str(kappa_true * 100) + '\n')

    from scipy.io import loadmat
    gt = loadmat('D:\HSI_Projects\hyperspectral_data\PaviaU\PaviaU_gt.mat')['paviaU_gt']

    # 将预测的结果匹配到图像中，生成一个类别标签图像 new_show
    new_show = np.zeros((gt.shape[0], gt.shape[1]), dtype=int)
    k = 0
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
                if gt[i][j] != 0:
                   new_show[i][j] = predict_labels[k]
                   new_show[i][j] += 1
                   k += 1

    # # 将预测的结果匹配到图像中
    hsi_pic = np.array([
    [0, 0, 0],    # 黑色
    [0, 0, 1],    # 蓝色
    [0, 1, 0],    # 绿色
    [0, 1, 1],    # 青色
    [1, 0, 0],    # 红色
    [1, 0, 1],    # 紫色
    [1, 1, 0],    # 黄色
    [0.5, 0.5, 1], # 浅蓝色
    [0.65, 0.35, 1],# 深紫色
    [0.75, 0.5, 0.75],# 浅紫色
    ])





    # # 展示地物
    import matplotlib.pyplot as plt
    import matplotlib.colors as mpl

    cmap = mpl.ListedColormap(hsi_pic)

    # 展示地物
    plt.xticks([])  # 移除 x 轴的刻度
    plt.yticks([])  # 移除 y 轴的刻度
    plt.imshow(new_show, cmap=cmap)
    # 使用constrained_layout=True自动调整布局
    plt.gcf().set_tight_layout('tight')

    # 保存图像，使用pad_inches=0避免额外空间
    plt.savefig(f"UP/UP_{LEARNING_RATE2}_"
             f"{str(np.sum(np.trace(matrix)) / float(n) * 100)}.png",
             dpi=1000, bbox_inches='tight', pad_inches=0)  # 保存图像
    plt.close()


if __name__ == '__main__':
    test(im_width, im_height, channels)
    

