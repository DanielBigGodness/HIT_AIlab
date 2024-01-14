import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from gensim.test.utils import datapath
import os
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from transformers import BertTokenizer, BertModel
import time
import logging

import jieba


class TextCNN(nn.Module):
    def __init__(self, bert_model, filter_num, sentence_max_size, label_size, kernel_list):
        super(TextCNN, self).__init__()
        self.bert_model = bert_model
        self.convs = nn.ModuleList([
            nn.Conv2d(1, filter_num, (kernel, bert_model.config.hidden_size))
            for kernel in kernel_list
        ])
        self.fc = nn.Linear(len(kernel_list) * filter_num, label_size)

    def forward(self, x):
        in_size = x.size(0)
        x = self.bert_model(x).last_hidden_state.unsqueeze(1)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.view(in_size, -1)
        out = F.dropout(out)
        out = self.fc(out)
        return out




# 修改MyDataset类以使用BERT嵌入
class MyDataset(Dataset):
    def __init__(self, file_list, label_list, sentence_max_size, tokenizer, bert_model, stopwords):
        self.x = file_list
        self.y = label_list
        self.sentence_max_size = sentence_max_size
        self.tokenizer = tokenizer
        self.bert_model = bert_model
        self.stopwords = stopwords

    def __getitem__(self, index):
        words = []
        with open(self.x[index], "r", encoding="utf8") as file:
            for line in file.readlines():
                words.extend(segment(line.strip(), stopwords))
        inputs = self.tokenizer(words, padding='max_length', truncation=True, max_length=self.sentence_max_size,
                                return_tensors="pt")
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            tensor = outputs.last_hidden_state  # BERT嵌入

        return tensor, self.y[index]

    def __len__(self):
        return len(self.x)


    # 加载停用词列表
def load_stopwords(stopwords_dir):
    stopwords = []
    with open(stopwords_dir, "r", encoding="utf8") as file:
        for line in file.readlines():
            stopwords.append(line.strip())
    return stopwords


def segment(content, stopwords):
    res = []
    for word in jieba.cut(content):
        if word not in stopwords and word.strip() != "":
            res.append(word)
    return res


def get_file_list(source_dir):
    file_list = []  # 文件路径名列表
    # os.walk()遍历给定目录下的所有子目录，每个walk是三元组(root,dirs,files)
    # root 所指的是当前正在遍历的这个文件夹的本身的地址
    # dirs 是一个 list ，内容是该文件夹中所有的目录的名字(不包括子目录)
    # files 同样是 list , 内容是该文件夹中所有的文件(不包括子目录)
    # 遍历所有文章
    if os.path.isdir(source_dir):
        for root, dirs, files in os.walk(source_dir):
            file = [os.path.join(root, filename) for filename in files]
            file_list.extend(file)
        return file_list
    else:
        print("the path is not existed")
        exit(0)


def get_label_list(file_list):
    # 提取出标签名
    label_name_list = [file.split("\\")[7] for file in file_list]
    # 标签名对应的数字
    label_list = []
    for label_name in label_name_list:
        if label_name == "neg":
            label_list.append(0)
        elif label_name == "pos":
            label_list.append(1)
    return label_list


def generate_tensor(sentence, sentence_max_size, tokenizer, bert_model):
    """
    为给定句子生成BERT嵌入。
    :param sentence: 句子中的词列表。
    :param sentence_max_size: 最大句子长度。
    :param tokenizer: BERT分词器。
    :param bert_model: BERT模型。
    :return: 句子的BERT嵌入。
    """
    inputs = tokenizer(sentence, padding='max_length', truncation=True, max_length=sentence_max_size, return_tensors="pt")
    outputs = bert_model(**inputs)
    embeddings = outputs.last_hidden_state  # BERT嵌入
    return embeddings


def train_textcnn_model(net, train_loader, epoch, lr):
    print("begin training")
    net.train()  # 必备，将模型设置为训练模式
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for i in range(epoch):  # 多批次循环
        print(enumerate(train_loader))
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()  # 清除所有优化的梯度
            output = net(data)  # 传入数据并前向传播获取输出
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # 打印状态信息
            logging.info("train epoch=" + str(i) + ",batch_id=" + str(batch_idx) + ",loss=" + str(loss.item() / 64))

    print('Finished Training')


def textcnn_model_test(net, test_loader, train_loader):
    net.eval()  # 必备，将模型设置为训练模式
    correct = 0
    total = 0
    test_acc = 0.0
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            logging.info("test batch_id=" + str(i))
            #data = data.to(cuda)
            outputs = net(data)
            # torch.max()[0]表示最大值的值，troch.max()[1]表示回最大值的每个索引
            _, predicted = torch.max(outputs.data, 1)  # 每个output是一行n列的数据，取一行中最大的值
            total += label.size(0)
            correct += (predicted == label).sum().item()
            print('Accuracy of the network on test set : %d %%' % (100 * correct / total))
            # test_acc += accuracy_score(torch.argmax(outputs.data, dim=1), label)
            # logging.info("test_acc=" + str(test_acc))
    with torch.no_grad():
        for i, (data, label) in enumerate(train_loader):
            logging.info("test batch_id=" + str(i))
            #data = data.to(cuda)
            outputs = net(data)
            # torch.max()[0]表示最大值的值，troch.max()[1]表示回最大值的每个索引
            _, predicted = torch.max(outputs.data, 1)  # 每个output是一行n列的数据，取一行中最大的值
            total += label.size(0)
            correct += (predicted == label).sum().item()
            print('Accuracy of the network on train set: %d %%' % (100 * correct / total))
            # test_acc += accuracy_score(torch.argmax(outputs.data, dim=1), label)
            # logging.info("test_acc=" + str(test_acc))

current_dir=os.getcwd()
if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)

    # 加载Bert模型和分词器
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    bert_model = BertModel.from_pretrained("bert-base-cased")

    # 冻结BERT模型的参数
    for param in bert_model.parameters():
        param.requires_grad = False

    # train_dir = "aclIdmb/train"
    # test_dir = "aclIdmb/test"
    # stopwords_dir = "stopwords.txt"
    train_dir = os.path.join(os.getcwd(), "aclIdmb\\train")  # 训练集路径
    test_dir = os.path.join(os.getcwd(), "aclIdmb\\test")  # 测试集路径
    stopwords_dir = os.path.join(os.getcwd(), "stopwords.txt")  # 停用词
    word2vec_dir = os.path.join(os.getcwd(), "bert-base-cased")
    #net_dir = "model/net.pkl"
    net_dir = ".\\model\\net.pkl"
    sentence_max_size = 300  # 每篇文章的最大词数量
    batch_size = 64
    filter_num = 100  # 每种卷积核的个数
    epoch = 1  # 迭代次数
    kernel_list = [3, 4, 5]  # 卷积核的大小
    label_size = 2
    lr = 0.001
    # 加载词向量模型
    logging.info("加载词向量模型")
    # 读取停用表
    stopwords = load_stopwords(stopwords_dir)
    # 加载词向量模型

    # embedding.weight.data.normal_(mean=0.0, std=0.5)
    # requires_grad指定是否在训练过程中对词向量的权重进行微调
    # embedding.weight.requires_grad = True
    # 获取训练数据
    logging.info("获取训练数据")
    train_set = get_file_list(train_dir)
    train_label = get_label_list(train_set)
    train_dataset = MyDataset(train_set, train_label, sentence_max_size, tokenizer, bert_model, stopwords)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 获取测试数据
    logging.info("获取测试数据")
    test_set = get_file_list(test_dir)
    test_label = get_label_list(test_set)
    test_dataset = MyDataset(test_set, test_label, sentence_max_size, tokenizer, bert_model, stopwords)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # 定义模型
    logging.info("定义模型")
    net = TextCNN(bert_model, filter_num, sentence_max_size, label_size, kernel_list)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    # 训练
    logging.info("开始训练模型")
    train_textcnn_model(net, train_dataloader, epoch, lr)

    # 保存模型
    torch.save(net.state_dict(), net_dir)

    # 测试模型
    logging.info("开始测试模型")
    textcnn_model_test(net, test_dataloader, train_dataloader)




