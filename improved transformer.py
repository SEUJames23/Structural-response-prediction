import copy
import glob
import math
import time
from matplotlib import pyplot as plt
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import sklearn.metrics as sm

epochs = 15000
d_model = 16  # Embedding Size
d_ff = 128 # FeedForward dimension
d_k = d_v = 2  # dimension of K(=Q), V
n_layers = 3  # number of Encoder of Decoder Layer
n_heads = 8  # number of heads in Multi-Head Attention
num_steps = 1500
input_size = output_size = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('You are using: ' + str(device))
 
 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        #pe的维度是（5000，512）
        pe = torch.zeros(max_len, d_model)
        #position是一个5000行1列的tensor
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        #div_term是一个256长度的一维tensor
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        #最终的pe是一个torch.Size([5000, 1, 512])的维度
        self.register_buffer('pe', pe)
 
    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

 
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
 
    def forward(self, Q, K, V):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        # print(attn.shape)
        # print(V.shape)
        # torch.Size([2, 8, 5, 5])
        # torch.Size([2, 8, 5, 64])
        return context, attn
 
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        #d_k * n_heads     64 * 8
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
 
    #input_Q  （2，5，512）    attn_mask （2，5，5）
    def forward(self, input_Q, input_K, input_V):
            # '''
            # input_Q: [batch_size, len_q, d_model] （2，5，512）
            # input_K: [batch_size, len_k, d_model]
            # input_V: [batch_size, len_v(=len_k), d_model]
            # attn_mask: [batch_size, seq_len, seq_len]
            # '''
        #print("input_Q的维度", input_Q.shape)
        residual, batch_size = input_Q, input_Q.size(0)
            # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
            # D_new这个新的维度就是原本的维度 × n个头，也就是
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)
            # Q: [batch_size, n_heads, len_q, d_k]
            #（2，5，512）-> (2,5,8,64) ->
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1,2)
            # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,2)
            # V: [batch_size, n_heads, len_v(=len_k), d_v]
           # torch.Size([2, 5, 5]) -》([2, 8, 5, 5]) 也就是复制了几份
        # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V)
        # (2,8,5,64)
        # (2,8,5,5)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)
        # context: [batch_size, len_q, n_heads * d_v]
        #2 8 5 64 -> 2 5 8 64 -> 2 5 512
        #self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        #8 * 64 -> 512
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(d_model).cuda()(output + residual), attn
 
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).cuda()(output + residual) # [batch_size, seq_len, d_model]
 
 
# enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()
 
    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn
 
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Linear(input_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        #返回的是一个二维的矩阵
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
 
    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        enc_outputs = self.src_emb(enc_inputs.reshape(-1, num_steps, input_size)) # [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1) # [batch_size, src_len, d_model]
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns
 
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder().cuda()
        self.projection = nn.Linear(d_model, output_size, bias=False).cuda()
 
    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        '''
        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)
        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        outputs = self.projection(enc_outputs)  # dec_logits: [batch_size, num_steps, output_size]
        return outputs, enc_self_attns

##数据集
fileset = glob.glob("artdata/traindata*")
trainX_, trainY_ = [], []
for file in fileset:
    traindata = np.loadtxt(file, dtype=float, unpack=True)
    train_ag, train_disp = traindata[0][:num_steps], traindata[1][:num_steps]
    trainX_.append(train_ag)
    trainY_.append(train_disp)
trainX_, trainY_ = np.array(trainX_), np.array(trainY_)

delta = np.max(trainX_)-np.min(trainX_)
trainX = (trainX_ - np.ones_like(trainX_)*np.min(trainX_))/delta
delta2 = np.max(trainY_)-np.min(trainY_)
trainY = (trainY_ - np.ones_like(trainY_)*np.min(trainY_))/delta2

testdata = np.loadtxt("artdata/testdata1.txt", dtype=float, unpack=True)
test_ag, test_disp = testdata[0][:num_steps], testdata[1][:num_steps]
testX = test_ag
testX = (testX - np.ones_like(testX)*np.min(trainX_))/delta
testY = test_disp
testY = (testY - np.ones_like(testY)*np.min(trainY_))/delta2
class traindataset(Dataset):
    def __init__(self):
        self.features = trainX
        self.labels = trainY
        self.len = len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return self.len

dataset_train = traindataset()
train_loader = DataLoader(dataset_train, batch_size=6, shuffle=True)

##训练
model = Transformer()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.1)
criterion = nn.MSELoss()
criterion = criterion.to(device)

best_loss = 100
best_model = None
start = time.time()
loss_sec = []
for epoch in range(epochs):
    loss_epoch = 0
    for trainx, trainy in train_loader:
        trainx = trainx.to(torch.float32).to(device)
        trainy = trainy.reshape(-1, num_steps, output_size).to(torch.float32).to(device)
        predy, self_attn = model(trainx)
        loss = criterion(predy, trainy)
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # back propagation, compute gradients
        optimizer.step()  # apply gradients
        loss_epoch += loss.item()
    loss_sec.append(loss_epoch)
    if epoch % 10 == 0:
        print('Train Epoch:{}\tLoss:{:.9f}'.format(epoch, loss_epoch))
    if loss_epoch<best_loss:
        best_loss = loss_epoch
        best_model = copy.deepcopy(model)
end = time.time()
print('Running time: %s S'%(end-start))
print( "best_loss:", best_loss)
torch.save(best_model.state_dict(), 'transformer3.params')

with open("tran_loss.txt", "w") as f:
    for loss in loss_sec:
        f.writelines("%f\n" %loss)

##测试
testx, testy = testX, testY
testx = torch.from_numpy(testx.reshape(-1, num_steps, output_size)).to(torch.float32).to(device)
predy, attn = best_model(testx)
predy = predy.detach().cpu().numpy()
predy = predy.reshape(-1, 1)

# 对比图
plt.plot(testy, c='green')
plt.plot(predy, c='orange')
plt.show()

