import torch.nn as nn
import torch

class Discriminator(nn.Module):
    def __init__(self, n_layers, hidden_dim, n_vocab, embed_dim, n_classes, dropout_p=0.2):
        super(Discriminator, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embed = nn.Embedding(n_vocab, embed_dim)
        self.dropout = nn.Dropout(dropout_p)
        # self.gru = nn.GRU(embed_dim, self.hidden_dim,
        #                   num_layers=self.n_layers,
        #                   batch_first=True)
        # self.gru = nn.LSTM(embed_dim, self.hidden_dim,
        #                   num_layers=self.n_layers,
        #                   batch_first=True,
        #                   bidirectional=True)
        self.lstm = nn.LSTM(embed_dim, self.hidden_dim,
                          num_layers=self.n_layers,
                          batch_first=True)
        self.out = nn.Linear(self.hidden_dim, n_classes)
        # self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.embed(x)
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).cuda()
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).cuda()
        
        # h_0 = self._init_state(batch_size=x.size(0)) # 첫번째 히든 스테이트를 0벡터로 초기화
        # x, _ = self.lstm(x, h_0)  # GRU의 리턴값은 (배치 크기, 시퀀스 길이, 은닉 상태의 크기)
        # x, _ = self.lstm(x)  # GRU의 리턴값은 (배치 크기, 시퀀스 길이, 은닉 상태의 크기)
        x, _ = self.lstm(x, (h0, c0))
        h_t = x[:,-1,:] # (배치 크기, 은닉 상태의 크기)의 텐서로 크기가 변경됨. 즉, 마지막 time-step의 은닉 상태만 가져온다.
        self.dropout(h_t)
        # logit = self.sig(self.out(h_t))  # (배치 크기, 은닉 상태의 크기) -> (배치 크기, 출력층의 크기)
        logit = self.out(h_t)
        # print(logit)
        return logit

    # def _init_state(self, batch_size=1):
    #     weight = next(self.parameters()).data
    #     return weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()