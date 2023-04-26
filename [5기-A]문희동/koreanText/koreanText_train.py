import numpy as np
import pandas as pd
import torch
import random
import os
import time

from konlpy.tag import Okt
# from torchtext.legacy import data
# from torchtext.legacy.data import TabularDataset
# from torchtext.data import TabularDataset

from dataset import TextFileCsvLabelDataset
from discriminator import Discriminator

import torch.optim as optim
import torch.nn as nn

# Training settings
import argparse
parser = argparse.ArgumentParser(description='trainingKoreanTextData')
parser.add_argument('--explain', type=str, default='trainingKoreanTextData')
parser.add_argument('--trainSet', type=str, default='OR', help='choose trainSets for training')
parser.add_argument('--data', type=str, default='koreanLanguageData')
parser.add_argument('--dataDir', type=str, default='data/koreanLanguageData')
parser.add_argument('--trainBatchSize', type=int, default=20, help='training batch size')
# parser.add_argument('--testBatchSize', type=int, default=20, help='test batch size')
parser.add_argument('--hiddenDim', type=int, default=128, help='hidden dim size')
parser.add_argument('--nEpochs', type=int, default=10, help='number of epochs to train for')
opt = parser.parse_args()

# information for save
# thisPath = os.path.dirname(os.path.realpath(__file__))
timestamp = time.strftime("%y%m%d_%H%M%S", time.localtime())

path_tar = './result_exp/resultWeight_%s_%s_%s_%s' % \
    (opt.explain ,timestamp, opt.data, opt.trainSet)

os.makedirs(path_tar)
baseName = '%s/basicLSTM_' % path_tar 
logName = baseName + opt.data + '_result'
filename_log = '%s.log' % (logName)
f = open(filename_log, 'w')

with open(__file__,'rt',encoding='UTF8') as fi: 
    f.write('\n'.join(fi.read().split('\n')[0:]))

# cuda setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# pytorch seed setting
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True

# legacy version
# TEXT = data.Field(sequential=True,
#                   use_vocab=True,
#                   tokenize=okt.morphs, # 토크나이저로는 Mecab 사용.
#                 #   lower=True,
#                   batch_first=True,
#                   fix_length=120)

# LABEL = data.Field(sequential=False,
#                    use_vocab=False,
#                    batch_first=False,
#                    is_target=True)

# field = {
#     'text' : ('text', TEXT),
#     'label' : ('label', LABEL)
# }

# train_data = data.TabularDataset.splits(
#         path='.', train='train_stopword.csv', format='csv',
#         fields=[('text', TEXT), ('label', LABEL)], skip_header=True)
# print(len(train_data))

# clean data load + 형태소 분석 -> 토큰화 -> 토큰화에서 불용어 제거
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
okt = Okt()

train_data = TextFileCsvLabelDataset(csvfile_path = opt.dataDir + '/train_clean.csv',
                                     tokenizer_type = okt,
                                     stop_words = stopwords,
                                     option = 'train')

# print(train_data[:5])
# print(train_data.vocav_len_return()) # 35179
# print(train_data.max_corpus_len_return()) # 116

trainset_OR, validationset_OR = torch.utils.data.random_split(train_data, [len(train_data) - 13000, 13000])

trainset = torch.utils.data.ConcatDataset([trainset_OR])

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=opt.trainBatchSize,
    shuffle=True,
    num_workers=0,
    worker_init_fn=np.random.seed(0),
    pin_memory=True
)

valloader = torch.utils.data.DataLoader(
    validationset_OR,
    batch_size=opt.trainBatchSize,
    shuffle=True,
    num_workers=0,
    worker_init_fn=np.random.seed(0),
    pin_memory=True
)

# discriminator setting
numclasses = train_data.unique_label()

discriminator = Discriminator(n_layers=1, hidden_dim=opt.hiddenDim, n_vocab=train_data.vocav_len_return(), embed_dim=train_data.max_corpus_len_return(), n_classes=numclasses, dropout_p=0.2)
discriminator.to(device)
discriminator.eval()
print(discriminator)

# loss setting
criterion_d = nn.CrossEntropyLoss().cuda()

# optimizer setting
optimizer_d = optim.Adam(discriminator.parameters())

for epoch in range(opt.nEpochs):  # loop over the dataset multiple times
    running_loss_d = 0.0    
    running_corrects_d = 0
    discriminator.train()
    
    for i, data in enumerate(trainloader, 0):
        
        corpus, labels = data

        corpus = corpus.cuda()
        labels = labels.cuda()
        # corpus = corpus.to(device)
        # labels = labels.to(device)
        
        print(corpus.shape)
        print(labels.shape)
        # print(corpus)
        # print(labels)
        # labels = labels.float().unsqueeze(1)
        
        optimizer_d.zero_grad()
        aux_d = discriminator(corpus)
        
        print(aux_d.shape)
        print(aux_d)
      
        loss_d = criterion_d(aux_d, labels)
        loss_d.backward()
        optimizer_d.step()
        
        _, pred_d = torch.max(aux_d.data, 1)
        
        print(pred_d.shape)
        print(pred_d)
        print(labels.data)
                        
        running_corrects_d += torch.sum(pred_d == labels.data).tolist()
        
        running_loss_d = loss_d.item()
        
        # recording
        disp_str = 'Train [%d, %d] d_loss: %.3f' % (epoch, i, running_loss_d)
        print(disp_str)
        f.write(disp_str)
        f.write('\n')
        disp_str = 'Train [%d, %d]  Data(OR) : %d / %d' % (epoch, i, running_corrects_d, (i+1)*opt.trainBatchSize)
        
        print(disp_str)
        f.write(disp_str)
        f.write('\n')
    
    outFileName = baseName + 'epoch_' + str(epoch) +'.pt'
    
    discriminator.eval()
    running_corrects_d = 0
    with torch.no_grad():
      for it, data in enumerate(valloader, 0):
          corpus, labels = data
          labels = labels.cuda()
          
          aux_d = discriminator(corpus.cuda())

          _, pred_d = torch.max(aux_d.data, 1)

          running_corrects_d += torch.sum(pred_d == labels.data).tolist()
    
    # recording
    disp_str = 'Val [%d]  OR : %d / %d' % (epoch, running_corrects_d, len(validationset_OR))
    print(disp_str)
    f.write(disp_str)
    f.write('\n')
    
    if running_corrects_d > best_score:
        best_score = running_corrects_d
        best_epoch = epoch
    # save weight    
    print("Save weight : %s" % outFileName)
    
    torch.save({
            'epoch': epoch,
            #'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_d_state_dict': optimizer_d.state_dict(),
            #'optimizer_g_state_dict': optimizer_g.state_dict(),            
            'loss_d': loss_d,
            #'loss_g': loss_g,
    }, outFileName)

disp_str = 'Best Epooch is [%d] epoch : %d / %d' % (epoch, running_corrects_d, len(validationset_OR))
print(disp_str)
f.write(disp_str)
f.write('\n')
        
print('Finished Training')