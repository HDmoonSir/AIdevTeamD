import torch
import pandas as pd
import numpy as np
# from konlpy.tag import Okt
from tqdm import tqdm
from nltk import FreqDist

import os

class TextFileCsvLabelDataset(torch.utils.data.Dataset):
    def __init__(self, csvfile_path , tokenizer_type=None, stop_words=None, option=None):
        self.csv_data = pd.read_csv(csvfile_path)
        self.tokenizer_type = tokenizer_type
        self.stopwords = stop_words
        # 결측치 제거
        self.korean_train_df = self.csv_data.dropna(how='any')
        self.option = option
        
        self.corpus = []
        for data in tqdm(self.korean_train_df['text'], desc="tokenizing and remove stopwords"):
            tokenized_data = self.tokenizer_type.morphs(data, stem=True) # 토큰화 다른거 쓰면 바꿔야 할 수 있음
            stopwords_removed_data = [word for word in tokenized_data if not word in self.stopwords] # 불용어 제거
            self.corpus.append(stopwords_removed_data)
        
        # 단어집 만들기
        self.vocab = FreqDist(np.hstack(self.corpus))
        
        # 필요하다면 vocab 축소가 필요함
        
        # word to int
        self.word_to_index = {word[0] : index + 2 for index, word in enumerate(tqdm(self.vocab, desc="word to int"))}
        self.word_to_index['pad'] = 1
        self.word_to_index['unk'] = 0
        
        self.max_corpus_len = 0
        
        self.encoded = []
        for line in tqdm(self.corpus, desc="encoding"): #입력 데이터에서 1줄씩 문장을 읽음
            temp = []
            for w in line: #각 줄에서 1개씩 글자를 읽음
                try:
                    temp.append(self.word_to_index[w]) # 글자를 해당되는 정수로 변환
                except KeyError: # 단어 집합에 없는 단어일 경우 unk로 대체된다.
                    temp.append(self.word_to_index['unk']) # unk의 인덱스로 변환
            
            if len(temp) > self.max_corpus_len:
                self.max_corpus_len = len(temp)
                
            self.encoded.append(temp)
        
        for line in tqdm(self.encoded, desc="padding"):
            if len(line) < self.max_corpus_len: # 현재 샘플이 정해준 길이보다 짧으면
                line += [self.word_to_index['pad']] * (self.max_corpus_len - len(line)) # 나머지는 전부 'pad' 토큰으로 채운다.
        
    def __len__(self):
        # return len(self.csv_data)
        return len(self.korean_train_df)
    
    def __getitem__(self, idx):
        encoded_data = self.encoded[idx]
        encoded_data = torch.tensor(encoded_data, dtype=torch.int64)
        
        if self.option =='train':
            # label[2]
            label = self.korean_train_df.iloc[idx,2]
            label = torch.tensor(label, dtype=torch.int64)
            return encoded_data, label
        
        return encoded_data
    
    def vocav_return(self):
        return self.vocab
    
    def vocav_len_return(self):
        return len(self.vocab)
    
    def max_corpus_len_return(self):
        return self.max_corpus_len
    
    def unique_label(self):
        return self.korean_train_df['label'].nunique()