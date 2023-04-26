import pandas as pd
from hanspell import spell_checker

import argparse
parser = argparse.ArgumentParser(description='trainingKoreanTextData')
parser.add_argument('--dataDir', type=str, default='data/koreanLanguageData')
opt = parser.parse_args()
# dataload and preprocessing

korean_train_df = pd.read_csv(opt.dataDir + '/train_stopword.csv')
print(korean_train_df[:5])

for i, text in enumerate(korean_train_df['text']):
    text = spell_checker.check(text=text)
    korean_train_df['text'][i] = text.checked

korean_train_df.to_csv(opt.dataDir + "/train_spacedByHanSpell.csv",index=False)
