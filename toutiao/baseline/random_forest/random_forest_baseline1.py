
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from icecream import ic
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score


TRAIN_CORPUS = 'E:\\AI_workspace\\ai-study\\toutiao\\data\\train_new.csv'

STOP_WORDS = 'E:\\AI_workspace\\ai-study\\toutiao\\data\\stopwords.txt'

WORD_COLUMN = 'words'

content = pd.read_csv(TRAIN_CORPUS)
corpus = content[WORD_COLUMN].values

stop_words_size = 749
WORD_LONG_TAIL_BEGIN = 1000
WORD_SIZE = WORD_LONG_TAIL_BEGIN - stop_words_size

stop_words = open(STOP_WORDS,encoding='utf-8').read().split()[: stop_words_size]

tfidf = TfidfVectorizer(max_features=WORD_SIZE, stop_words=stop_words)

text_vectors = tfidf.fit_transform(corpus)

print(text_vectors.shape)

targets = content['label']

x_train,x_test,y_train,y_test = train_test_split(text_vectors,targets,test_size=0.2,random_state=0)

print('数据分割完毕，开始模型训练...')

model = RandomForestClassifier()

model.fit(x_train,y_train)

print('模型预测结束，开始预测...')

accuracy = accuracy_score(model.predict(x_test),y_test)

ic(accuracy)



