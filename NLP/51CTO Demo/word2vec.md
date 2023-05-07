```python
from gensim.models import word2vec
import logging

logging.basicConfig(format='%(asctime)s:%(message)s', level=logging.INFO)
```


```python
raw_sentences = ['the quick brown fox jumps over the lazy dogs','yoyoyo you go home now to sleep']
```


```python
sentences = [ s.split() for s in raw_sentences]
print(sentences)
```


```python
model = word2vec.Word2Vec(sentences, min_count=1)
```


```python
# min_count表示忽略出现大于等于该词频的单词。一般设置为0-100之间
```


```python
model.wv.similarity('dogs','you')
```


```python
# 以下是维基百科中文读取
```


```python
import jieba
import jieba.analyse
import jieba.posseg as pseg
import codecs,sys
```


```python
def cut_words(sentence):
    return " ".join(jieba.cut_words(sentence)).encode('utf-8')
```


```python
f=codecs.open('/Users/zheyiwang/Downloads/word2vec/wiki_data/wiki.zh.jian.text','r',encoding='utf8')
target = codecs.open('/Users/zheyiwang/Downloads/word2vec/wiki_data/wiki.seg-1.3g.txt','w',encoding='utf8')
print('open files')
```


```python
# 转换为简体字并分词
```


```python
line_num=1
line = f.readline()
while line:
    print('---processing',line_num,'artical-----------')
    line_seg =" ".join(jieba.cut(line))
    target.writelines(line_seg)
    line_num = line_num + 1
    line = f.readline()
f.close()
target.close()
exit()
```


```python
# 建模
```


```python
import logging
import os.path
import sys
import multiprocessing
from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
```


```python
inp = '/Users/zheyiwang/Downloads/word2vec/wiki_data/wiki.seg-1.3g.txt'
outp1 = '/Users/zheyiwang/Downloads/word2vec/wiki_data/wiki.zh.text.model'
outp2 = '/Users/zheyiwang/Downloads/word2vec/wiki_data/wiki.zh.text.vector'
model = Word2Vec(LineSentence(inp), window=5, min_count=5, workers=multiprocessing.cpu_count())
model.save(outp1)
model.model.wv.save_word2vec_format(outp2, binary=False)
print("done")
```


```python

```
