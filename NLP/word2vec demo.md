# word2vec模型


```python
from gensim.models.word2vec import Word2Vec
import gensim.downloader
```


```python
list(gensim.downloader.info()['models'].keys())
```




    ['fasttext-wiki-news-subwords-300',
     'conceptnet-numberbatch-17-06-300',
     'word2vec-ruscorpora-300',
     'word2vec-google-news-300',
     'glove-wiki-gigaword-50',
     'glove-wiki-gigaword-100',
     'glove-wiki-gigaword-200',
     'glove-wiki-gigaword-300',
     'glove-twitter-25',
     'glove-twitter-50',
     'glove-twitter-100',
     'glove-twitter-200',
     '__testing_word2vec-matrix-synopsis']




```python
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
word_vectors = KeyedVectors.load_word2vec_format(datapath(r"/Users/zheyiwang/gensim-data/word2vec-google-news-300/GoogleNews-vectors-negative300.bin"), binary=True)
```

# 训练Word2vec模型


```python
sentences =[['猫','吃','鱼'],['狗','吃','肉']] # 构造数据集
model = Word2Vec(sentences, min_count=1, sg=1) # 训练模型,min_count最小词频
model_path = 'model/demo.model'
model.save(model_path)
```

# 词向量


```python
model = Word2Vec.load(model_path) #加载模型
model.wv['猫']
```




    array([-0.00713902,  0.00124103, -0.00717672, -0.00224462,  0.0037193 ,
            0.00583312,  0.00119818,  0.00210273, -0.00411039,  0.00722533,
           -0.00630704,  0.00464722, -0.00821997,  0.00203647, -0.00497705,
           -0.00424769, -0.00310898,  0.00565521,  0.0057984 , -0.00497465,
            0.00077333, -0.00849578,  0.00780981,  0.00925729, -0.00274233,
            0.00080022,  0.00074665,  0.00547788, -0.00860608,  0.00058446,
            0.00686942,  0.00223159,  0.00112468, -0.00932216,  0.00848237,
           -0.00626413, -0.00299237,  0.00349379, -0.00077263,  0.00141129,
            0.00178199, -0.0068289 , -0.00972481,  0.00904058,  0.00619805,
           -0.00691293,  0.00340348,  0.00020606,  0.00475375, -0.00711994,
            0.00402695,  0.00434743,  0.00995737, -0.00447374, -0.00138926,
           -0.00731732, -0.00969783, -0.00908026, -0.00102275, -0.00650329,
            0.00484973, -0.00616403,  0.00251919,  0.00073944, -0.00339215,
           -0.00097922,  0.00997913,  0.00914589, -0.00446183,  0.00908303,
           -0.00564176,  0.00593092, -0.00309722,  0.00343175,  0.00301723,
            0.00690046, -0.00237388,  0.00877504,  0.00758943, -0.00954765,
           -0.00800821, -0.0076379 ,  0.00292326, -0.00279472, -0.00692952,
           -0.00812826,  0.00830918,  0.00199049, -0.00932802, -0.00479272,
            0.00313674, -0.00471321,  0.00528084, -0.00423344,  0.0026418 ,
           -0.00804569,  0.00620989,  0.00481889,  0.00078719,  0.00301345],
          dtype=float32)


