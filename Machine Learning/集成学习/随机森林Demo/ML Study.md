# 数据集


```python
import numpy as np
import matplotlib.pyplot as plt
```


```python
from sklearn.datasets import make_moons
```


```python
x,y= make_moons(
    n_samples=1000,
    noise=0.4,
    random_state=20
)
x.shape,y.shape
```




    ((1000, 2), (1000,))




```python
plt.scatter(x[:,0],x[:,1],c=y,s=10)
plt.show()
```


    
![png](output_4_0.png)
    



```python
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0)
```

# Random Forest Trees

可以设置决策树和bagging的参数


```python
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(
    n_estimators=100, # 基分类器的数量也就是决策树的数量
    max_samples=500, # 
    oob_score=True, # 放回取样
    n_jobs=1, #并行job树
    random_state=20,
    max_leaf_nodes=16
)    
```


```python
rf_clf.fit(x,y)
rf_clf.oob_score_
```




    0.852



# 提取特征的重要性feature_importances_


```python
rf_clf.feature_importances_
```




    array([0.45660686, 0.54339314])



上面输出的内容是两个特征分别的重要性


```python

```


```python

```


```python

```
