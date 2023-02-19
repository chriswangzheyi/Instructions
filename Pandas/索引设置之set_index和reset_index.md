```python
#set_index方法
```


```python
import pandas as pd
df = pd.DataFrame({'a':range(7),
                   'b':range(7,0,-1),
                   'c':['one','one','one','two','two','two','two'],
                   'd':[0,1,2,0,1,2,3]})
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>7</td>
      <td>one</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>6</td>
      <td>one</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>5</td>
      <td>one</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>4</td>
      <td>two</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>3</td>
      <td>two</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>2</td>
      <td>two</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>1</td>
      <td>two</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 1.保留索引值
df.set_index(['c','d'], drop=False) 
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>7</td>
      <td>one</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>6</td>
      <td>one</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>5</td>
      <td>one</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>4</td>
      <td>two</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>3</td>
      <td>two</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>2</td>
      <td>two</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>1</td>
      <td>two</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 2.添加到原有索引
df.set_index('c', append=True)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>d</th>
    </tr>
    <tr>
      <th></th>
      <th>c</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <th>one</th>
      <td>0</td>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <th>one</th>
      <td>1</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <th>one</th>
      <td>2</td>
      <td>5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <th>two</th>
      <td>3</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <th>two</th>
      <td>4</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <th>two</th>
      <td>5</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <th>two</th>
      <td>6</td>
      <td>1</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 3.多重索引
df.set_index(['c','d'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>a</th>
      <th>b</th>
    </tr>
    <tr>
      <th>c</th>
      <th>d</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">one</th>
      <th>0</th>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>5</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">two</th>
      <th>0</th>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 4.修改原数据框
df.set_index(['c','d'], inplace=True)
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>a</th>
      <th>b</th>
    </tr>
    <tr>
      <th>c</th>
      <th>d</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">one</th>
      <th>0</th>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>5</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">two</th>
      <th>0</th>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# reset_index方法
```


```python
import pandas as pd
import numpy as np
df = pd.DataFrame({'Country':['China','China', 'India', 'India', 'America', 'Japan', 'China', 'India'], 
                   'Income':[10000, 10000, 5000, 5002, 40000, 50000, 8000, 5000], 
                    'Age':[50, 43, 34, 40, 25, 25, 45, 32]})

df_new = df.set_index('Country', drop=True, append=False, inplace=False)
```


```python
df_new
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Income</th>
      <th>Age</th>
    </tr>
    <tr>
      <th>Country</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>China</th>
      <td>10000</td>
      <td>50</td>
    </tr>
    <tr>
      <th>China</th>
      <td>10000</td>
      <td>43</td>
    </tr>
    <tr>
      <th>India</th>
      <td>5000</td>
      <td>34</td>
    </tr>
    <tr>
      <th>India</th>
      <td>5002</td>
      <td>40</td>
    </tr>
    <tr>
      <th>America</th>
      <td>40000</td>
      <td>25</td>
    </tr>
    <tr>
      <th>Japan</th>
      <td>50000</td>
      <td>25</td>
    </tr>
    <tr>
      <th>China</th>
      <td>8000</td>
      <td>45</td>
    </tr>
    <tr>
      <th>India</th>
      <td>5000</td>
      <td>32</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 索引的列被还原
df_new.reset_index() # drop=False
df_new.reset_index(drop=True) # 列被删除
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Income</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10000</td>
      <td>50</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10000</td>
      <td>43</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5000</td>
      <td>34</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5002</td>
      <td>40</td>
    </tr>
    <tr>
      <th>4</th>
      <td>40000</td>
      <td>25</td>
    </tr>
    <tr>
      <th>5</th>
      <td>50000</td>
      <td>25</td>
    </tr>
    <tr>
      <th>6</th>
      <td>8000</td>
      <td>45</td>
    </tr>
    <tr>
      <th>7</th>
      <td>5000</td>
      <td>32</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 原始数据框操作
df.reset_index(drop=True)
df.reset_index()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>Country</th>
      <th>Income</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>China</td>
      <td>10000</td>
      <td>50</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>China</td>
      <td>10000</td>
      <td>43</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>India</td>
      <td>5000</td>
      <td>34</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>India</td>
      <td>5002</td>
      <td>40</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>America</td>
      <td>40000</td>
      <td>25</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>Japan</td>
      <td>50000</td>
      <td>25</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>China</td>
      <td>8000</td>
      <td>45</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>India</td>
      <td>5000</td>
      <td>32</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = pd.DataFrame(columns=['a', 'b'])
print(df)
print("---------")
b = pd.Series([1,1,1,1,1],index=[0,1,3,4,5])
a = pd.Series([2,2,2,2,2,2],index=[1,2,3,4,6,7])
df['a'] = a
df['b'] = b
print(df)
```

    Empty DataFrame
    Columns: [a, b]
    Index: []
    ---------
       a    b
    1  2  1.0
    2  2  NaN
    3  2  1.0
    4  2  1.0
    6  2  NaN
    7  2  NaN



```python

```
