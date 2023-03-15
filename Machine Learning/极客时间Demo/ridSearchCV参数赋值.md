# ridSearchCV参数赋值

## 参数
###estimator：
分类器或者叫学习器。代码中用的是，随机森林策略模型rf RandomForestClassifier 

###param_grid: 
参数列表，类型是字典。代码中的param_grid指定了特征选择的标准criterion，是用基尼系数gini，还是信息熵entropy；还指定了"min_samples_leaf" ，叶子节点最小样本数，min_samples_split等。n_estimators是用来防止过拟合的。（参数的选择，需要熟悉决策树。篇幅受限无法扩展）

param_grid = { "criterion" : ["gini", "entropy"], "min_samples_leaf" : [1, 5, 10], "min_samples_split" : [2, 4, 10, 12, 16], "n_estimators": [50, 100, 400, 700, 1000]}

###scoring：
 模型准确率评价标准。默认是None。代码示例中用的是accuracy，精确度评价标准

###n_jobs: 
并行数，-1 意思是同cpu核数一致。就是CPU全部占用的意思

###cv: 
交叉验证。代码中 cv =3. 三折交叉验证。均等三份样本集。第一次：三分之二做训练集，三分之一作为测试集。拟合后，得到一个泛化结果。第二次，用上一次三分之二里的训练集一分为二，一部分验证，一部分训练。看下图吧（举例是五折交叉验证，都是一个意思）五折交叉，先五等分，80%训练集，20%测试集。第二次，training训练集的等分部分拿出来做验证，剩余做训练。依次类推。总共五次交叉验证。

### verbose:
控制详细程度，数字越大越详细


## 代码例子

### 对随机森林算法进行参数优化

	rfr_param_grid = {'bootstrap': [True, False],
	                 'max_depth': [10, 50, 100, None],
	                 'max_features': ['auto', 'sqrt'],
	                 'min_samples_leaf': [1, 2, 4],
	                 'min_samples_split': [2, 5, 10],
	                 'n_estimators': [50, 500, 2000]}

	from sklearn.model_selection import GridSearchCV # 导入网格搜索工具
	model_rfr_gs = GridSearchCV(model_rfr,
	                            param_grid = rfr_param_grid, cv=3,
	                            scoring="r2", n_jobs= 10, verbose = 1)
	model_rfr_gs.fit(X_train, y_train) # 用优化后的参数拟合训练数据集
	

## 决策树参数

* criterion: 衡量拆分质量。支持的标准是基尼杂质的“基尼”和信息增益的“熵”。
* splitter：用于在每个节点处选择拆分的策略。支持的策略是“best”选择最佳拆分和“random”选择随机拆分。
* max_depth：树的最大深度。如果为None，则展开节点，直到所有叶节点，或者直到所有叶包含的样本小于min_samples_split。
* min_samples_split：拆分内部节点所需的最小样本数。
* min_samples_leaf:叶节点上所需的最小样本数。
* min_weight_fraction_leaf:叶节点上所需的总权重的最小加权分数。当没有提供sample_weight时，样本具有相等的权值。
* max_features：寻找最佳拆分时要考虑的特征数量。
* max_leaf_nodesmax_leaf_nodes：以最佳优先的方式使用max_leaf_nodes形成树。最佳节点定义为杂质的相对减少。如果为None，则有无限数量的叶节点。
* min_impurity_decrease：如果该拆分导致杂质减少大于或等于该值，则该节点将被拆分。
* min_impurity_split: 提前停止的阈值。如果一个节点的杂质高于阈值，则该节点将拆分，否则，它是一个叶子。
* bootstrap：数默认True，代表采用这种有放回的随机抽样技术（有放回抽样也会有自己的问题。由于是有放回，一些样本可能在同一个自助集中出现多次，而其他一些却可能被忽略，自助集大约平均会包含63%的原始数据。），通常，这个参数不会被我们设置为False
* n_estimators：这是森林中树木的数量，即基评估器的数量。n_estimators越大，模型的效果往往越好。n_estimators达到一定的程度之后，随机森林的精确性往往不在上升或开始波动，并且，n_estimators越大，需要的计算量和内存也越大，训练的时间也会越来越长。