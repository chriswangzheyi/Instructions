# 房价数据预测及分析Demo


## 简单线性回归


### 代码

	import pandas as pd
	import statsmodels.api as sm
	from statsmodels.formula.api import ols  # Statsmodels是Python的统计建模和计量经济学工具包，包括一些描述统计、统计模型估计和推断
	import matplotlib.pyplot as plt
	import seaborn as sn
	
	# 读取数据
	housing_price_index = pd.read_csv('E:/data/house/monthly-hpi.csv')  #时间、房价指数
	unemployment = pd.read_csv('E:/data/house/unemployment-macro.csv')  #失业情况
	federal_funds_rate = pd.read_csv('E:/data/house/fed_funds.csv')   # 联邦政府贷款利率
	shiller= pd.read_csv('E:/data/house/shiller.csv')    #其他指数，比如500强指数，消费指数
	gross_domestic_product = pd.read_csv('E:/data/house/gdp.csv')  # gdp
	
	# 合并数据
	df = shiller.merge(housing_price_index, on='date')\
	    .merge( unemployment, on='date')\
	    .merge( federal_funds_rate, on='date')\
	    .merge(gross_domestic_product, on='date')
	
	
	# 简单线性回归
	housing_model = ols('housing_price_index ~ total_unemployed', data=df).fit() #Ordinary Least-Squares (OLS) Regression）
	
	print(housing_model.summary())  #其中 R-squared表示相关性


输出：

	                             OLS Regression Results                            
	===============================================================================
	Dep. Variable:     housing_price_index   R-squared:                       0.952
	Model:                             OLS   Adj. R-squared:                  0.949
	Method:                  Least Squares   F-statistic:                     413.2
	Date:                 Wed, 26 Feb 2020   Prob (F-statistic):           2.71e-15
	Time:                         18:18:00   Log-Likelihood:                -65.450
	No. Observations:                   23   AIC:                             134.9
	Df Residuals:                       21   BIC:                             137.2
	Df Model:                            1                                         
	Covariance Type:             nonrobust                                         
	====================================================================================
	                       coef    std err          t      P>|t|      [0.025      0.975]
	------------------------------------------------------------------------------------
	Intercept          313.3128      5.408     57.938      0.000     302.067     324.559
	total_unemployed    -8.3324      0.410    -20.327      0.000      -9.185      -7.480
	==============================================================================
	Omnibus:                        0.492   Durbin-Watson:                   1.126
	Prob(Omnibus):                  0.782   Jarque-Bera (JB):                0.552
	Skew:                           0.294   Prob(JB):                        0.759
	Kurtosis:                       2.521   Cond. No.                         78.9
	==============================================================================


![](../Images/3.png)


### 代码分析

#### import statsmodels.api as sm

statsmodels是一个Python模块,它提供对许多不同统计模型估计的类和函数


#### housing_model = ols('housing_price_index ~ total_unemployed', data=df).fit()

OLS： 最小二乘法。给定序列X(x1,x2...xn),y,估计一个向量A(a0,a1.a2....)令y'=a0+a1*x1+a2*x2+...+an*xn, 使得(y'-y)^2最小，计算A。 


#### print(housing_model.summary())

	 OLS Regression Results                            
	===============================================================================
	Dep. Variable:     housing_price_index   R-squared:                       0.952
	Model:                             OLS   Adj. R-squared:                  0.949
	Method:                  Least Squares   F-statistic:                     413.2
	Date:                 Thu, 27 Feb 2020   Prob (F-statistic):           2.71e-15
	Time:                         10:24:32   Log-Likelihood:                -65.450
	No. Observations:                   23   AIC:                             134.9
	Df Residuals:                       21   BIC:                             137.2
	Df Model:                            1                                         
	Covariance Type:             nonrobust                                         
	====================================================================================
	                       coef    std err          t      P>|t|      [0.025      0.975]
	------------------------------------------------------------------------------------
	Intercept          313.3128      5.408     57.938      0.000     302.067     324.559
	total_unemployed    -8.3324      0.410    -20.327      0.000      -9.185      -7.480
	==============================================================================
	Omnibus:                        0.492   Durbin-Watson:                   1.126
	Prob(Omnibus):                  0.782   Jarque-Bera (JB):                0.552
	Skew:                           0.294   Prob(JB):                        0.759
	Kurtosis:                       2.521   Cond. No.                         78.9
	==============================================================================


##### R-squared	
The coefficient of determination. A statistical measure of how well the regression line approximates the real data points

可决系数，说明估计的准确性

“可决系数”是通过数据的变化来表征一个拟合的好坏。由上面的表达式可以知道“确定系数”的正常取值范围为[0 1]，越接近1，表明方程的变量对y的解释能力越强，这个模型对数据拟合的也较好




##### adj. R-squared

The above value adjusted based on the number of observations and the degrees-of-freedom of the residuals


-----


## 多参数线性回归


### 代码

	import pandas as pd
	import statsmodels.api as sm
	from statsmodels.formula.api import ols  # Statsmodels是Python的统计建模和计量经济学工具包，包括一些描述统计、统计模型估计和推断
	import matplotlib.pyplot as plt
	import seaborn as sn
	
	# 读取数据
	housing_price_index = pd.read_csv('E:/data/house/monthly-hpi.csv')  #时间、房价指数
	unemployment = pd.read_csv('E:/data/house/unemployment-macro.csv')  #失业情况
	federal_funds_rate = pd.read_csv('E:/data/house/fed_funds.csv')   # 联邦政府贷款利率
	shiller= pd.read_csv('E:/data/house/shiller.csv')    #其他指数，比如500强指数，消费指数
	gross_domestic_product = pd.read_csv('E:/data/house/gdp.csv')  # gdp
	
	# 合并数据
	df = shiller.merge(housing_price_index, on='date')\
	    .merge( unemployment, on='date')\
	    .merge( federal_funds_rate, on='date')\
	    .merge(gross_domestic_product, on='date')
	
	
	# 简单线性回归
	housing_model = ols("""housing_price_index ~ total_unemployed 
	                    + long_interest_rate
	                    + federal_funds_rate
	                    + consumer_price_index
	                    + gross_domestic_product""", data=df).fit() #Ordinary Least-Squares (OLS) Regression）
	
	print(housing_model.summary())  #其中 R-squared表示相关性
	
	fig = plt.figure(figsize=(20,12))
	fig = sm.graphics.plot_partregress_grid(housing_model, fig=fig) # plot_partregress_grid 针对一个回归模型绘制回归结果,多图
	
	plt.show()


![](../Images/4.png)