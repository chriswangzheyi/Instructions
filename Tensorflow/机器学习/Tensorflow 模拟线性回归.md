# Tensorflow 模拟线性回归


## 代码

	import tensorflow as tf
	
	# tf.random_normal()函数用于从“服从指定正态分布的序列”中随机取出指定个数的值
	random_normal = tf.initializers.RandomNormal()
	
	# 100样本,1个特征
	X = random_normal(shape=[100,1])
	# 假设目标值与真实值的关系: x*0.7 + 0.8
	y_true = tf.matmul(X,[[0.7]]) + 0.8
	# 声明权重与偏置
	w = tf.Variable(random_normal(shape=[1,1]))
	b = tf.Variable(0.0)  # 一个数,不需要添加[]
	
	for i in range(100):
	    with tf.GradientTape() as g: # 梯度下降算法
	        y_predict = tf.matmul(X,w) + b
	        loss = tf.reduce_mean(tf.square(y_true - y_predict)) #reduce_mean计算张量沿着指定的数轴（tensor的某一维度）上的的平均值
	        # 本质就是通过求导,来找到最小值
	        gradients = g.gradient(loss,[w,b])
	        lr = 0.01  # 学习率
	        w.assign_sub(lr * gradients[0])
	        b.assign_sub(lr * gradients[1])
	        print(loss)
	
	print("w=",w.value(),", b=",b.value())
	

## 显示

	tf.Tensor(0.64299715, shape=(), dtype=float32)
	tf.Tensor(0.61757964, shape=(), dtype=float32)
	tf.Tensor(0.5931686, shape=(), dtype=float32)
	tf.Tensor(0.56972444, shape=(), dtype=float32)
	tf.Tensor(0.5472085, shape=(), dtype=float32)
	tf.Tensor(0.52558416, shape=(), dtype=float32)
	tf.Tensor(0.50481623, shape=(), dtype=float32)
	tf.Tensor(0.48487073, shape=(), dtype=float32)
	tf.Tensor(0.46571496, shape=(), dtype=float32)
	tf.Tensor(0.4473179, shape=(), dtype=float32)
	tf.Tensor(0.4296492, shape=(), dtype=float32)
	tf.Tensor(0.41268027, shape=(), dtype=float32)
	tf.Tensor(0.3963833, shape=(), dtype=float32)
	tf.Tensor(0.3807317, shape=(), dtype=float32)
	tf.Tensor(0.36569986, shape=(), dtype=float32)
	tf.Tensor(0.3512633, shape=(), dtype=float32)
	tf.Tensor(0.3373984, shape=(), dtype=float32)
	tf.Tensor(0.32408255, shape=(), dtype=float32)
	tf.Tensor(0.31129405, shape=(), dtype=float32)
	tf.Tensor(0.29901195, shape=(), dtype=float32)
	tf.Tensor(0.2872162, shape=(), dtype=float32)
	tf.Tensor(0.27588758, shape=(), dtype=float32)
	tf.Tensor(0.26500756, shape=(), dtype=float32)
	tf.Tensor(0.25455838, shape=(), dtype=float32)
	tf.Tensor(0.24452297, shape=(), dtype=float32)
	tf.Tensor(0.23488499, shape=(), dtype=float32)
	tf.Tensor(0.22562866, shape=(), dtype=float32)
	tf.Tensor(0.21673888, shape=(), dtype=float32)
	tf.Tensor(0.20820111, shape=(), dtype=float32)
	tf.Tensor(0.20000145, shape=(), dtype=float32)
	tf.Tensor(0.19212648, shape=(), dtype=float32)
	tf.Tensor(0.18456337, shape=(), dtype=float32)
	tf.Tensor(0.17729975, shape=(), dtype=float32)
	tf.Tensor(0.17032377, shape=(), dtype=float32)
	tf.Tensor(0.163624, shape=(), dtype=float32)
	tf.Tensor(0.1571896, shape=(), dtype=float32)
	tf.Tensor(0.15100996, shape=(), dtype=float32)
	tf.Tensor(0.14507504, shape=(), dtype=float32)
	tf.Tensor(0.13937514, shape=(), dtype=float32)
	tf.Tensor(0.13390096, shape=(), dtype=float32)
	tf.Tensor(0.12864353, shape=(), dtype=float32)
	tf.Tensor(0.12359429, shape=(), dtype=float32)
	tf.Tensor(0.118745, shape=(), dtype=float32)
	tf.Tensor(0.114087775, shape=(), dtype=float32)
	tf.Tensor(0.10961492, shape=(), dtype=float32)
	tf.Tensor(0.105319194, shape=(), dtype=float32)
	tf.Tensor(0.1011936, shape=(), dtype=float32)
	tf.Tensor(0.097231366, shape=(), dtype=float32)
	tf.Tensor(0.09342602, shape=(), dtype=float32)
	tf.Tensor(0.089771345, shape=(), dtype=float32)
	tf.Tensor(0.08626141, shape=(), dtype=float32)
	tf.Tensor(0.08289047, shape=(), dtype=float32)
	tf.Tensor(0.07965302, shape=(), dtype=float32)
	tf.Tensor(0.07654376, shape=(), dtype=float32)
	tf.Tensor(0.07355762, shape=(), dtype=float32)
	tf.Tensor(0.07068974, shape=(), dtype=float32)
	tf.Tensor(0.067935415, shape=(), dtype=float32)
	tf.Tensor(0.06529017, shape=(), dtype=float32)
	tf.Tensor(0.06274965, shape=(), dtype=float32)
	tf.Tensor(0.060309734, shape=(), dtype=float32)
	tf.Tensor(0.057966452, shape=(), dtype=float32)
	tf.Tensor(0.055715937, shape=(), dtype=float32)
	tf.Tensor(0.053554554, shape=(), dtype=float32)
	tf.Tensor(0.05147876, shape=(), dtype=float32)
	tf.Tensor(0.04948515, shape=(), dtype=float32)
	tf.Tensor(0.047570486, shape=(), dtype=float32)
	tf.Tensor(0.04573163, shape=(), dtype=float32)
	tf.Tensor(0.04396558, shape=(), dtype=float32)
	tf.Tensor(0.04226949, shape=(), dtype=float32)
	tf.Tensor(0.040640537, shape=(), dtype=float32)
	tf.Tensor(0.039076082, shape=(), dtype=float32)
	tf.Tensor(0.03757358, shape=(), dtype=float32)
	tf.Tensor(0.036130577, shape=(), dtype=float32)
	tf.Tensor(0.034744702, shape=(), dtype=float32)
	tf.Tensor(0.033413723, shape=(), dtype=float32)
	tf.Tensor(0.03213544, shape=(), dtype=float32)
	tf.Tensor(0.030907774, shape=(), dtype=float32)
	tf.Tensor(0.02972871, shape=(), dtype=float32)
	tf.Tensor(0.028596334, shape=(), dtype=float32)
	tf.Tensor(0.027508805, shape=(), dtype=float32)
	tf.Tensor(0.026464336, shape=(), dtype=float32)
	tf.Tensor(0.025461221, shape=(), dtype=float32)
	tf.Tensor(0.02449783, shape=(), dtype=float32)
	tf.Tensor(0.023572583, shape=(), dtype=float32)
	tf.Tensor(0.022683963, shape=(), dtype=float32)
	tf.Tensor(0.02183054, shape=(), dtype=float32)
	tf.Tensor(0.021010904, shape=(), dtype=float32)
	tf.Tensor(0.02022372, shape=(), dtype=float32)
	tf.Tensor(0.019467697, shape=(), dtype=float32)
	tf.Tensor(0.01874161, shape=(), dtype=float32)
	tf.Tensor(0.01804427, shape=(), dtype=float32)
	tf.Tensor(0.017374538, shape=(), dtype=float32)
	tf.Tensor(0.016731324, shape=(), dtype=float32)
	tf.Tensor(0.016113576, shape=(), dtype=float32)
	tf.Tensor(0.015520286, shape=(), dtype=float32)
	tf.Tensor(0.014950492, shape=(), dtype=float32)
	tf.Tensor(0.014403262, shape=(), dtype=float32)
	tf.Tensor(0.013877685, shape=(), dtype=float32)
	tf.Tensor(0.013372912, shape=(), dtype=float32)
	tf.Tensor(0.0128881335, shape=(), dtype=float32)
	w= tf.Tensor([[0.04840675]], shape=(1, 1), dtype=float32) , b= tf.Tensor(0.6948996, shape=(), dtype=float32)