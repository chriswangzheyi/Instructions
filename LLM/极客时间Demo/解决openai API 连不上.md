# 解决openai API 连不上

## urllib3

	pip uninstall urllib3
	pip install urllib3==1.25.11


## 代理

	proxy = {
	'http': 'http://127.0.0.1:10887',
	'https': 'http://127.0.0.1:10887'
	}
	
	openai.proxy = proxy;