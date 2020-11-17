# opentsDB语法规则



## 写数据

使用post：

	http://47.112.142.231:4242/api/put?details

请求内容：

	#请求体

	{
	"metric": "sys.cpu.nice",
	"timestamp": 1346846402,
	"value": 18,
	"tags": {
		"host": "web01",
	    "dc": "lga"
	        }
	}

	#写入成功返回的内容
	{
	    "success": 1,
	    "failed": 0,
	    "errors": []
	}


## 查数据

post 请求：

	http://47.112.142.231:4242/api/query

请求体:

	{
	    "start": 1346846402,
	    "end": 1346846403,
	    "showTSUIDs":"true",  
	    "queries": [
	        {
	            "aggregator": "avg",
	            "metric": "sys.cpu.nice",
	            "tags": {
	                 "host": "web01",
	                 "dc": "lga"
	             }
	        }
	    ]
	}

返回：

	[
	    {
	        "metric": "sys.cpu.nice",
	        "tags": {
	            "host": "web01",
	            "dc": "lga"
	        },
	        "aggregateTags": [],
	        "tsuids": [
	            "000002000001000001000002000003"
	        ],
	        "dps": {
	            "1346846402": 18
	        }
	    }
	]



## 查询所有的metrics

	http://47.112.142.231:4242/api/suggest?max=1000&q=&type=metrics

返回：

	["sys.cpu.data"]

## 查询全量tag

 	http://47.112.142.231:4242/api/suggest?max=1000&q=&type=tagk

返回：

	["host"]

## 查询全量tagv

	http://47.112.142.231:4242/api/suggest?max=1000&q=&type=tagv

返回：

	["web01","web02"]