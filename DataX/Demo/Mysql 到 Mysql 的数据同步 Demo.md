# Mysql 到 Mysql 的数据同步 Demo

## 代码

	{
	    "job": {
	        "setting": {
	            "speed": {
	                 "channel": 3
	            },
	            "errorLimit": {
	                "record": 0,
	                "percentage": 0.02
	            }
	        },
	        "content": [
	            {
	                "reader": {
	                    "name": "mysqlreader",
	                    "parameter": {
	                        "username": "root",
	                        "password": "1qa2ws#ED",
	                        "column": [
	                            "city_code",
	                            "city_name"
	                        ],
	                        "splitPk": "city_code",
	                        "connection": [
	                            {
	                                "table": [
	                                    "cities"
	                                ],
	                                "jdbcUrl": [
	     "jdbc:mysql://localhost:3306/test"
	                                ]
	                            }
	                        ]
	                    }
	                },
	            "writer": {
	                "name": "mysqlwriter",
	                "parameter": {
	                    "writeMode": "insert",
	                    "username": "root",
	                    "password": "1qa2ws#ED",
	                    "column": [
	                            "city_code",
	                            "city_name"
	                    ],
	                    "connection": [{
	                        "jdbcUrl": "jdbc:mysql://localhost:3306/test",
	                        "table": [
	                            "cities2"
	                        ]
	                    }]
	                }
	            }
	        }]
	    }
	}
