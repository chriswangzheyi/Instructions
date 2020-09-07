# Spark 端口设置



Master节点的web端口是8080 仅在 standalone模式使
配置方式在spark-env.sh加一行

	export SPARK_MASTER_WEBUI_PORT=8080

work节点的web端口是8081
配置方式在spark-env.sh加一行

	export SPARK_WORKER_WEBUI_PORT=8081

Master通信端口是7077
配置方式在spark-env.sh加一行

	export SPARK_MASTER_PORT=7077

Spark历史服务器端口是18080
配置方式在spark-defaults.conf加一行

	spark.history.ui.port             18080

Spark外部服务端口是6066，这个端口有被黑客攻击的漏洞建议关闭
关闭方式在spark-defaults.conf加一行

	spark.master.rest.enabled         false

修改方式

	spark.master.rest.port               16066

Spark当前执行的任务页面查看端口4040
修改配置方式在spark-defaults.conf加一行
	
	spark.ui.port 14040