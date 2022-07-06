# Flink的4种重启策略


* Fixed Delay Restart Strategy
* Failure Rate Restart Strategy
* No Restart Strategy
* Fallback Restart Strategy


##  固定间隔 (Fixed delay)

For example:

	restart-strategy.fixed-delay.attempts: 3
	restart-strategy.fixed-delay.delay: 10 s
	
失败后，重启3次（每次重启间隔10s），如果第3次还是失败，则任务最终是失败，不再重启。

## 失败率 (Failure rate)

For example:

	restart-strategy.failure-rate.max-failures-per-interval: 3
	restart-strategy.failure-rate.failure-rate-interval: 5 min
	restart-strategy.failure-rate.delay: 10 s
失败后，5分钟内重启3次（每次重启间隔10s），如果第3次还是失败，则任务最终是失败，不再重启。


## 无重启 (No restart)

第一次失败后就最终失败，不再重启

## Fallback(备用重启策略)

使用群集定义的重新启动策略。这对于启用检查点的流式传输程序很有帮助。默认情况下，如果没有定义其他重启策略，则选择固定延迟重启策略。
