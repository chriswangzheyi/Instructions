# Cursor 集成 Git



## 步骤

点击左侧的双点=>远程=》添加远程存储库：



![](Images/1.png)



以gitee为例



```
Git remote set-url 仓库名 Https://oauth2:{私钥}@git路径
```

例如：

```
git remote set-url training https://oauth2:XXXXX@gitee.com/chongqing-jx/training.git
```

XXX为私钥



## 回滚



找到提交的id



![](Images/2.png)



在terminal中输入



```
git revert 提交id
```



然后输入 :wq! 保存修改内容