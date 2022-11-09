#!/bin/bash


# $1 生成数据的种类：user | log
# 如果生成user
# $2 如果生成user，则是user数据的用户量
# $3 如果生成user，则是user数据的输出路径（到文件名）

# 如果生成log
# $2， 在线人数
# $3,  要生成的日期
# $4,  1:流式生成  0:批量生成
# $5,  user数据的路径
# $6,  日志数据的输出路径（到目录）
# $7,  是否日志模式（每行flush）

export JAVA_HOME=/opt/apps/jdk1.8.0_191/

if [ $1 = user ];then
echo "generating user data ..."
$JAVA_HOME/bin/java -cp datagen.jar cn.doitedu.datagen.beans.GenUsers $2 $3  &
fi



if [ $1 = log ];then
echo "generating log data ..."
$JAVA_HOME/bin/java -jar datagen.jar ${2} ${3}  ${4} ${5} ${6} ${7}  &
fi
