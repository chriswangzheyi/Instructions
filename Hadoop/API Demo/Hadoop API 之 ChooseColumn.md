#暂时不可用

vi test2.txt

姓名 性别 id
张三 男 1
李四 女 2
王五 女 3



hadoop fs -mkdir /input4 


hadoop fs -put /root/test2.txt /input4


hadoop fs -ls /input4 



hadoop jar /root/choose_column-1.0-SNAPSHOT.jar com.wzy.ChoosecolumnDriver /input4 /output/1014/1