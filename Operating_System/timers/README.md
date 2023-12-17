# OS_project

#### 文件说明

jiffies.c 将cpu的节拍数取整为变成HZ（250）的整数倍  
jiffies_up.c 将cpu的节拍数向上取整为变成HZ（250）的整数倍  
mod_timer.c 用于更改动态定时器的到期时间，从而可更改定时器的执行顺序  
jiffies_relative.c cpu相对节拍显示  
jiffies_relative_up.c cpu相对节拍向上取整  
setup_timer.c 定时器  
onstack.c  动态定时器  
time_addns.c timespec结构体变量与整数的相加，无符号整数表示的是纳秒数，结果保存在结构体变量中。  
time_equal.c 判断两个timespec类型的变量表示的时间是否相同  
timecompare.c比较两个timespec类型的变量所表示的时间的大小。  
task_pid.c 获取任务的任务描述符信息，此任务在进程pid的使用链表中，并且搜索的链表的起始元素的下标为参数type的值。  
writeproc.c 将待显示变量写入proc动态日志文件  
fortest.c 测试writeproc的结果(目前有bug，显示global变量问题）  
