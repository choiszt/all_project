#include <linux/module.h>
#include<linux/time.h>
MODULE_LICENSE("GPL");
int timespec_add_ns_init(void)
{
    printk("timespec_add_ns begin.\n");
    // 声明变量，函数的第一个参数
    struct timespec ts={
        .tv_sec=1,
        .tv_nsec=1
    };
    u64 ns=1000000001;                     //64位无符号整数，函数的第二个参数
    printk("the value of the timespec before timespec_add_ns\n");
    printk("the tv_sec of the timespec is:%ld\n", ts.tv_sec); //显示参与加之前的数据
    printk("the tv_nsec of the timespec is:%ld\n", ts.tv_nsec);
    printk("the add ns is:%lld\n", ns);
    timespec_add_ns(&ts, ns);              //调用函数实现结构体变量与加整数的相加
    printk("the value of timespec after the timespec_add_ns :\n");
                                           //显示参与加之后的数据
    printk("the new tv_sec of the timespec is:%ld\n", ts.tv_sec);
    printk("the new tv_nsec of the timespec is:%ld\n", ts.tv_nsec);
    printk("timespec_add_ns over.\n");
    return 0;
}
void timespec_add_ns_exit(void)
{
    printk("Goodbye timespec_add_ns\n");
}
module_init(timespec_add_ns_init);
module_exit(timespec_add_ns_exit);