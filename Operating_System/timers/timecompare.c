#include <linux/module.h>
#include<linux/time.h>
MODULE_LICENSE("GPL");
int __init timespec_compare_init(void)
{
    int result_equ, result_com;    //用于保存函数timespec_compare( )的返回结果

    /*定义两个timespec类型的变量，作为函数的参数*/
    struct timespec lhs=
    {
        .tv_sec=10,
        .tv_nsec=110
    };
    struct timespec rhs=
    {
        .tv_sec=9,
        .tv_nsec=100
    };
    printk("timespec_compare begin.\n");
    result_com=timespec_compare(&lhs, &rhs);      //比较时间的大小
    printk("the timespec compare result is: %d\n", result_com); //显示时间比较大小的结果
    printk("timespec_compare over\n");
    return 0;
}
void __exit timespec_compare_exit(void)
{
    printk("Goodbye timespec_compare\n");
}
module_init(timespec_compare_init);
module_exit(timespec_compare_exit);