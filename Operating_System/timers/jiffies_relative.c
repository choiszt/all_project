#include <linux/module.h>
#include<linux/timer.h>
MODULE_LICENSE("GPL");
int __init __round_jiffies_relative_init(void)
{
    printk("into __round_jiffies_relative_init");
    unsigned long j=jiffies; //获取当前节拍数

    /*第一个参数0代表相对节拍数，相对于当前的节拍，第二个参数0代表CPU编号*/
    unsigned long __result1=__round_jiffies_relative(0,0);

    /*第一个参数0代表相对节拍数，相对于当前的节拍，第二个参数1代表CPU编号*/
    unsigned long __result2=__round_jiffies_relative(0,1);
    printk("the current jiffies is :%ld\n", j);  //显示当前节拍

    /*显示函数调用结果*/
    printk("the __result1 of the __round_jiffies_relative(0,0) is :%ld\n", __result1);
    printk("the __result2 of the __round_jiffies_relative(0,1) is :%ld\n", __result2);
    printk("out __round_jiffies_relative_init");
    return 0;
}

void __exit __round_jiffies_relative_exit(void)
{
    printk("Goodbye __round_jiffies_relative\n");
}
module_init(__round_jiffies_relative_init);
module_exit(__round_jiffies_relative_exit);