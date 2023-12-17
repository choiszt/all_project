#include <linux/module.h>
#include<linux/timer.h>
MODULE_LICENSE("GPL");
int round_jiffies_up_relative_init(void)
{
    printk("into round_jiffies_up_relative\n");
    unsigned long j=jiffies; //当前节拍数

    /*第一个参数0代表相对节拍数，相对于当前的节拍，第二个参数0代表CPU编号*/
    unsigned long __result1=__round_jiffies_up_relative(0,0);

    /*第一个参数0代表相对节拍数，相对于当前的节拍，第二个参数1代表CPU编号*/
    unsigned long __result2=__round_jiffies_up_relative(0,1);
    unsigned long result1=round_jiffies_up_relative(0);
                                          // 参数0代表相对节拍数，相对于当前的节拍
    unsigned long result2=round_jiffies_up_relative(0);
                                          // 参数0代表相对节拍数，相对于当前的节拍
    printk("the current jiffies is :%ld\n", j);                 //显示当前的节拍数

    /*显示函数调用结果*/
    printk("the __result1 of the __round_jiffies_up_relative(0,0) is :%ld\n", __result1);
    printk("the __result2 of the __round_jiffies_up_relative(0,1) is :%ld\n", __result2);
    printk("the result1 of the round_jiffies_up_relative(0) is :%ld\n", result1);
    printk("the result2 of the round_jiffies_up_relative(0) is :%ld\n", result2);
    printk("out round_jiffies_up_relative\n");
    return 0;
}
void round_jiffies_up_relative_exit(void)
{
    printk("Goodbye round_jiffies_up_relative\n");
}

module_init(round_jiffies_up_relative_init);
module_exit(round_jiffies_up_relative_exit);