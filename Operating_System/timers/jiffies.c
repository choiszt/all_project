#include <linux/module.h>
#include<linux/timer.h>
MODULE_LICENSE("GPL");
int __init __round_jiffies_init(void)
{
    printk("the __round_jiffies test begin\n");
    unsigned long j=jiffies; //记录当前节拍
    unsigned long __result1=__round_jiffies(j,0); //参数j代表当前节拍数，0是CPU编号
    unsigned long __result2=__round_jiffies(j,1); //参数j代表当前节拍数，1是CPU编号
    printk("the jiffies is :%ld\n", j);             //显示当前节拍
    // 显示函数调用结果
    printk("the __result1 of __round_jiffies(j,0) is :%ld\n", __result1);
    printk("the __result2 of __round_jiffies(j,1) is :%ld\n", __result2);
    printk("out __round_jiffies_init");
    return 0;
}
void __exit __round_jiffies_exit(void)
{
    printk("Goodbye __round_jiffies\n");
}
module_init(__round_jiffies_init);
module_exit(__round_jiffies_exit);