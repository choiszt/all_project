#include <linux/module.h>
#include<linux/timer.h>
MODULE_LICENSE("GPL");
struct timer_list my_timer1; //自定义动态定时器，全局变量
// 自定义动态定时器到期处理函数，此函数在此只有显示功能，不做任何处理
void my_timer1_function(unsigned long data)
{
    printk("In the my_timer1_function\n");
    struct timer_list *mytimer = (struct timer_list *)data;
    printk("the current jiffies is:%ld\n", jiffies);                 //显示当前节拍数

    // 显示动态定时器的到期节拍数
    printk("the expires of my_timer1 is:%ld\n", mytimer->expires);

    // 重新设定动态定时器到期节拍数
    int result1=mod_timer(&my_timer1, my_timer1.expires+10);
    printk("the mod result of my_timer1 is: %d\n", result1);         //显示函数调用结果

    // 显示动态定时器更新之后的到期节拍数
    printk("the new expires of my_timer1 is: %ld\n", my_timer1.expires);

    // 显示动态定时器的base字段
    printk("the new base of my_timer1 is: %u\n", (unsigned int) my_timer1.base);
    del_timer(&my_timer1);                                           //删除定时器变量
}
int __init mod_timer_init(void)
{
    printk("my_timer1 will be created.\n");
    printk("the current jiffies is :%ld\n", jiffies);       //显示当前节拍数
    init_timer(&my_timer1);                                 //初始化动态定时器
    my_timer1.expires = jiffies + 1*HZ;                    //初始化字段expires, HZ=250
    my_timer1.data = &my_timer1;                            //初始化字段data
    my_timer1.function = my_timer1_function;                //初始化字段function
    add_timer(&my_timer1);                                  //激活动态定时器

    // 显示字段expires的值
    printk("the expires of my_timer1 after function add_timer( ) is:%ld\n",my_timer1.expires);

    // 显示字段base的值
    printk("the base of my_timer1 after function add_timer( ) is:%u\n", (unsigned int)my_timer1.base);

    // 重新设定动态定时器到期节拍数
    int result1=mod_timer(&my_timer1, my_timer1.expires+10);
    printk("the mod result of my_timer1 is: %d\n", result1);    //显示函数调用结果

    // 显示动态定时器更新之后的到期节拍数
    printk("the new expires of my_timer1 is: %ld\n", my_timer1.expires);

    // 显示动态定时器的base字段
    printk("the new base of my_timer1 is: %u\n", (unsigned int)my_timer1.base);
    printk("my_timer1 init.\n");
    return 0;
}
void __exit mod_timer_exit(void)
{
        printk("Goodbye mod_timer\n");
        del_timer(&my_timer1); //删除定时器变量
}
module_init(mod_timer_init);
module_exit(mod_timer_exit);