#include <linux/module.h>
#include <linux/timer.h>

MODULE_LICENSE("GPL");

struct timer_list my_timer;  //声明定时器全局变量

void my_timer_function(struct timer_list *t)
{
    printk("In the my_timer_function\n");
    printk("the jiffies is :%ld\n", jiffies);     //显示当前的节拍数
    printk("the expries of my_timer is :%lu\n", t->expires); // 显示字段expires的值
}

int __init setup_timer_init(void)
{
    printk("my_timer will be created.\n");
    printk("the jiffies is :%ld\n", jiffies);      //显示当前的节拍数
    my_timer.expires = jiffies + 1 * HZ;            //HZ=250，初始化字段expires的值

    // 初始化定时器变量的function和data字段
    timer_setup(&my_timer, my_timer_function, 0);
    add_timer(&my_timer);                          //将定时器变量加入到合适的链表，激活定时器
    printk("my_timer init.\n");
    return 0;
}

void __exit setup_timer_exit(void)
{
    printk("Goodbye setup_timer\n");
    del_timer_sync(&my_timer);  //删除定时器变量
}

module_init(setup_timer_init);
module_exit(setup_timer_exit);