#include <linux/module.h>
#include <linux/time.h>

MODULE_LICENSE("GPL");

int set_timespec_init(void)
{
    struct timespec ts = {
        .tv_sec = 0,
        .tv_nsec = 0
    };
    printk("set_timespec begin.\n");
    printk("the value of struct timespec before the set_timespec:\n");
    printk("the tv_sec value is:%ld\n", ts.tv_sec);
    printk("the tv_nsec value is:%ld\n", ts.tv_nsec);

    ts = ktime_to_timespec(ktime_set(1, 1000000010L));

    printk("the value of struct timespec after the set_timespec:\n");
    printk("the tv_sec value is:%ld\n", ts.tv_sec);
    printk("the tv_nsec value is:%ld\n", ts.tv_nsec);
    printk("set_timespec over.\n");
    return 0;
}

void set_timespec_exit(void)
{
    printk("Goodbye set_timespec\n");
}

module_init(set_timespec_init);
module_exit(set_timespec_exit);