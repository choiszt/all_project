#include <linux/init.h>
#include <linux/module.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>
#include <linux/sched.h>
#include "../writeproc.h"

MODULE_LICENSE("GPL");

// 测试 log_write 函数是否能够成功写入日志信息
static int __init test_log_write(void)
{
    log_buffer[0] = '\0';
    log_buffer_pos = 0;

    log_write("This is a test message.\n");
    printk(KERN_INFO "log_buffer: %s", log_buffer);

    return 0;
}

static void __exit test_log_write_exit(void)
{
    printk(KERN_INFO "test_log_write module unloaded\n");
}

module_init(test_log_write);
module_exit(test_log_write_exit);