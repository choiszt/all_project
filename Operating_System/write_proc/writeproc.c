#include <linux/init.h>       // 内核初始化函数所需的头文件
#include <linux/module.h>     // 内核模块所需的头文件
#include <linux/proc_fs.h>    // proc 文件系统所
#include <linux/seq_file.h>   // seq_file API 
#include <linux/sched.h>      // 进程调度
MODULE_LICENSE("GPL");         // 模块许可证
#define PROC_FILENAME "kernellog"  // 定义 proc 文件的名称

static struct proc_dir_entry *proc_file;  // 定义 proc 文件的指针

// 缓冲区，用于记录 log 信息
// static char log_buffer[1024];
// static int log_buffer_pos;
// modified by Wang Xuan
// 不能用static，否则外部无法访问
char log_buffer[1024*1024]={0};
int log_buffer_pos=0;
EXPORT_SYMBOL(log_buffer);   // 导出 log_buffer
EXPORT_SYMBOL(log_buffer_pos);   // 导出 log_buffer_pos

// 将 log 信息写入缓冲区
void log_write(const char *fmt, ...)
{
    va_list args;
    int len;

    va_start(args, fmt);
    len = vsnprintf(log_buffer + log_buffer_pos, sizeof(log_buffer) - log_buffer_pos, fmt, args);
    va_end(args);

    if (len > 0) {
        log_buffer_pos += len;
    }
}

EXPORT_SYMBOL(log_write);   // 导出 log_write 函数

// 将缓冲区中的 log 信息写入 /proc/kernellog 文件中
static int log_show(struct seq_file *m, void *v)
{
    seq_printf(m, "%s", log_buffer);
    return 0;
}

// 打开 /proc/kernellog 文件
static int log_open(struct inode *inode, struct file *file)
{
    return single_open(file, log_show, NULL);
}

// 定义文件操作回调函数
static const struct proc_ops log_fops = {
    // .owner = THIS_MODULE,
    .proc_open = log_open,
    .proc_read = seq_read,
    .proc_lseek = seq_lseek,
    .proc_release = single_release,
};

// 模块初始化函数
static int __init kernellog_init(void)
{
    log_buffer[0] = '\0';   // 将缓冲区清空
    log_buffer_pos = 0;

    // 在 /proc 文件系统中创建一个新的文件 /proc/kernellog，并将 log_fops 结构体作为文件操作的回调函数
    proc_file = proc_create(PROC_FILENAME, 0, NULL, &log_fops);
    if (proc_file == NULL) {
        return -ENOMEM;   // 创建文件失败，返回错误码
    }

    printk(KERN_INFO "kernellog module loaded\n");   // 打印模块加载信息
    return 0;
}

// 模块退出函数
static void __exit kernellog_exit(void)
{
    // 从 /proc 文件系统中删除 /proc/kernellog 文件
    proc_remove(proc_file);

    printk(KERN_INFO "kernellog module unloaded\n");   // 打印模块卸载信息
}


module_init(kernellog_init);   // 注册模块初始化函数
module_exit(kernellog_exit);   // 注册模块退出函数

MODULE_DESCRIPTION("Kernel log module written by Choiszt");
