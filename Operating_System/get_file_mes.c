#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/fs.h>
#include <linux/delay.h>
#include <linux/sched.h>
#include <linux/mm.h>
#include <linux/time.h>
#include <linux/types.h>
#include <linux/kernel_stat.h>
#include <linux/sched/signal.h>
#include <linux/mm.h>
#include <linux/slab.h>
#include <linux/sched/signal.h>
#include <linux/proc_fs.h>
#include <linux/module.h>
#include <linux/init.h>
#include <linux/sched.h>
#include <linux/fdtable.h>
#include <linux/net.h>
#include <linux/inet.h>
#include <linux/inet_diag.h>
#include <linux/fs.h>
#include <linux/file.h>

static int get_file_mes(int pid)
{
    struct task_struct *task = pid_task(find_vpid(pid), PIDTYPE_PID);
    struct files_struct *files = task->files;
    struct fdtable *fdt = files_fdtable(files);
    unsigned long fd;

    printk(KERN_INFO "Searching Process File Table\n");

    //spin_lock(&files->file_lock); 
    pr_info("total open files: %d\n", fdt->max_fds);
    for (fd = 0; fd < fdt->max_fds ; fd++) {
        if (test_bit(fd, fdt->open_fds)) {
            struct file *file = fdt->fd[fd];
            struct inode *inode = file->f_inode;

            int mode = inode->i_mode;

            struct tm tm_time;
            ktime_t timestamp = inode->i_atime.tv_sec;
            time64_to_tm(timestamp, 0, &tm_time);

            char strtime[20];
            snprintf(strtime, sizeof(strtime), "%04d-%02d-%02d %02d:%02d:%02d",
                    tm_time.tm_year + 1900, tm_time.tm_mon + 1, tm_time.tm_mday,
                    tm_time.tm_hour, tm_time.tm_min, tm_time.tm_sec);

            //strftime(strtime, 20, "%Y-%m-%d %H:%M:%S", &timeinfo);
            printk("\n");

            printk(KERN_INFO "Current file name: %s\n", file->f_path.dentry->d_name.name);

            printk(KERN_INFO "File access mode: %o\n", mode & 0777);

            printk(KERN_INFO "Last accessed time: %s", strtime);
        }
    }
    return 0;
}