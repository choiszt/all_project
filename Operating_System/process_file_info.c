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
#include "write_proc/writeproc.h"
extern struct pid_namespace *ins_ns;
extern const char* get_access_mode_string(fmode_t mode);
static bool is_regular_file(struct file *file)
{
    /* Check if the file is a regular file
     * If it is, return true
     * If it is not, return false
     * added by X.Wang 2023-4-2
     * 
     * @file: the file to be checked    
     */
    struct inode *inode = file->f_inode;
    umode_t mode = inode->i_mode & S_IFMT;
    if (mode == S_IFREG)
    {
        return true;
    }
    return false;    
}

int processFileInfo(int pid)
{
    // struct task_struct *task = pid_task(find_vpid(pid), PIDTYPE_PID);
    struct task_struct *task = pid_task(find_pid_ns(pid, ins_ns), PIDTYPE_PID);
    /* modified by X.Wang 2023-4-1
     * success get task_struct
     * modified by X.Wang 2023-3-26
     * add exception handling
     */
    if (task == NULL)
    {
        pr_info("No such process, pid: %d\n", pid);
        return -1;
    }
    struct files_struct *files = task->files;
    struct fdtable *fdt = files_fdtable(files);
    unsigned long fd;

    pr_debug("Searching Process File Table\n");

    spin_lock(&files->file_lock);
    for (fd = 0; fd < fdt->max_fds; fd++) {
        if (test_bit(fd, fdt->open_fds)) {
            struct file *file = fdt->fd[fd];
            if (file && is_regular_file(file)) {

                struct inode *inode = file->f_inode;

                // int mode = inode->i_mode;

                struct tm tm_time;
                ktime_t timestamp = inode->i_atime.tv_sec;
                time64_to_tm(timestamp, 0, &tm_time);

                char strtime[20];
                snprintf(strtime, sizeof(strtime), "%04d-%02d-%02d %02d:%02d:%02d",
                        tm_time.tm_year + 1900, tm_time.tm_mon + 1, tm_time.tm_mday,
                        tm_time.tm_hour, tm_time.tm_min, tm_time.tm_sec);

                //strftime(strtime, 20, "%Y-%m-%d %H:%M:%S", &timeinfo);
                // printk("\n");

                // printk(KERN_INFO "Current file name: %s\n", file->f_path.dentry->d_name.name);
                log_write("\tCurrent file name: %s\n", file->f_path.dentry->d_name.name);

                // printk(KERN_INFO "File access mode: %o\n", mode & 0777);
                log_write("\tFile access mode: %s\n", get_access_mode_string(file->f_mode));

                // printk(KERN_INFO "Last accessed time: %s", strtime);
                log_write("\tLast accessed time: %s\n", strtime);
            }

        }
    }
    spin_unlock(&files->file_lock);
    return 0;
}


int processFileInfo_task(struct task_struct *task)
{
    struct files_struct *files = task->files;
    struct fdtable *fdt = files_fdtable(files);
    unsigned long fd;
    pid_t task_pid = task_pid_nr(task);

    // printk(KERN_INFO "Searching Process File Table\n");

    spin_lock(&files->file_lock); 
    // pr_info("total open files: %d\n", fdt->max_fds);
    for (fd = 0; fd < fdt->max_fds; fd++) {
        if (test_bit(fd, fdt->open_fds)) {
            struct file *file = fdt->fd[fd];
            if (file && is_regular_file(file)) {

                struct inode *inode = file->f_inode;

                // int mode = inode->i_mode;

                struct tm tm_time;
                ktime_t timestamp = inode->i_atime.tv_sec;
                time64_to_tm(timestamp, 0, &tm_time);

                char strtime[20];
                snprintf(strtime, sizeof(strtime), "%04d-%02d-%02d %02d:%02d:%02d",
                        tm_time.tm_year + 1900, tm_time.tm_mon + 1, tm_time.tm_mday,
                        tm_time.tm_hour, tm_time.tm_min, tm_time.tm_sec);

                //strftime(strtime, 20, "%Y-%m-%d %H:%M:%S", &timeinfo);


                // modifided by X.Wang 2023-5-14 change to log_write interface
                // printk(KERN_INFO "Current file name: %s\n", file->f_path.dentry->d_name.name);
                log_write("\tCurrent file name: %s\n", file->f_path.dentry->d_name.name);

                // printk(KERN_INFO "File access mode: %o\n", mode & 0777);
                log_write("\tFile access mode: %s\n", get_access_mode_string(file->f_mode));

                // printk(KERN_INFO "Last accessed time: %s", strtime);
                log_write("\tLast accessed time: %s\n", strtime);
            }
        }
    }
    spin_unlock(&files->file_lock);
    return 0;
}

bool query_file_in_read(struct task_struct *task, char *file_name)
{
    struct files_struct *files = task->files;
    struct fdtable *fdt = files_fdtable(files);
    unsigned long fd;
    spin_lock(&files->file_lock); 
    for (fd = 0; fd < fdt->max_fds; fd++) {
        if (test_bit(fd, fdt->open_fds)) {
            struct file *file = fdt->fd[fd];
            if (file && is_regular_file(file) && strcmp(file_name, file->f_path.dentry->d_name.name) == 0) {
                spin_unlock(&files->file_lock);
                return true;
            } 
        }
    }
    spin_unlock(&files->file_lock);
    return false;
}

int file2task(char *file_name, int files_len) {
    struct pid *pid; // process PID
    struct task_struct *task;
    struct task_struct *reading_file_task;
    // int reading_file_tasks_n = 0;
    int i;
    i = 0;

    for (;i < files_len; i++) {
        bool file_found = false;
        for_each_process(task) {
            if (query_file_in_read(task, file_name)) {
                file_found = true;
                reading_file_task = task;
                break;
            } 
        }
        if (file_found == false) {
            pr_info("file %s not found opened by some process", file_name);
            return -1;
        }
    }

    if (reading_file_task != NULL) {
        pid = get_task_pid(reading_file_task, PIDTYPE_PID);
        // pr_debug("file %s is read by process %d", file_name, pid->numbers[0].nr);
        log_write("file %s is read by process %d", file_name, pid->numbers[0].nr);
    } else {
        pr_info("file %s is not read", file_name);
    }
    return 0;
}

// static int __init my_init(void)
// {
//     // long long int totalCpuTime = get_totalCpuTime();
//     // get_proCpuTime(1658);
//     // long long int pcpu = cal_cpu_use(1658);
//     // printk("pcpu:%lld",pcpu);
//     processFileInfo(16686);
//     pr_info("end\n");
//     return 0;
// }

// static void __exit my_exit(void)
// {
//     printk(KERN_INFO "Module unloaded.\n");
// }

// module_init(my_init);
// module_exit(my_exit);

// MODULE_LICENSE("GPL");
