#include <linux/module.h>
#include <linux/slab.h>
#include <linux/namei.h>
#include <linux/fcntl.h>
#include <linux/uaccess.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/delay.h>
#include <linux/sched.h>
#include <linux/mm.h>
#include <linux/time.h>
#include <linux/types.h>
#include <linux/kernel_stat.h>
#include <linux/sched/signal.h>
#include <linux/proc_fs.h>
#include <linux/ktime.h>
#include <linux/fdtable.h>
#include <linux/fs.h>
#include <linux/file.h>
#include <linux/limits.h>
#include "write_proc/writeproc.h"

struct inode *get_inode(const char *path)
{
    struct path name_path;
    struct inode *inode = NULL;
    int error;

    error = kern_path(path, LOOKUP_FOLLOW, &name_path);
    if (!error) {
        inode = name_path.dentry->d_inode;
        if (inode)
            ihold(inode);
        path_put(&name_path);
    }
    else {
        pr_err("Failed to get inode of %s. Error code: %d\n", path, error);
    }

    return inode;
}

void file_process_info(const char *path)
{
    struct inode *inode = get_inode(path);
    // struct file *file;
    // ssize_t bytes;

    // file = filp_open(path, O_RDONLY, 0);
    // if (IS_ERR(file)) {
    //     long err_code = PTR_ERR(file);
    //     pr_err("Failed to open file %s. Error code: %ld\n", path, err_code);
    //     // return -1;
    //     return ; // void function
    // }

    // struct inode *inode = file_inode(file);
    struct task_struct *task;
    for_each_process(task)
    {
        if (task->files)
        {
            // Iterate through the process's file descriptor table
            int i;
            for (i = 0; i < files_fdtable(task->files)->max_fds; ++i)
            {
                // printk("\n");
                struct file *filp = files_fdtable(task->files)->fd[i];
                if (filp == NULL)
                {
                    continue;
                }

                // Check to see if the file descriptor is for the given file
                if (filp->f_inode == inode)
                {
                    // Log the information about this process's access of the file  
                    // printk("PID: %d, Program Name: %s, Access Mode: %o, Access Time: %ld\n", task->pid, task->comm, filp->f_mode, ktime_get_real_seconds());
                    log_write("\tProgram Name: %s, Access Mode: %o, Access Time: %ld\n", task->comm, filp->f_mode, ktime_get_real_seconds());
                }
            }
        }
    }
}

static char *mode_to_string(mode_t access_mode)
{
    char *mode_str = kmalloc(11 * sizeof(char), GFP_KERNEL);

    // 文件类型
    if (S_ISREG(access_mode))
    {
        mode_str[0] = '-';
    }
    else if (S_ISDIR(access_mode))
    {
        mode_str[0] = 'd';
    }
    else if (S_ISLNK(access_mode))
    {
        mode_str[0] = 'l';
    }
    else
    {
        mode_str[0] = '?';
    }

    // 文件用户权限
    mode_str[1] = (access_mode & S_IRUSR) ? 'r' : '-';
    mode_str[2] = (access_mode & S_IWUSR) ? 'w' : '-';
    mode_str[3] = (access_mode & S_IXUSR) ? 'x' : '-';

    // 文件组权限
    mode_str[4] = (access_mode & S_IRGRP) ? 'r' : '-';
    mode_str[5] = (access_mode & S_IWGRP) ? 'w' : '-';
    mode_str[6] = (access_mode & S_IXGRP) ? 'x' : '-';

    // 其他用户权限
    mode_str[7] = (access_mode & S_IROTH) ? 'r' : '-';
    mode_str[8] = (access_mode & S_IWOTH) ? 'w' : '-';
    mode_str[9] = (access_mode & S_IXOTH) ? 'x' : '-';
    mode_str[10] = '\0'; // 字符串结束符

    return mode_str;
}

const char* get_access_mode_string(fmode_t mode) {
    char *buf = kzalloc(8, GFP_KERNEL);
    // const char* str;
    pr_info("mode: %d\n", mode);
    buf[0] = '\0';
    if (mode & FMODE_READ)
        strcat(buf, "r");
    if (mode & FMODE_WRITE)
        strcat(buf, "w");
    if (mode & FMODE_EXEC)
        strcat(buf, "x");
    if (mode & FMODE_PREAD)
        strcat(buf, "a");
    return buf;
}

// 将 Unix 时间戳转换为字符串格式的时间
static void convert_time(time64_t access_time, char *buf,size_t buflen)
{
    if (buflen < 20)
    { //防止buffer过小
        printk(KERN_ERR "buffer过小\n");
        return;
    }
    struct tm timeinfo;
    time64_to_tm(access_time, 0, &timeinfo);
    snprintf(buf, buflen, "%04d-%02d-%02d %02d:%02d:%02d", timeinfo.tm_year + 1900, timeinfo.tm_mon + 1, timeinfo.tm_mday,
             timeinfo.tm_hour, timeinfo.tm_min, timeinfo.tm_sec);
}


void file_process_info_inode(struct inode *inode)
{
    struct task_struct *task;
    for_each_process(task)
    {
        if (task->files)
        {
            // Iterate through the process's file descriptor table
            int i;
            for (i = 0; i < files_fdtable(task->files)->max_fds; ++i)
            {
                // printk("\n");
                struct file *filp = files_fdtable(task->files)->fd[i];
                if (filp == NULL)
                {
                    continue;
                }

                // Check to see if the file descriptor is for the given file
                if (filp->f_inode == inode)
                {
                    // Log the information about this process's access of the file  
                    // printk("PID: %d, Program Name: %s, Access Mode: %o, Access Time: %ld\n", task->pid, task->comm, filp->f_mode, ktime_get_real_seconds());
                    // log_write("File: %s\n", filp->f_path.dentry->d_name.name);
                    char * time_buf = kmalloc(21 * sizeof(char), GFP_KERNEL);
                    char * mode_str = get_access_mode_string(filp->f_mode);
                    convert_time(ktime_get_real_seconds(), time_buf, 21);
                    log_write("\tPID: %d, Program Name: %s, Access Mode: %s, File Permissions: %s, Access Time: %s\n", task->pid, task->comm, mode_str, mode_to_string(inode->i_mode), time_buf);
                    // log_write("\tPID: %d, Program Name: %s, File Permissions: %s, Access Time: %s\n", task->pid, task->comm, mode_to_string(inode->i_mode), time_buf);
                    kfree(time_buf);
                    kfree(mode_str);
                }
            }
        }
    }
}