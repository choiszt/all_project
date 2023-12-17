#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/sched.h>
#include <linux/in.h>
#include <net/sock.h>
#include <linux/inet.h>
#include <linux/fdtable.h>
#include <net/inet_sock.h>
#include "write_proc/writeproc.h"

MODULE_LICENSE("GPL");

static int pid = -1; // 默认的PID为-1，表示未指定

int process_ip(struct task_struct *task)
{
    struct sock *sk; // 定义套接字
    struct sk_buff *skb;
    struct inet_sock *inet; // 定义INET套接字
    pid_t task_pid = task_pid_nr(task);

    /* modified by X.Wang 2023-4-1
     * success get task_struct
     * modified by X.Wang 2023-3-26
     * add exception handling
     */
    if (task == NULL)
    {
        pr_info( "No such process, pid: %d\n", pid);
        return -1;
    }
    struct files_struct *files = task->files;
    struct fdtable *fdt = files_fdtable(files);
    unsigned long fd;

    // printk(KERN_INFO "Searching Process File Table\n");

    spin_lock(&files->file_lock); 
    // pr_info("total open files: %d\n", fdt->max_fds);
    for (fd = 0; fd < fdt->max_fds; fd++) {
        if (test_bit(fd, fdt->open_fds)) {
            struct file *file = fdt->fd[fd];
            if (file->f_inode && S_ISSOCK(file->f_inode->i_mode)) {// 判断是否为套接字
                struct socket *sock = sock_from_file(file);
                if (sock == NULL) {
                    pr_info("sock is NULL\n");
                    continue;
                }

                struct sockaddr src_addr, dest_addr;
                int remote = sock->ops->getname(sock, &dest_addr, 1);
                int local = sock->ops->getname(sock, &src_addr, 0);
                if (remote < 0 || local < 0) {
                    pr_info("getname failed\n");
                    continue;
                }
                
                if (src_addr.sa_family != AF_INET || dest_addr.sa_family != AF_INET) {
                    pr_info("not ipv4\n");
                    continue;
                }

                // cast the sockaddr to sockaddr_in                
                struct sockaddr_in *src_addr_in = (struct sockaddr_in *)&src_addr;
                struct sockaddr_in *dest_addr_in = (struct sockaddr_in *)&dest_addr;

                /* print out the ipv4 address
                 * It should be note that the format `%pI4' take a PONITER!!! to a u32(big endian)
                 */
                // pr_info("Local address: %pI4\n", &(src_addr_in->sin_addr));
                // log_write("Process: %d\n", task_pid);
                log_write("\tLocal address: %pI4\n", &(src_addr_in->sin_addr));
                // pr_info("Remote address: %pI4\n", &(dest_addr_in->sin_addr));
                log_write("\tRemote address: %pI4\n", &(dest_addr_in->sin_addr));
            }
        }
    }
    spin_unlock(&files->file_lock);
    return 0;
}

// static int __init my_init(void)
// {
    
//     struct task_struct *task; // 定义进程任务结构体

//     printk(KERN_INFO "Module loaded\n");

//     if (pid < 0) {
//         printk(KERN_INFO "PID not specified. Exiting.\n");
//         return -1;
//     }

//     rcu_read_lock(); // 读取进程列表时加锁

//     task = pid_task(find_vpid(pid), PIDTYPE_PID); // 根据PID获取进程任务结构体

//     if (!task) {
//         printk(KERN_INFO "Invalid PID. Exiting.\n");
//         rcu_read_unlock();
//         return -1;
//     }

//     rcu_read_unlock(); // 释放锁

//     return 0;
// }

// static void __exit my_exit(void)
// {
//     printk(KERN_INFO "Module unloaded\n");
// }

// module_init(my_init);
// module_exit(my_exit);