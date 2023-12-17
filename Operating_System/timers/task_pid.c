#include <linux/module.h>
#include <linux/sched.h>
#include <linux/pid.h>
MODULE_LICENSE("GPL");
static int __init __task_pid_nr_ns_init(void)
{
    printk("into __task_pid_nr_ns_init.\n");

    //获取当前进程的进程描述符，current为struct task_struct类型变量，记录当前进程的信息
    struct pid * kpid=find_get_pid(current->pid);

    // 获取进程所属任务的任务描述符
    struct task_struct * task=pid_task(kpid, PIDTYPE_PID);

    // 获取任务对应进程的进程描述符
    pid_t result1=__task_pid_nr_ns(task, PIDTYPE_PID, kpid->numbers[kpid->level].ns);

    //显示函数find_get_pid( )返回值的进程描述符的进程号
    printk("the pid of the find_get_pid is :%d\n", kpid->numbers[kpid->level].nr);

    //显示函数__task_pid_nr_ns( )的返回值
    printk("the result of the __task_pid_nr_ns is:%d\n", result1);
    printk("the pid of current thread is :%d\n", current->pid);  //显示当前进程号
    printk("out __task_pid_nr_ns_init.\n");
    return 0;
}
static void __exit __task_pid_nr_ns_exit(void)
{
    printk("Goodbye __task_pid_nr_ns\n");
}
module_init(__task_pid_nr_ns_init);
module_exit(__task_pid_nr_ns_exit);