#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/sched/signal.h>
#include <linux/sched.h>
#include <linux/slab.h>
#include <linux/pid.h>
#include <linux/string.h>
#include <linux/pid_namespace.h>


int prog2task(char *process_name, struct task_struct *task_in) {
    struct pid *pid; // process PID
    struct task_struct *task;
    char task_name[TASK_COMM_LEN] = {0}; // process name

    for_each_process(task) {
        get_task_comm(task_name, task); // get process name
        if (strcmp(task_name, process_name) == 0) { // compare process name
            pid = get_task_pid(task, PIDTYPE_PID); // get process PID
            break;
        }
    }

    if (task != NULL) {
        pr_info("Process \"%s\" found, PID: %d\n", process_name, pid_nr(pid));
        pr_info("task->pid: %d, task->tgid: %d\n", task->pid, task->tgid);
    } else {
        pr_info("Process \"%s\" not found\n", process_name);
    }

    // copy task_struct
    memcpy(task_in, task, sizeof(struct task_struct));

    return 0;
}

// static int __init prog2task_init(void) {
//     printk(KERN_INFO "prog2task module loaded\n");
//     struct task_struct *task = (struct task_struct *) kmalloc(sizeof(struct task_struct), GFP_KERNEL);
//     if (task == NULL) {
//         pr_alert("kmalloc failed\n");
//         return -1;
//     }
//     prog2task("python3", task);
//     pr_info("[main] task->pid: %d, task->tgid: %d\n", task->pid, task->tgid);
//     return 0;
// }

// static void __exit prog2task_exit(void) {
//     printk(KERN_INFO "prog2task module unloaded");
// }

// module_init(prog2task_init);
// module_exit(prog2task_exit);

// MODULE_LICENSE("GPL");
// MODULE_AUTHOR("X.Wang");
// MODULE_DESCRIPTION("prog2task");