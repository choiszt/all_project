#include <linux/init.h>
#include <linux/module.h>
#include <linux/string.h>
#include <linux/slab.h>
#include <linux/kernel.h>
#include <linux/fs.h>
#include "read.h"
#include "check_inform.h"

// 输出进程状态，当进程not found的时候返回-1
int check_process_state(int pid) {
    // rcu_read_lock(); // 在读取task_struct成员前加上RCU读锁
    printk(KERN_INFO "Process %d", pid);
    struct task_struct *task = pid_task(find_vpid(pid), PIDTYPE_PID);
    if (task == NULL) {
        printk(KERN_INFO "Process %d not found.\n", pid);
        return -1;
    }
    unsigned long state = task->__state;
    if (state == TASK_RUNNING) {
        printk(KERN_INFO "Process %d is running.\n", pid);
    } else if (state == TASK_INTERRUPTIBLE || state == TASK_UNINTERRUPTIBLE) {
        printk(KERN_INFO "Process %d is waiting.\n", pid);
    } else if (state == TASK_STOPPED) {
        printk(KERN_INFO "Process %d is stopped.\n", pid);
    } else if ((int)(state) == 1026){
    	printk (KERN_INFO "Process %d is blocked\n", pid); 
    } else {
        printk(KERN_INFO "Process %d is in state %lu.\n", pid, state);
    }
    
    return 0;
}

void check_parser (struct parser_result res){
    int i;
    printk (KERN_INFO "*************Check initial Result*****************\n");
    i = 0;
    for (; i < res.pids_len; i++) {
        pr_info("pids: %ld", res.pids[i]);
    }
    i = 0;
    for (; i < res.progs_len; i++) {
        pr_info("progs: %s", res.progs[i]);
    }
    i = 0;
    for (; i < res.files_len; i++) {
        pr_info("files: %s", res.files[i]);
    }
}

// screen out invalid pids
int filter_existing_pids(int *pid_list, int pid_count, int *new_pid_list) {
    int i, new_count = 0;
    printk (KERN_INFO "*************Check Process States*****************\n");
    for (i = 0; i < pid_count; i++) {
        if (check_process_state(pid_list[i]) == 0) {
            new_pid_list[new_count++] = pid_list[i];
        } 
    }
    return new_count;
}

void edit_result (struct parser_result *ans){
    // init, fliter existing pids
    int *pid_list = ans->pids;
    int pid_count = filter_existing_pids(pid_list, ans->pids_len, pid_list);
    printk(KERN_INFO "*************Check valid count*********************\n");
    // check if pid_count > 0
    if (pid_count == 0) {
        printk(KERN_INFO "Process list does not contain valid PID\n");
        return ;
    }
    printk (KERN_INFO "valid count = %d\n", pid_count);
    for (int i = 0; i < pid_count; i++){
    	printk (KERN_INFO "valid pid: %d\n", pid_list[i]);
    }
    // edit ans value
    ans->pids = pid_list;
    ans->pids_len = pid_count;
}






