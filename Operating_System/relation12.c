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
#include <linux/rcupdate.h>
#include <linux/namei.h>
#include <linux/path.h>
#include <linux/types.h>
#include <linux/hashtable.h>
#include "hash.h"
#include "write_proc/writeproc.h"


/*
 * Edited Xiao Zhang 2023-5-14
 * replace pid_list with task_list
 */
MODULE_LICENSE("GPL");
MODULE_AUTHOR("Xiao Zhang");
MODULE_DESCRIPTION("to print relationship between processes");

/* Function I
 * print the father-chile-brother relation between progresses
 */

// Print the relationship of processes in pid_list
void print_pstree(struct task_struct **task_list, int task_count, struct task_struct *task, int indent, int *flag) {
    /*
    * @task_list: list of tasks
    * @task_count: the number of tasks
    * @task: process
    * @indent: the generation the task is in
    * @flag: whether the task has been printed
    */
    struct task_struct *child;
    char buf[2 * FILENAME_LEN] = {0};
    int i;
    bool found;

    // Check if current process is in task_list
    found = false;
    for (i = 0; i < task_count; i++) {
        if (task->pid == task_list[i]-> pid) {
            found = true;
            break;
        }
    }
    // printk(KERN_INFO "flag[%d]=%d", i, flag[i]);
    if (!found) { return;}
    if (flag[i] == 1){ return;}
    else{ flag[i] = 1;}

    // Add indentation based on current level
    for (i = 0; i < indent; i++) {
        sprintf(buf + strlen(buf), "|    ");
    }

    // Print current process
    // printk(KERN_INFO "%s|-- %s[%d]\n", buf, task->comm, task->pid);
    log_write("%s|-- %s[%d]\n", buf, task->comm, task->pid);
    

    // Recursively print child processes
    list_for_each_entry(child, &task->children, sibling) {
        print_pstree(task_list, task_count, child, indent + 1, flag);
    }
}

// Print the relationforest between processes in pid_list
void process_relationtree(struct task_struct **task_list, int task_count) {
    /*
    * @task_list: list of tasks
    * @task_count: the number of tasks
    */
    int i;
    // record printed processes
    int flag[MAX_PID_NUM] = {0};
    // printk(KERN_INFO "*************Print Relation Tree ***************************\n");
    // printk(KERN_INFO "*************Print Relation Tree ***************************\n");
    log_write("*************Print Relation Tree ***************************\n");

    // Print each process and its children in pid_list
    for (i = 0; i < task_count; i++) {
        if (!flag[i]){
            print_pstree(task_list, task_count, task_list[i], 0, flag);
        }
    }

    // printk(KERN_INFO "***************************************************\n");
    log_write("***************************************************\n");
}

/* Function II 
 * iterating every file accessed by progresses,
 * print processes concurrenting the file
 */
// subfunction: add a node to hash table
void add_to_hash_table(struct task_struct *task, char *path, struct task_struct **task_list, int task_count) {
    // Structure to hold file information
    struct my_data *file_inform;
    const char* filename;
    int pids[MAX_PID_NUM] = {0};
    int pid_count = 0;
    int cnt;

    // Check hash table, if file in hash table, print
    filename = path;
    file_inform = get_data(path);
    if (file_inform == NULL){
        // If the file is not in the hash table, add it
        pids[0] = (int)task->pid;
        insert_data(filename, pids, 1);
    } else {
        // If the file is already in the hash table, add the new pid to the list
        memcpy(pids, file_inform->value, sizeof(int) * MAX_PID_NUM);
        pid_count = file_inform->count;
        
        // if in the table, edited
        for (cnt = 0; cnt < pid_count; cnt++){
            if (pids[cnt] == (int)(task->pid)){
                break;
            }
        }
        if (pid_count < MAX_PID_NUM && cnt == pid_count) {
            pids[pid_count] = (int)(task->pid);
            pid_count++;
            insert_data(filename, pids, pid_count);
        }
    }
}

bool is_regular_file(struct file *file)
{
    struct inode *inode = file->f_inode;
    umode_t mode = inode->i_mode & S_IFMT;
    if (mode == S_IFREG)
    {
        return true;
    }
    return false;    
}

// relation of process concurrent
void process_concurrent(struct task_struct **task_list, int task_count) {
    struct files_struct *files;
    struct fdtable *fdt;
    char path_buffer[FILENAME_LEN];
    // char p_path_buffer[FILENAME_LEN];
    char *path;
    struct path *path_ptr;
    struct file *file;
    struct task_struct *task;
    int i, k;

    // Outer loop through all tasks in task_list
    for (k = 0; k < task_count; k++) {
        task = task_list[k];
        files = task->files;

        if (!files) {
            pr_alert("No files_struct available\n");
            continue;
        }

        fdt = files_fdtable(files);
        if (!fdt) {
            pr_alert("No fdtable available\n");
            continue;
        }

        // Inner loop through the file descriptors of each task
        for (i = 0; i < fdt->max_fds; i++) {
            file = fdt->fd[i];
            if (file && is_regular_file(file)) {
                path_ptr = &file->f_path;
                memset(path_buffer, 0, sizeof(path_buffer));
                path = d_path(path_ptr, path_buffer, FILENAME_LEN);
                if (!IS_ERR(path)) {
                    // printk(KERN_INFO "Process %d has file: %s open\n", task->pid, path);
                    add_to_hash_table(task, path, task_list, task_count);
                }
            }
        }
    }
}

// given pids, return a list of task_struct
struct task_struct **get_task_structs_by_pids(int *pid_list, int pid_count, int *valid_count) {
    // Allocate memory for the task_struct pointers array
    struct task_struct **task_list = kmalloc(pid_count * sizeof(struct task_struct *), GFP_KERNEL);
    if (!task_list) {
        printk(KERN_ERR "Failed to allocate memory for task_list\n");
        return NULL;
    }

    *valid_count = 0;
    for (int i = 0; i < pid_count; i++) {
        int pid = pid_list[i];
        // check_process_state(pid);
        struct task_struct *task = pid_task(find_vpid(pid), PIDTYPE_PID);
        if (task == NULL) {
            printk(KERN_WARNING "task_struct not found for PID %d\n", pid);
            continue;
        }

        task_list[*valid_count] = task;
        (*valid_count)++;
    }
    
    return task_list;
}


// static int __init test_read_init(void)
// {
//     int pid_list[] = {1, 2, 1234, 4421, 5306, 4539};
//     int pid_count = ARRAY_SIZE(pid_list);
//     int valid_count = 0;
//     struct task_struct **task_list;

//     printk(KERN_INFO "Relation Module Loaded!");
//     task_list = get_task_structs_by_pids(pid_list, pid_count, &valid_count);
//     for (int i = 0; i < valid_count; i++){
//         printk(KERN_INFO "Task %d with pid %d", i+1, task_list[i]->pid);
//     }
//     printk(KERN_INFO "Print process tree...\n");
//     process_relationtree(task_list, valid_count);
//     printk(KERN_INFO "Print concurret relation...\n");
//     process_concurrent(task_list, valid_count);
//     iterate_hash_table(1);
//     // print_concurrent_file_access(task_list, valid_count);
//     return 0;
// }

// static void __exit test_exit(void)
// {   
//     struct my_data *data;
//     struct hlist_node *tmp;
//     int bkt;
//     hash_for_each_safe(my_hash_table, bkt, tmp, data, node) {
//         hash_del(&data->node);
//         kfree(data->value);
//         kfree(data);
//     }
//     pr_info("Relation Module Unload!\n");
// }

// module_init(test_read_init);
// module_exit(test_exit);




