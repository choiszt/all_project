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
#include "read.h"
#include "relation.h"

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Xiao Zhang");
MODULE_DESCRIPTION("to print relationship between processes");

/* Function I
 * print the father-chile-brother relation between progresses
 */

// Print the relationship of processes in pid_list
void print_pstree(int *pid_list, int pid_count, struct task_struct *task, int indent, int *flag) {
    /*
    * @pid_list: list of pids
    * @pid_count: the number of pids
    * @task: process
    * @indent: the generation the task is in
    */
    struct task_struct *child;
    char buf[100] = {0};
    int i, j;
    bool found;

    // Check if current process is in pid_list
    found = false;
    for (i = 0; i < pid_count; i++) {
        if (task->pid == pid_list[i]) {
            found = true;
            break;
        }
    }
    // printk(KERN_INFO "flag[%d]=%d", i, flag[i]);
    if (!found) {
        return;
    }
    if (flag[i] == 1){
        return;
    }
    else{
        flag[i] = 1;
    }

    // Add indentation based on current level
    for (i = 0; i < indent; i++) {
        sprintf(buf + strlen(buf), "|    ");
    }

    // Print current process
    printk(KERN_INFO "%s|-- %s[%d]\n", buf, task->comm, task->pid);
    

    // Recursively print child processes
    list_for_each_entry(child, &task->children, sibling) {
        print_pstree(pid_list, pid_count, child, indent + 1, flag);
    }
}

// Print the relationforest between processes in pid_list
void process_relationtree(int *pid_list, int pid_count) {
    /*
    * @pid_list: list of pids
    * @pid_count: the number of pids
    */
    int i, j;
    struct pid *pid_struct;
    struct task_struct *task;
    // record printed processes
    int flag[PROCESS_NUM] = {0};
    printk(KERN_INFO "*************Print Relation Tree ***************************\n");

    // Print each process and its children in pid_list
    for (i = 0; i < pid_count; i++) {
        if (flag[i]){
            continue;
        }
        // printk(KERN_INFO "flag[%d]=%d", i, flag[i]);
        pid_struct = find_get_pid(pid_list[i]);
        if (!pid_struct) {
            printk(KERN_INFO "Process with PID %d not found\n", pid_list[i]);
            continue;
        }
        task = pid_task(pid_struct, PIDTYPE_PID);
        if (!task) {
            printk(KERN_INFO "Process with PID %d not found\n", pid_list[i]);
            continue;
        }
        if (!flag[i]){
            print_pstree(pid_list, pid_count, task, 0, flag);
        }
    }

    printk(KERN_INFO "***************************************************\n");
}

// print the relationship of process read
void process_relationship(int *pid_list, int pid_count)
{
    /*
    * @pid_list: list of pids
    * @pid_count: the number of pids
    */
    int i, j;
    printk(KERN_INFO "**********Relations Between Processes **********************\n");

    // 遍历 pid_list 中的进程
    for (i = 0; i < pid_count; i++) {
        struct pid *pid_struct = find_get_pid(pid_list[i]);
        if (!pid_struct) {
            printk(KERN_INFO "Process with PID %d not found\n", pid_list[i]);
            continue;
        }
        struct task_struct *task = pid_task(pid_struct, PIDTYPE_PID);
        // if find_vpid(pid) == NULL, the kernel corupts
        // struct task_struct *task = pid_task(find_vpid(pid_list[i]), PIDTYPE_PID);
        if (task == NULL) {
            printk(KERN_INFO "Process with PID %d not found\n", pid_list[i]);
            continue;
        }

        // 输出进程的父子关系
        if (task->real_parent != NULL) {
            for (j = 0; j < pid_count; j++) {
                // Judge if the process in pid_list
                struct task_struct *another_task = pid_task(find_vpid(pid_list[j]), PIDTYPE_PID);
                if (task->real_parent->pid == another_task->pid){
                    printk(KERN_INFO "Process with PID %d has parent with PID %d\n", task->pid, task->real_parent->pid);
                    break;
                }
            }
        }
        if (!list_empty(&task->children)) {
            struct task_struct *child;
            list_for_each_entry(child, &task->children, sibling) {
                // Judge if the process in pid_list
                for (j = 0; j < pid_count; j++){
                    struct task_struct *another_task = pid_task(find_vpid(pid_list[j]), PIDTYPE_PID);
                    if (child->pid == another_task->pid){
                        printk(KERN_INFO "Process with PID %d has child with PID %d\n", task->pid, child->pid);
                        break;
                    }
                }
            }
        }

        // 输出进程之间的兄弟关系,但是只能找到pid比自身小的
        if (i > 0) {
            for (j = 0; j < i; j++) {
                struct task_struct *sibling_task = pid_task(find_vpid(pid_list[j]), PIDTYPE_PID);
                if (sibling_task != NULL && sibling_task->real_parent == task->real_parent) {
                    printk(KERN_INFO "Process with PID %d and process with PID %d are siblings\n", task->pid, sibling_task->pid);
                }
            }
        }
    }
}

/* Function II 
 * iterating every file accessed by progresses,
 * print processes concurrenting the file
 */

// judge if the file is valid 
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

// given pid, get every name of file visited by the process
struct process_file_info process_file(int pid)
{
    struct task_struct *task = pid_task(find_vpid(pid), PIDTYPE_PID);
    struct files_struct *files = task->files;
    struct fdtable *fdt = files_fdtable(files);
    unsigned long fd;

    /*
     * 2023-4-8 edited by ZhangXiao
     * store file information into a struct
     */
    struct process_file_info file_info;
    file_info.process_name = task->comm;
    file_info.pid = pid;

    /*
    * 2023-4-8 edited by ZhangXiao
    Add a list to store all files(size limited 1024)
    */
    char **file_list;
    int file_count = 0;
    file_list = kmalloc(sizeof(char*) * MAX_LIST_SIZE, GFP_KERNEL);

    printk(KERN_INFO "Searching Process File Table\n");

    spin_lock(&files->file_lock); 
    // pr_info("total open files: %d\n", fdt->max_fds);
    for (fd = 0; fd < fdt->max_fds; fd++) {
        if (test_bit(fd, fdt->open_fds)) {
            struct file *file = fdt->fd[fd];
            if (file && is_regular_file(file)) {
                char *filename = file->f_path.dentry->d_name.name;
                /* 2023-4-9 Edited by ZhangXiao
                 * Judge if the filename is in the list
                 */
                bool file_found = false;
                for (int k = 0; k < file_count; k++) {
                    if (strcmp(filename, file_list[k]) == 0){file_found=true; break;}
                }
                if (!file_found){ // if file not in file list, add a file
                    int filename_length = strlen(filename);
                    file_list[file_count] = kmalloc(sizeof(char) * (filename_length + 1), GFP_KERNEL);
                    strncpy(file_list[file_count], filename, filename_length + 1);
                    // printk(KERN_INFO "File list added: %s", filename);
                    file_count++;
                }
                
            } 
        }
    }
    spin_unlock(&files->file_lock);

    // inform the num of file
    printk(KERN_INFO "***** Process with PID %d has accessed %d files********", pid, file_count);

    // kfree(file_list);
    
    file_info.file_count = file_count;
    file_info.file_list = file_list;
    // print top 10 files and the last
    for (int i = 0; i < file_count; i++) {
        if (i == 10 && file_count > 10){
            printk(KERN_INFO "...");
        }
        else if (i < 10 || i == file_count - 1){
            printk(KERN_INFO "File name %d: %s \n", i + 1, file_list[i]);
        }
    }
    
    return file_info;
}

void process_concurrent(int *pid_list, int pid_count){
    // print start information 
    printk(KERN_INFO "********Start Building Hash Map! ******");
    // allocate memory for the array of file_info structs
    struct file_info *fi_list = kmalloc(HASH_TABLE_SIZE * sizeof(struct file_info), GFP_KERNEL);
    // num of file in file_set
    int file_num = 0;

    // record the number of file concurrented by processes
    int concurrented_file = 0;

    for (int i = 0; i < pid_count; i++) {
        printk(KERN_INFO "Loading process_file_info module with pid = %d\n", pid_list[i]);
        struct process_file_info pfi = process_file(pid_list[i]);
        // PFI_list[i] = pfi;
        char **file_list = pfi.file_list;
        int file_count = pfi.file_count;
        // iterate files accessed by the process
        for (int j = 0; j < file_count; j++) {
            bool found = false;
            // iterate fi_list, found the filename
            if (file_num > 0){
                for (int k = 0; k < file_num; k++) {
                    // if file in fi_list, add pid_list[i] into the list
                    if (strcmp(fi_list[k].filename, file_list[j]) == 0) {
                        // find the filename
                        found = true;
                        // find if pid in pid_list
                        bool pid_found = false;
                        for (int l = 0; l < fi_list[k].pid_count; l++){
                            if (fi_list[k].pid_list[l] == pid_list[i]) {
                                pid_found = true;
                                break;
                            }
                        }
                        if (!pid_found) {
                            fi_list[k].pid_list[fi_list[k].pid_count] = pid_list[i];
                            fi_list[k].pid_count++;
                        }
                        break;
                    } 
                }
            }
            
            // if file not in fi_list, create a node
            if (!found) {
                // printk(KERN_INFO "process access file %s:", file_list[j]);
                struct file_info new_fi;
                new_fi.filename = file_list[j];
                // int new_pid_list[MAX_PID_NUM] = {0};
                int *new_pid_list = kmalloc(MAX_PID_NUM * sizeof(int), GFP_KERNEL);
                new_pid_list[0] = pid_list[i];
                new_fi.pid_list = new_pid_list;
                new_fi.pid_count = 1;
                fi_list[file_num++] = new_fi;
            }
        
        }
    }
    printk(KERN_INFO "*******************************");
    printk(KERN_INFO "size of fi_list : %d", file_num);
    
    
    // out put files concurrented by processes
    for (int i = 0; i < file_num; i++) {
        // printk(KERN_INFO "File: %s, pids: %d", fi_list[i].filename, fi_list[i].pid_count);
        if (fi_list[i].pid_count > 1) {
            concurrented_file++;
            int pcount = fi_list[i].pid_count;
            // int *pids = fi_list[i].pid_list;
            printk(KERN_INFO "File '%s' is accessed by:", fi_list[i].filename);
            for (int j = 0; j < pcount; j++){
                printk(KERN_INFO "%d Process with PID %d.", j+1, fi_list[i].pid_list[j]);
            }
        }
    }
    printk(KERN_INFO "Num of files concurrented: %d ",concurrented_file);
    printk(KERN_INFO "************ End of file relations! ********");
    
    // free the memory allocated for the array of file_info structs
    kfree(fi_list);
    return;
    // return fi_list;
}


// print relations between pids
void check_relations (struct parser_result ans){
    int *pid_list = ans.pids;
    int pid_count = ans.pids_len;
    // print relations
    process_relationship(pid_list, pid_count);
    process_relationtree(pid_list, pid_count);
    process_concurrent(pid_list, pid_count);
}



