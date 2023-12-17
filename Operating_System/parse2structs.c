#include "read.h"
#include "parse2structs.h"
#include <linux/fs.h>
#include <linux/sched.h>
#include <linux/slab.h>
#include <linux/namei.h>


extern struct inode *get_inode(const char *path);
// {
//     struct path name_path;
//     struct inode *inode = NULL;
//     int error;

//     error = kern_path(path, LOOKUP_FOLLOW, &name_path);
//     if (!error) {
//         inode = name_path.dentry->d_inode;
//         if (inode)
//             ihold(inode);
//         path_put(&name_path);
//     }
//     else {
//         pr_err("Failed to get inode of %s. Error code: %d\n", path, error);
//     }

//     return inode;
// }

struct parser_result_structs struct_parse_result(struct parser_result *res) {
    struct parser_result_structs ret = {
        // .tasks = (struct task_struct**) kmalloc(sizeof(struct task_struct*) * (res->pids_len), GFP_KERNEL),
        // .tasks_len = res->pids_len,//+res->progs_len,
        .inodes = (struct inode**) kmalloc(sizeof(struct inode*) * res->files_len, GFP_KERNEL),
        .inodes_len = res->files_len,
    };
  
    int i;
    // 根据pid获取对应的task_struct
    // for (i = 0; i < res->pids_len; i++) {
    //     ret.tasks[i] = pid_task(find_vpid(res->pids[i]), PIDTYPE_PID);
    //     if (!ret.tasks[i]) {
    //         // pid对应的进程不存在，报错并退出
    //         printk(KERN_ERR "Process with pid %d does not exist.\n", res->pids[i]);
    //         goto error;
    //     }
    // }
    // 根据program获取对应的task_struct
    //  for (i = res->pids_len; i < res->pids_len+res->progs_len; i++) {
    //     struct task_struct *task = NULL;
    //     for_each_process(task) {
    //         if (strcmp(task->comm, res->progs[i]) == 0) {
    //             ret.tasks[i] = task;
    //             break;
    //         }
    //     }
    //     if (!ret.tasks[i]) {
    //         // program对应的进程不存在，报错并退出
    //         printk(KERN_ERR "Process with program name %s does not exist.\n", res->progs[i- res->pids_len]);
    //         goto error;
    //     }
    // }
    // 根据file获取对应的inode
    for (i = 0; i < res->files_len; i++) {
        // get_inode(res->files[i]);
        ret.inodes[i] = get_inode(res->files[i]);
    }
    return ret;

error:
    // 出现错误时，释放已经分配的内存
    // for (i = 0; i < ret.tasks_len; i++) {
    //     if (ret.tasks[i]) {
    //         put_task_struct(ret.tasks[i]);
    //     }
    // }
    for (i = 0; i < ret.inodes_len; i++) {
        if (ret.inodes[i]) {
            iput(ret.inodes[i]);
        }
    }
    kfree(ret.tasks);
    kfree(ret.inodes);
    // 将长度设置为0，避免在其他模块中出现释放错误的情况
    ret.tasks_len = 0;
    ret.inodes_len = 0;
    return ret;
}
