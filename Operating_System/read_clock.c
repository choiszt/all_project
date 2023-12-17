#include <linux/fs.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/timer.h>
#include <linux/kthread.h>
#include <linux/slab.h>
#include <linux/sched.h>
#include <linux/smp.h>
#include "read.h"
#include "start_stop.h"
#include "write_proc/writeproc.h"
#include "parse2structs.h"


#define BUF_SIZE 1024
#define READ_INTERVAL (10 * HZ) // 10 seconds

/* #define INTERVAL_EXE(interval, func, args) \
//     static void interval_exe(struct timer_list *t) { \
//         pr_info("current jiffies: %lu", jiffies); \
//         func(args); \
//         mod_timer(&interval, jiffies + READ_INTERVAL); \
//     } \
//     timer_setup(&read_timer, interval_exe, 0);*/

static struct timer_list read_timer;
static bool timer_started = false;
static struct parser_result *given_data;
struct parser_result_structs *given_data_structs;
extern int processFileInfo(int pid);
extern int processFileInfo_task(struct task_struct *task);
extern int prog2task(char *process_name, struct task_struct *task_in);
extern int process_ip(struct task_struct *task);
extern int file2task(char *file_name, int files_len);
extern void file_process_info(const char *path);
extern void file_process_info_inode(struct inode *inode);
static struct task_struct *task;
extern enum clock_state clock_state;
extern struct pid_namespace *ins_ns;
extern void record_cpu_time(u64 *total_time, u64 *idle_time, int module_cpu_id);
// extern long long int cal_cpu_use(int pid);
extern int get_phy_mem(struct task_struct *task);
extern int get_sys_mem_info(void);

static void interval_exe(struct timer_list *t)
{

    pr_debug("current jiffies: %lu", jiffies);
    pr_debug("get given_data->pids_len: %d", given_data->pids_len);
    // reset the timer for the next read
    int i;
    // for(i=0; i<given_data_structs->tasks_len;i++) {
    //     processFileInfo_task(given_data_structs->tasks[i]);
    //     process_ip(given_data_structs->tasks[i]);
    //     get_phy_mem(given_data_structs->tasks[i]);
    // }

    
    for (i = 0; i < given_data->pids_len; i++) {
        // for pids
        log_write("PID: %d\n", given_data->pids[i]);
        pr_debug("get given_data->pids[%d]: %d", i, given_data->pids[i]);
        processFileInfo(given_data->pids[i]);
        // struct task_struct *task = pid_task(find_vpid(given_data->pids[i]), PIDTYPE_PID);
        struct task_struct *task = pid_task(find_pid_ns(given_data->pids[i], ins_ns), PIDTYPE_PID);
        // get_phy_mem(task);
        if (task != NULL) {
            process_ip(task);
            get_phy_mem(task);
        }
        // process_ip(task);
        // cal_cpu_use(given_data->pids[i]);
        pr_debug("processFileInfo finished");
    }
    for (i = 0; i < given_data->progs_len; i++) {
        // for progs
        pr_debug("get given_data->progs[%d]: %s", i, given_data->progs[i]);
        log_write("Program: %s\n", given_data->progs[i]);
        memset(task, 0, sizeof(struct task_struct));    // clear the task struct, reuse the memory
        prog2task(given_data->progs[i], task);
        if (task != NULL) {
            processFileInfo_task(task);
            process_ip(task);
            get_phy_mem(task);
        }
        pr_debug("processFileInfo_task finished");
    }
    for(i=0; i<given_data_structs->inodes_len;i++) {
        log_write("File: %s\n", given_data->files[i]);
        file_process_info_inode(given_data_structs->inodes[i]);
    }
    // for (i = 0; i < given_data->files_len; i++) {
    //     log_write("File: %s\n", given_data->files[i]);
    //     pr_debug("get given_data->progs[%d]: %s", i, given_data->files[i]);
    //     // file2task(given_data->files[i], given_data->files_len);
    //     file_process_info(given_data->files[i]);
    //     pr_debug("file2task finished");
    // }

    // 获取CPU信息
    struct task_struct *task = get_current();
    ktime_t utime, stime, module_total_time;
    // 获取当前进程的utime和stime
    utime = task->utime;
    stime = task->stime;
    module_total_time = utime + stime;
    int module_cpu;
    u64 idle_time, total_time;  // get CPU where module running
    module_cpu = get_cpu();
    put_cpu();
    record_cpu_time(&total_time, &idle_time, module_cpu);
    log_write("Module usage info:\n");
    log_write("\tget times: %lld %lld\n", total_time, idle_time);
    log_write("\tget module time: %lld\n", module_total_time);
    // pr_info("detect CPU usage: %d%%", (total_time - idle_time) * 100 / total_time);
    // pr_info("detect module usage: %d%%", module_total_time * 100 / total_time);
    log_write("\tdetect CPU usage: %d%%\n", (total_time - idle_time) * 100 / total_time);
    log_write("\tdetect module usage: %d%%\n", module_total_time * 100 / total_time);
    get_phy_mem(current);

    // get_sys_mem_info();

    pr_debug("clock state: %d\n", clock_state);
    if (clock_state == START) {
        // 重新设置定时器的到期时间，实现每分钟执行一次任务
        mod_timer(&read_timer, jiffies + READ_INTERVAL);
        pr_debug("mod_timer\n");
    }
}

int start_timer(struct parser_result *parser_result, struct parser_result_structs *structs_parser_result)
{
    // Initialize the task struct
    task = (struct task_struct *) kmalloc(sizeof(struct task_struct), GFP_KERNEL);
    if (task == NULL) {
        pr_err("kmalloc failed\n");
        return -1;
    }
    given_data = parser_result;
    given_data_structs = structs_parser_result;
    timer_setup(&read_timer, interval_exe, 0);
    clock_state = START;
    timer_started = true;

    mod_timer(&read_timer, jiffies + READ_INTERVAL);

    return 0;
}

int stop_timer(void)
{
    clock_state = STOP;

    del_timer_sync(&read_timer);

    return 0;
}

/*
 *这个例子中，我们首先定义了一个全局变量read_timer，用于存储定时器的信息。然后我们在初始化函数中打开了文件，并使用timer_setup和mod_timer函数初始化并启动了定时器。我们指定了定时器到期后要执行的函数为read_file，并设置了第一次到期的时间为当前时间加上READ_INTERVAL，即60秒。在read_file函数中，我们和之前一样读取文件，并在读完后重新设置定时器的到期时间为当前时间加上READ_INTERVAL，从而实现每分钟读取一次文件。在退出函数中，我们使用del_timer_sync函数停止并删除了定时器，并关闭了文件。
 */
