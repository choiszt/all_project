#include <linux/module.h>
//#include <unistd.h>
//#include <sys/types.h>
#include <linux/fs.h>
#include <linux/slab.h>
#include <linux/fcntl.h>
#include <linux/uaccess.h>
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
#include <linux/ktime.h>
#include <linux/module.h>
#include <linux/init.h>
#include <linux/sched.h>
#include <linux/fdtable.h>
#include <linux/net.h>
#include <linux/inet.h>
#include <linux/inet_diag.h>
#include <linux/fs.h>
#include <linux/file.h>
#include <linux/fcntl.h>
#include <linux/limits.h>
#include <linux/string.h>
#include "write_proc/writeproc.h"
int get_sys_mem_info(void)
{
	struct sysinfo si;
	si_meminfo(&si);
	log_write("\tTotal memory: %lu kB\n", si.totalram * si.mem_unit / 1024);
    log_write("\tFree memory: %lu kB\n", si.freeram * si.mem_unit / 1024);
    struct task_struct *task;
	//printk("procs number: %s \n", si.procs);
	//printk("uptime: %li \n", si.uptime);
	return 0;
}



int get_phy_mem(struct task_struct *task)
{
	
	// struct task_struct *task = pid_task(find_vpid(pid), PIDTYPE_PID);
	//printk("\n\n\n\n\n\n");
    // pid_t task_pid = task_pid_nr(task);
    // log_write("Process: %d\n", task_pid);
	// log_write("\thiwater_rss:%ld\n",task->mm->hiwater_rss);
	// pr_debug("\n\n\n\n\n\n");
    long int rss;
    if (task == NULL) {
        pr_alert("task is NULL\n");
        return 1;
    }
    if (task->mm == NULL) {
        pr_alert("mm is NULL\n");
        if (task->active_mm == NULL) {
            pr_alert("active_mm is NULL\n");
            return 1;
        }
	    rss = task->active_mm->hiwater_rss * 4;
        // return 1;
    }
    else {
	    rss = task->mm->hiwater_rss * 4;
    }
	log_write("\tvmrss:%ld KB\n", rss);
	return 0;
}

static long long int get_totalCpuTime(void)
{
    struct file *filp;
    char buf[256];
    ssize_t ret;
    
    // get kernel semaphore
    mmap_read_lock(current->mm);
    
    // open /proc/stat file
    filp = filp_open("/proc/stat", O_RDONLY, 0);
    
    if (IS_ERR(filp)) {
        printk(KERN_ERR "Failed to open /proc/stat file.\n");
        return -1;
    }
    
    // read /proc/stat file
    // ret = vfs_read(filp, buf, sizeof(buf), &filp->f_pos);
    ret = kernel_read(filp, buf, sizeof(buf), &filp->f_pos);
    
    if (ret < 0) {
        printk(KERN_ERR "Failed to read /proc/stat file.\n");
        filp_close(filp, NULL);
        return -1;
    }
    
    // print the content of /proc/stat file
    //printk(KERN_INFO "The content of /proc/stat file is:\n%s\n", buf);

    // split
    char *split_str, *cur = buf;
    char* const delim = " ";
    long long int totalCpuTime = 0, count = 0;
    while (split_str = strsep(&cur, delim)){
        //printk("%lld %s\n", count, split_str);
        if (count <= 10){ 
            // printk("%lld %s\n", count, split_str);
             totalCpuTime += simple_strtoll(split_str, NULL, 10);
            // printk("totalcputime:%lld\n", totalCpuTime);
             count ++;
        }
        else break;
    }
    
    // close /proc/stat file
    filp_close(filp, NULL);
    
    // release kernel semaphore
    mmap_read_unlock(current->mm);

    //printk(KERN_INFO "Module loaded.\n");
    return totalCpuTime;
}


static long long int get_proCpuTime(int pid)
{
	struct task_struct *task = pid_task(find_vpid(pid), PIDTYPE_PID);
	long long unsigned int proCpuTime = task->utime + task->stime;
	//printk("stime: %llu", task->utime);
	//printk("proCpuTime:%llu", proCpuTime);
	//change the ruler
	return proCpuTime / 10000000;
}	

long long int cal_cpu_use(int pid)
{
    long long int pcpu = 0;
    long long int pro1, total1, pro2, total2;
    // wait until there is a result
    // pcpu = 100*( threadCpuTime2 – threadCpuTime1) / (totalCpuTime2 – totalCpuTime1)
    // if multi-kernel, you have to time the kernel number

    int count = 1;
    while (pcpu <=0 || pcpu >= 100)
    {
        pro1 = get_proCpuTime(pid);
        total1 = get_totalCpuTime();
        // sleep  time slice = 10ms
        //udelay(10000);
	    schedule_timeout(100);

        pro2 = get_proCpuTime(pid);
        total2 = get_totalCpuTime();
	
	pr_debug("pro1:%lld", pro1);
	pr_debug("pro2:%lld", pro2);
	pr_debug("total1:%lld", total1);
	pr_debug("total2:%lld", total2);
        // printf("count %d\n", count);
        count ++;
        // if no changes, use = 0
        if (count == 100) {
            pcpu = 10000 * pro2 / total2;
	    pr_debug("\n");
            break;
        }
        if (total2 - total1 == 0) continue;
        pcpu = 100 * (pro2-pro1) / (total2 - total1);
	//pcpu = fp_mul(100,fp_div(fp_sub(pro2-pro1),fp_sub(total2-total1)))
    }
    //printk("process_cputime1:%lld\n", pro1);
    //printk("process_cputime2:%lld\n", pro2);
    //printk("total_cputime1:%lld\n", total1);
    //printk("total_cputime2:%lld\n", total2);
    // printk("cpu_use:%lld * 1e-4", pcpu);
    // log_write("for process: %d, cpu_use:%lld * 1e-4", pid, pcpu);
    log_write("for process: %d, cpu_use: %d.%d\%", pid, pcpu/10000, pcpu%10000);
    return pcpu;
}

// static int __init my_init(void)
// {
// 	//long long int totalCpuTime = get_totalCpuTime();
// 	//get_proCpuTime(1658);
// 	//long long int pcpu = cal_cpu_use(1658);
// 	//printk("pcpu:%lld",pcpu);
// 	//get_phy_mem(1746);
// 	//processFileInfo(1746);
// 	//get_sys_mem_info();
// 	//get_file_mes(1718);
// 	write_target();
// 	return 0;	


// }

// static void __exit my_exit(void)
// {
//  printk(KERN_INFO "Module unloaded.\n");
// }

// module_init(my_init);
// module_exit(my_exit);

MODULE_LICENSE("GPL");