#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/kernel_stat.h>

void record_cpu_time(u64 *total_time, u64 *idle_time, int module_cpu_id) {
    struct kernel_cpustat cpu_stat;
    
    cpu_stat = kcpustat_cpu(module_cpu_id);
    *total_time = cpu_stat.cpustat[CPUTIME_USER] + cpu_stat.cpustat[CPUTIME_NICE] +
                    cpu_stat.cpustat[CPUTIME_SYSTEM] + cpu_stat.cpustat[CPUTIME_IDLE] +
                    cpu_stat.cpustat[CPUTIME_IOWAIT] + cpu_stat.cpustat[CPUTIME_IRQ] +
                    cpu_stat.cpustat[CPUTIME_SOFTIRQ] + cpu_stat.cpustat[CPUTIME_STEAL] +
                    cpu_stat.cpustat[CPUTIME_GUEST] + cpu_stat.cpustat[CPUTIME_GUEST_NICE];
    *idle_time = cpu_stat.cpustat[CPUTIME_IDLE];
}

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Zihong Zhang");
MODULE_DESCRIPTION("Record CPU time for calculating usage");