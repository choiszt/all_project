# OS_project

## 文件说明

* `process_file_info.c`  当前进程访问的文件信息
* `process_ip.c`  当前进程访问的ip信息（暂未测试正确）
* `prog2task.c`  通过进程名转化为 `task_struct` 
* `read_clock.c`  定时任务相关代码
* `read.c` 读取和解析targets文件
* `start_stop.c` 建立ioctl字符设备，接收用户指令相关代码
* `user/` 用户程序代码（给ioctl设备发信号）
* `record_cpu_time.c` 获取cpu监控相关信息


## 当前整合进度

* [x] `targets`文件信息读取
* [x] `targets`文件信息解析
* [x] 进程文件信息
* [x] 进程ip（测试暂不正确）
* [x] prog文件信息
* [x] 进程树	user.c r
* [x] 定时读取给定进程信息
* [x] 用户开启或关闭定时
* [x] ioctl设备相关
* [x] 模块CPU统计（等待验证）
* [x] 模块内存使用统计
* [x] 文件的相关信息（等待验证）
* [ ] 进程内存和CPU统计

## 有关输出模块
将刘帅编写的输出到proc文件改成了一个单独的模块，在`/write_proc`中，需要单独使用make编译并插入模块，这个模块会导出符号 `log_write`，其他地方要使用该函数输出日志，需要include头文件 `write_proc/writeproc.h`，并且修改了makefile，添加了 `KBUILD_EXTRA_SYMBOLS` 路径，使外部模块能够获得到导出的符号。

使用案例：
首先在write_proc文件夹下make并ins

```bash
cd write_proc
make
sudo insmod writeproc.ko
```

接着在项目根目录的模块中include头文件，并使用log_write函数输出日志

```c
#include <linux/init.h>
#include <linux/module.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>
#include <linux/sched.h>
#include "write_proc/writeproc.h"

MODULE_LICENSE("GPL");

static int __init test_log_write(void)
{
    log_write("This is a test message.\n");
    // printk(KERN_INFO "log_buffer: %s", log_buffer);

    return 0;
}

static void __exit test_log_write_exit(void)
{
    printk(KERN_INFO "test_log_write module unloaded\n");
}

module_init(test_log_write);
module_exit(test_log_write_exit);
```

在这之后可以去查看 `/proc/kernellog` 文件，里面就有刚才输出的日志了。

```bash
$ cat /proc/kernellog
This is a test message.
```