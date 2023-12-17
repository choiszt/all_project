#ifndef _KERNELLOG_H
#define _KERNELLOG_H

#include <linux/seq_file.h>

// buffer，记录 log 信息
extern char log_buffer[1024];
extern int log_buffer_pos;

// log_write
void log_write(const char *fmt, ...);
int log_show(struct seq_file *m, void *v);
int log_open(struct inode *inode, struct file *file);
extern const struct file_operations log_fops;
int __init kernellog_init(void);
void __exit kernellog_exit(void);

#endif // _KERNELLOG_H