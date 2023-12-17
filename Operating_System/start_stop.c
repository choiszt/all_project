/*
 * start_stop
 */
#include <linux/cdev.h>
#include <linux/fs.h>
#include <linux/file.h>
#include <linux/init.h>
#include <linux/ioctl.h>
#include <linux/module.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <linux/device.h>
#include <linux/pid.h>
#include <linux/pid_namespace.h>
#include "read.h"
#include "start_stop.h"
#include "check_inform.h"
// #include "relation.h"
#include "write_proc/writeproc.h"
#include "hash.h"
#include "parse2structs.h"

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Xuan Wang");
MODULE_DESCRIPTION("test ioctl to start and stop timer");

struct ioctl_arg
{
    unsigned int val;
};

#define IOC_MAGIC '\x66'

/* The ioctl command macro,
 * params:
 *      magic num: identify the device type, 0~255
 *      base num: identify the device subcommand
 *      variable type: the data type to exchange from user space to kernel
 */
#define IOCTL_START_CLOCK _IOR(IOC_MAGIC, 0, int)
#define IOCTL_STOP_CLOCK _IOR(IOC_MAGIC, 1, int)
#define IOCTL_RELATION _IOR(IOC_MAGIC, 2, int)
// #define IOCTL_STOP_CLOCK _IOR(IOC_MAGIC, 1, int)

#define IOCTL_VAL_MAXNR 4
#define DRIVER_NAME "test_device"
// #define TARGET_FILE "/home/away/develop/kernel/Merge/targets"
#define TARGET_FILE "/home/xav1er/targets"

static unsigned int test_ioctl_major = 0;
static unsigned int num_of_dev = 1;
static struct cdev test_ioctl_cdev;
static struct device *dev_struct;
static char device_path[100] = {0};

struct test_ioctl_data
{
    unsigned char val;
    rwlock_t lock;
};

static struct class *cls;
static int my_dev_uevent(struct device *dev, struct kobj_uevent_env *env)
{
    /* Change the permission of the device to 666
     * =========== appear to be useless =============
     */
    add_uevent_var(env, "DEVMODE=%#o", 0666);
    return 0;
}

extern int start_timer(struct parser_result *parser_result, struct parser_result_structs *structs_parser_result);

extern int stop_timer(void);

extern ssize_t read_file(const char *path, char *buf, int size, loff_t *pos);

extern struct parser_result parse_string(char *buf, ssize_t file_string_length);

extern void check_parser (struct parser_result res);

extern void edit_result (struct parser_result *ans);

// extern void check_relations (struct parser_result ans);
extern void process_relationtree(struct task_struct **task_list, int task_count);

extern struct task_struct **get_task_structs_by_pids(int *pid_list, int pid_count, int *valid_count);

extern void process_concurrent(struct task_struct **task_list, int task_count);



enum clock_state clock_state = STOP;

// The file parsed result
struct parser_result file_data;
// The file parsed result to struct
struct parser_result_structs struct_result;
/* Edited Xiao Zhang 23-04-24
 * Added a mark to show whether file_data has been initialized
 */
bool if_memset = false;

struct pid_namespace *ins_ns;

static long test_ioctl_ioctl(struct file *filp, unsigned int cmd,
                             unsigned long arg)
{
    /* @filp: File struct of the device file
     * @cmd: The command send to ioctl device
     * @arg: The given args from user
     */
    int retval = 0;
    int i;
    struct ioctl_arg data;
    memset(&data, 0, sizeof(data));
    if (!if_memset) {
        memset(&file_data, 0, sizeof(file_data));
        if_memset = true;
    }
    
    switch (cmd)
    {
    case IOCTL_START_CLOCK:
        pr_debug("IOCTL start clock.\n");
        if (clock_state == STOP)
        {

            // Read target file for preparation
            char buf[BUF_SIZE] = {0};
            loff_t pos = 0;
            // const char* target_file = (char *) arg;
            pr_debug("Reading targets file: %s", TARGET_FILE);
            ssize_t file_string_length = read_file(TARGET_FILE, buf, BUF_SIZE - 10, &pos);
            if (file_string_length < 0)
            {
                // pr_alert("Read file error.\n");
                return -1;
            }
            pr_debug("Read file success.\n");
            
            // Parse the file string
            file_data = parse_string(buf, file_string_length);
            struct_result = struct_parse_result(&file_data);
            // pr_alert("Show Relations.\n");
    	    // //edit ans, filter out invalid processes
    	    check_parser (file_data);
            // // redit parser result
            // edit_result (&file_data);
            // // print relation information
            // check_relations (file_data);
            
			// change the flag to START state
            // clock_state = START;
            start_timer(&file_data, &struct_result);
        }
        else
        {
            pr_alert("Clock is already running.\n");
        }
        goto done;
        
    case IOCTL_RELATION:
    	/* Edited Xiao Zhang 
         * Upload Relation Module.
         */
    	if (clock_state == START)
    	{
    	    pr_alert("IOCTL show relations.\n");
            //edit ans, filter out invalid processes
    	    // check_parser (file_data);
            // redit parser result
            // edit_result (&file_data);
            // print relation information
            struct task_struct **task_list;
            int valid_count = 0;
            task_list = get_task_structs_by_pids(file_data.pids, file_data.pids_len, &valid_count);
            printk(KERN_INFO "Print process tree...\n");
            process_relationtree(task_list, valid_count);
            printk(KERN_INFO "Print concurret relation...\n");
            process_concurrent(task_list, valid_count);
            log_write("File concurrency relations:\n");
            iterate_hash_table(1);
            // check_relations (file_data);
    	}
    	else 
    	{
    		pr_alert ("Clock is not running.\n");
    	}
    	goto done;

    case IOCTL_STOP_CLOCK:
        pr_alert("IOCTL stop clock.\n");
        if (clock_state == STOP)
        {
            pr_alert("Clock is already stopped.\n");
        }
        else
        {
            // change the flag to STOP state, used to stop the timer
            // clock_state = STOP;
            stop_timer();
        }
        goto done;

    default:
        retval = -ENOTTY;
    }

done:
    return retval;
}

// static ssize_t test_ioctl_read(struct file *filp, char __user *buf,
//                                size_t count, loff_t *f_pos)
//{
//     struct test_ioctl_data *ioctl_data = filp->private_data;
//     unsigned char val;
//     int retval;
//     int i = 0;

//    read_lock(&ioctl_data->lock);
//    val = ioctl_data->val;
//    read_unlock(&ioctl_data->lock);

//    for (; i < count; i++) {
//        if (copy_to_user(&buf[i], &val, 1)) {
//            retval = -EFAULT;
//            goto out;
//        }
//    }

//    retval = count;
// out:
//    return retval;
//}

static int test_ioctl_close(struct inode *inode, struct file *filp)
{
    pr_alert("%s call.\n", __func__);

    if (filp->private_data)
    {
        kfree(filp->private_data);
        filp->private_data = NULL;
    }

    return 0;
}

static int test_ioctl_open(struct inode *inode, struct file *filp)
{
    struct test_ioctl_data *ioctl_data;

    pr_alert("%s call.\n", __func__);
    ioctl_data = kmalloc(sizeof(struct test_ioctl_data), GFP_KERNEL);

    if (ioctl_data == NULL)
        return -ENOMEM;

    rwlock_init(&ioctl_data->lock);
    ioctl_data->val = 0xFF;
    filp->private_data = ioctl_data;

    return 0;
}

static struct file_operations fops = {
    .owner = THIS_MODULE,
    .open = test_ioctl_open,            // open device file
    .release = test_ioctl_close,        // close device file
    .unlocked_ioctl = test_ioctl_ioctl, // ioctl device file
};

static int ioctl_init(void)
{
    // get the pid namespace of the current process
    ins_ns = task_active_pid_ns(current);
    dev_t dev;
    int alloc_ret = -1;
    int cdev_ret = -1;
    pr_info("module init");
    // dynamic alloc character device
    alloc_ret = alloc_chrdev_region(&dev, 0, num_of_dev, DRIVER_NAME);

    if (alloc_ret)
        // exception handling
        goto error;

    // get major device number
    test_ioctl_major = MAJOR(dev);
    // init cdev struct with the file_operations struct
    cdev_init(&test_ioctl_cdev, &fops);
    // add a driver to kernel, register the driver device num
    cdev_ret = cdev_add(&test_ioctl_cdev, dev, num_of_dev);

    if (cdev_ret)
        goto error;

    cls = class_create(THIS_MODULE, DRIVER_NAME);
    if (IS_ERR(cls))
    {
        cdev_del(&test_ioctl_cdev);
        unregister_chrdev_region(test_ioctl_major, 1);
        printk(KERN_ALERT "Failed to create device class\n");
        return PTR_ERR(cls);
    }
    cls->dev_uevent = my_dev_uevent;
    // allocate memory for device struct
    dev_struct = kzalloc(sizeof(*dev_struct), GFP_KERNEL);
    dev_struct = device_create(cls, NULL, MKDEV(test_ioctl_major, 0), NULL, DRIVER_NAME);
    sprintf(device_path, "/dev/%s", DRIVER_NAME);
    struct file *dev_filp;
    dev_filp = filp_open(device_path, O_RDONLY, 0);
    if (IS_ERR(dev_filp))
    {
        pr_alert("Failed to open file\n");
        // goto error;
        // return PTR_ERR(dev_filp);
    }
    else
    {
        /* Set file permissions to read-write for user and group */
        dev_filp->f_path.dentry->d_inode->i_mode |= S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH;
        fput(dev_filp);
    }

    pr_alert("%s driver(major: %d) installed.\n", DRIVER_NAME,
             test_ioctl_major);
    return 0;
error:
    pr_alert("Error: %s driver(major: %d) installation failed.", DRIVER_NAME,
             test_ioctl_major);
    if (IS_ERR(dev_filp))
    {
        pr_alert("Failed to open device file\n");
    }
    if (cdev_ret == 0)
    {
        cdev_del(&test_ioctl_cdev);
        unregister_chrdev_region(dev, num_of_dev);
    }

    if (alloc_ret == 0)
        unregister_chrdev_region(dev, num_of_dev);
    return -1;
}

static void ioctl_exit(void)
{
    dev_t dev = MKDEV(test_ioctl_major, 0);

    cdev_del(&test_ioctl_cdev);
    device_destroy(cls, MKDEV(test_ioctl_major, 0));
    class_unregister(cls);
    class_destroy(cls);
    unregister_chrdev_region(dev, num_of_dev);
    // device_del(dev_struct);
    pr_alert("%s driver removed.\n", DRIVER_NAME);
}

module_init(ioctl_init);
module_exit(ioctl_exit);
