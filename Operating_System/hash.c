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

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Xiao Zhang");
MODULE_DESCRIPTION("A simple hash table example");

struct hlist_head my_hash_table[1 << HASH_TABLE_SIZE];

// turn file name to hash keys
static u32 hash_key(const char *key) {
    u32 hash;
    hash = full_name_hash(NULL, key, strlen(key));
    return hash;
}


// add data to hash table
void insert_data(const char *key, int *value, int count) {
    struct my_data *data = get_data(key);
    if (data) {
        // Key already exists, update the value and count.
        kfree(data->value);
        data->value = kmalloc(sizeof(int) * count, GFP_KERNEL);
        if (!data->value) {
            // If allocation fails, remove this data from hash table.
            hash_del(&data->node);
            kfree(data);
            return;
        }
        memcpy(data->value, value, sizeof(int) * count);
        data->count = count;
    } else {
        // Key does not exist, insert new data.
        data = kmalloc(sizeof(*data), GFP_KERNEL);
        if (!data)
            return;

        strncpy(data->key, key, sizeof(data->key));
        data->value = kmalloc(sizeof(int) * count, GFP_KERNEL);
        if (!data->value) {
            kfree(data);
            return;
        }
        memcpy(data->value, value, sizeof(int) * count);
        data->count = count;
        INIT_HLIST_NODE(&data->node);
        hash_add(my_hash_table, &data->node, hash_key(key));
    }
}


// get a data of hash table
struct my_data *get_data(const char *key) {
    struct my_data *data;
    hash_for_each_possible(my_hash_table, data, node, hash_key(key)) {
        if (strcmp(data->key, key) == 0)
            return data;
    }
    return NULL;
}

// remove a data from hash table
void remove_data(const char *key) {
    struct my_data *data;
    hash_for_each_possible(my_hash_table, data, node, hash_key(key)) {
        if (strcmp(data->key, key) == 0) {
            hash_del(&data->node);
            kfree(data->value);
            kfree(data);
            return;
        }
    }
}

// iterate the hash table and output data
void iterate_hash_table(int out_num) {
    struct my_data *data;
    int bkt, i;
    hash_for_each(my_hash_table, bkt, data, node) {
        if (data->count > out_num){
            log_write("\tfilename: %s,\n", data->key);
            log_write("\tconcurrented by tasks with pid:%s", " ");
            for (i = 0; i < data->count; i++) {
                log_write("%d", data->value[i]);
                if (i < data->count - 1){
                    log_write(", ");
                }
            }
            log_write("\n");
        }
        
    }
}

void test_hash(void){
    int values1[] = {1, 2, 3, 5, 6};
    int values2[] = {4, 5, 6};
    int values3[] = {7, 8, 9};

    // add data
    insert_data("one", values1, ARRAY_SIZE(values1));
    insert_data("two", values2, ARRAY_SIZE(values2));
    insert_data("three", values3, ARRAY_SIZE(values3));
    insert_data("four", values1, ARRAY_SIZE(values1));
    

    // get data 
    struct my_data *data = get_data("two");
    if (data) {
        int i;
        printk(KERN_INFO "found: key=%s, values= ", data->key);
        for (i = 0; i < data->count; i++) {
            printk(KERN_CONT "%d", data->value[i]);
            if (i < data->count - 1){
                printk(KERN_CONT ", ");
            }
        }
        printk(KERN_CONT "\n");
    }
    // iterate hash table
    printk(KERN_INFO "**** check hash table ****\n");
    iterate_hash_table(0);

    // remove a data
    printk(KERN_INFO "**** remove a data from hash table ****\n");
    remove_data("two");
    // edit a data
    insert_data("one", values3, ARRAY_SIZE(values3));

    printk(KERN_INFO "**** check hash table again ****\n");
    // check again
    iterate_hash_table(0);
}

