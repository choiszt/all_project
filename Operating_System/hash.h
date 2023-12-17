#ifndef __HASH_H__
#define __HASH_H__

#define FILENAME_LEN 128 // max length of a filename
#define HASH_TABLE_SIZE 8
#define MAX_PID_NUM 20
struct my_data {
    char key[FILENAME_LEN]; // filename
    int *value; // value: pid of processes concurrenting file
    int count; // count: number of pids
    struct hlist_node node;
};

// initializ hashtable
extern struct hlist_head my_hash_table[1 << HASH_TABLE_SIZE];

// add data to hash table
void insert_data(const char *key, int *value, int count);

// get a data of hash table
struct my_data *get_data(const char *key);

// remove a data from hash table
void remove_data(const char *key);

// iterate the hash table and output data
void iterate_hash_table(int out_num);

#endif
