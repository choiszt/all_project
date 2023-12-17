#ifndef __RELATION_H__
#define __RELATION_H__

#define MAX_NAME_LENGTH 256 // max length of file
#define MAX_LIST_SIZE 256 // max number of file accessed by a process
#define MAX_PID_NUM 10 // max number of process in the list

/* 2023-4-8 edited by ZhangXiao
 * build a hash map judging whether the file is visited by multiple processes
 */

#define HASH_TABLE_SIZE 1024

// struct to store the information about a process and the files it accessed
struct process_file_info {
    char *process_name;  // name of process
    int pid; // pid of process
    int file_count; // the num of file visited by process
    char **file_list; // list of file names
};

// struct to store information about a file and its corresponding processes
struct file_info {
    char *filename;  // name of the file
    int pid_count;  // number of processes that have accessed the file
    int *pid_list;  // list of pids that have accessed the file
};

void check_relations (struct parser_result ans);
#endif