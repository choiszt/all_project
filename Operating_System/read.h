#ifndef READ_H
#define READ_H
struct parser_result
{
    int *pids;
    int pids_len;  // need?
    char **progs;
    int progs_len;
    char **files;
    int files_len;
};

#define BUF_SIZE 1024
#define MAX_SIZE 1024
#define PROCESS_NUM 20

#endif