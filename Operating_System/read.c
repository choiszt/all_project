#include <linux/init.h>
#include <linux/module.h>
#include <linux/string.h>
#include <linux/slab.h>
#include <linux/kernel.h>
#include <linux/fs.h>
#include "read.h"

ssize_t read_file(const char* path, char* buf, int size, loff_t* pos)
{/*
 * @path: path to the file to read
 * @buf: the buffer used to store the content of the file
 * @size: the size to read from the file
 * @pos: the position to start reading file
 */
    struct file *file;
    ssize_t bytes;

    file = filp_open(path, O_RDONLY, 0);
    if (IS_ERR(file)) {
        long err_code = PTR_ERR(file);
        pr_err("Failed to open file. Error code: %ld\n", err_code);
        return -1;
    }

    if (size < MAX_SIZE && (bytes = kernel_read(file, buf, size, pos)) > 0) {
        pr_debug("Read %ld bytes from file: %s\n", bytes, buf);
    }

    if (bytes < 0) {
        pr_err("Failed to read file\n");
        pr_err("bytes value: %d\n", bytes);
        filp_close(file, NULL);
        return -1;
    }

    filp_close(file, NULL);
    return bytes;
}




struct parser_result parse_string(char *buf, ssize_t file_string_length) {
/*
 * @buf: file chars to be parse
 * @file_string_length: length of buf
 */
    const char pid_string[] = "pid:";
    const char prog_string[] = "prog:";
    const char file_string[] = "file:";

    // int pids[PROCESS_NUM] = {0};
    /* modified by X.Wang 2023-3-26
     * use kzalloc to allocate memory for pids, or the data would be wrong when returned
     */
    int *pids = (int*)kzalloc(PROCESS_NUM * sizeof(int), __GFP_ZERO);


    char **progs, **files;
    progs = (char**)kzalloc(PROCESS_NUM * sizeof(char*), __GFP_ZERO);
    int i = 0;
    for (;i < PROCESS_NUM; i++) {
        progs[i] = (char*)kzalloc(BUF_SIZE * sizeof(char), __GFP_ZERO);
    }

    files = (char**)kzalloc(PROCESS_NUM * sizeof(char*), __GFP_ZERO);
    i = 0;
    for (;i < PROCESS_NUM; i++) {
        files[i] = (char*)kzalloc(BUF_SIZE * sizeof(char), __GFP_ZERO);
    }

    int pids_len = 0, progs_len = 0, files_len = 0;
    if (strstr(buf, pid_string) != NULL) {
        int pid_index = strstr(buf, pid_string) - buf;
        pid_index += strlen(pid_string);  // goto first digit
        // pr_info("find pid at %ld", pid_index);
        char identify_str[BUF_SIZE] = {0};  // identify str now
        int identify_index = 0;
        while (buf[pid_index] != '\n') {
            while (buf[pid_index] == ' ') {  // strip space
                pid_index++;
            }
            while (buf[pid_index] != ',' && buf[pid_index] != '\n') {
                identify_str[identify_index++] = buf[pid_index++];
            }
            if (buf[pid_index] == ',')
                pid_index++;

            // append identify_str to pids
            int pid = simple_strtol(identify_str, NULL, 10);
            memset(identify_str, 0, sizeof(identify_str));
            identify_index = 0;
            // pr_info("pid now: %d", pid);
            pids[pids_len++] = pid;
            // pr_info("pids_len now: %d", pids_len);
        }
    }
    if (strstr(buf, prog_string) != NULL) {
        int prog_index = strstr(buf, prog_string) - buf;
        prog_index += strlen(prog_string);  // goto first digit
        // pr_info("find prog at %ld", prog_index);
        char identify_str[BUF_SIZE] = {0};  // identify str now
        int identify_index = 0;
        while (prog_index < file_string_length && buf[prog_index] != '\n') {
            while (buf[prog_index] == ' ') {  // strip space
                prog_index++;
            }
            while (prog_index < file_string_length && buf[prog_index] != ',' && buf[prog_index] != '\n') {
                identify_str[identify_index++] = buf[prog_index++];
            }
            if (prog_index < file_string_length && buf[prog_index] == ',')
                prog_index++;

            // append identify_str to pids
            strcpy(progs[progs_len++], identify_str);
            memset(identify_str, 0, sizeof(identify_str));
            identify_index = 0;
        }
    }
    if (strstr(buf, file_string) != NULL) {
        int file_index = strstr(buf, file_string) - buf;
        file_index += strlen(file_string);  // goto first digit
        char identify_str[BUF_SIZE] = {0};  // identify str now
        int identify_index = 0;
        while (file_index < file_string_length && buf[file_index] != '\n') {
            while (buf[file_index] == ' ') {  // strip space
                file_index++;
            }
            while (file_index < file_string_length && buf[file_index] != ',' && buf[file_index] != '\n') {
                identify_str[identify_index++] = buf[file_index++];
            }
            if (file_index < file_string_length && buf[file_index] == ',')
                file_index++;

            // append identify_str to pids
            strcpy(files[files_len++], identify_str);
            memset(identify_str, 0, sizeof(identify_str));
            identify_index = 0;
        }
    }
    struct parser_result res;
    res.pids = pids;
    res.pids_len = pids_len;
    res.progs = progs;
    res.progs_len = progs_len;
    res.files = files;
    res.files_len = files_len;

    // i = 0;
    // for (; i < res.pids_len; i++) {
    //     pr_info("pids: %ld", res.pids[i]);
    // }
    // i = 0;
    // for (; i < res.progs_len; i++) {
    //     pr_info("progs: %s", res.progs[i]);
    // }
    // i = 0;
    // for (; i < res.files_len; i++) {
    //     pr_info("pids: %s", res.files[i]);
    // }

    return res;
}

// static int __init test_read_init(void)
// {
//     pr_info("***********Testing read function***********\n");
//     const char* file_path = "/home/sarlren/os_design/file.txt";
//     char buf[BUF_SIZE]={0};
//     ssize_t file_string_length = read_file(file_path, buf, 200, 0);
//     pr_info("file content: %s", buf);
//     pr_info("file length: %ld", file_string_length);
//     parse_string(buf, file_string_length);
//     return 0;
// }

// static void __exit test_exit(void)
// {
//     pr_info("***********Removing test module**************\n");
// }

// module_init(test_read_init);
// module_exit(test_exit);


MODULE_LICENSE("GPL");
MODULE_AUTHOR("Zihong Zhang");
MODULE_DESCRIPTION("Used to read file");