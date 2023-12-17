struct parser_result_structs {
    struct task_struct **tasks;
    int tasks_len;
    struct inode **inodes;
    int inodes_len;
};
struct parser_result_structs struct_parse_result(struct parser_result *res);