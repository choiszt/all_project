#ifndef BMON_H_   // 防止头文件被重复引用
#define BMON_H_
// 定义一个结构体类型 iface_stat，用于存储接口的名称、接收和发送数据量，并且在链表中指向下一个结构体。
struct iface_stat{ 
    char iface_name[MAX];  // 存储接口名称
    long long iface_data_recv;  // 存储接口接收到的数据量
    long long iface_data_send;  // 存储接口发送的数据量
    struct iface_stat *next;  // 存储指向下一个结构体的指针
};
struct buffer_index{
	int l_index; //buffer的起始下标
	int u_index; //buffer的终止下标
	};
typedef struct buffer_index index_t;
typedef struct iface_stat iface_stat_t; 
void get_iface_name(iface_stat_t *iface_stat);//获取接口名称，并将其存储在结构体类型 iface_stat 中
int number_of_interface(void);  //获取系统接口的数量
void get_iface_data(iface_stat_t *iface_stat);// 获取接口的接收和发送数据量，并将其存储在结构体类型 iface_stat 中
void calculate_iface_name(index_t *tmp_index , char *buffer, char *tmp);// 计算接口名称在缓冲区中的下标
void calculate_iface_data(index_t *tmp_index , char *buffer, int test_case);//计算接口数据在缓冲区中的下标
void io_error(void);//用于输出错误信息

#endif