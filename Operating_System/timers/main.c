#define STAT "/proc/net/dev"
#define MAX 255
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>
#include "bmon.h"
int main(void) {  // 主函数定义，无参数，返回一个整数
	int n_iface=number_of_interface(); // get number of interfaces  // 获取系统中网络接口的数量
	long long iface_odata_recv[n_iface];  // 接收数据量
	long long iface_odata_send[n_iface];  // 发送数据量
	for(int i=0 ; i<n_iface ; i++){  // 初始化数组 iface_odata_recv 和 iface_odata_send，将每个元素设置为 0
		iface_odata_recv[i]=0;
		iface_odata_send[i]=0;
		}
	iface_stat_t *iface_stat , *head;  // 定义一个指向结构体的指针 iface_stat 和一个指向结构体的指针 head
	head=NULL;  // 初始化 head 指针为空
	/*Initialize our linked list*/  // 初始化链表
	while(1>0){  // 无限循环
	for(int i=0;i<n_iface;i++){  // 遍历每个接口，为其创建一个新的结构体并将其加入链表中
		iface_stat=(iface_stat_t *)malloc(sizeof(iface_stat_t));
		iface_stat->next=head;
		head=iface_stat;
		}
	get_iface_name(iface_stat);  // 获取接口名称
	printf("Got %d interfaces\n",n_iface);  // 打印获取到的接口数量
	get_iface_data(iface_stat);  // 获取接口数据
	head=iface_stat;  // 将 head 指针重新指向链表的起始位置
	printf("\033[2J\033[1;1H"); // clear screen output  // 清空终端输出
	for(int i=0;i<n_iface;i++){  // 遍历每个接口，计算其速度并输出到终端
	// 如果接收数据量减去上一次的数据量为负数，则说明数据量已经被重置，将上一次的数据量更新为当前的数据量
		if((iface_stat->iface_data_recv - iface_odata_recv[i])<0){ 
			iface_odata_recv[i]=iface_stat->iface_data_recv;
			}
		if((iface_stat->iface_data_send - iface_odata_send[i])<0){  // 如果发送数据量减去上一次的数据量为负数，则说明数据量已经被重置，将上一次的数据量更新为当前的数据量
			iface_odata_send[i]=iface_stat->iface_data_send;
			}	
			
		// 输出接口名称、接收数据量、下行速度、发送数据量、上行速度
		printf("%s : %lld KB | %lld KBps | %lld KB | %lld KBps \n",iface_stat->iface_name,((iface_stat->iface_data_recv)/1024) , 
		((iface_stat->iface_data_recv - iface_odata_recv[i])/1024) , ((iface_stat->iface_data_send)/1024) ,
		(iface_stat->iface_data_send - iface_odata_send[i])/1024);
		// 更新上一次的数据量为当前的数据量
		iface_odata_recv[i]=iface_stat->iface_data_recv;
		iface_odata_send[i]=iface_stat->iface_data_send;
		iface_stat=iface_stat->next;// 指向下一个接口节点
	}
	
	// 释放链表内存
	iface_stat=head;
	iface_stat_t *tmp;
	tmp=iface_stat;
	for(int i=0;i<n_iface;i++){
		head=tmp->next;
		free(tmp);
		tmp=head;
		}
		
	sleep(1);
	
	}
	return 0;
	}
	
int number_of_interface(void) {  // 函数定义，无参数，返回一个整数
	FILE *statfp;  // 定义一个指向文件的指针
	statfp=fopen(STAT,"r");  // 打开文件，如果失败，则输出错误信息
	if(statfp==NULL){
		io_error();
		exit(1);
	}
	int n_iface=0;  // 定义一个整数变量 n_iface，表示接口的数量，初始化为 0
	char buffer[MAX];  // 定义一个字符数组 buffer，用于存储读取到的文件内容
	fgets(buffer,MAX,statfp); //读取文件的第一行和第二行，但不处理这些内容
	fgets(buffer,MAX,statfp); 
	while(fgets(buffer,MAX,statfp)){  // 循环读取文件的每一行，直到文件末尾
			n_iface++;  // 对于每一行，增加接口的数量
		}
	fclose(statfp);  // 关闭文件
	return(n_iface);  // 返回网络接口的数量
}

	
// 定义一个函数，获取网络接口的名称
void get_iface_name(iface_stat_t *iface_stat){
    // 打开文件指针，读取网络接口信息
    FILE *statfp;
    statfp=fopen(STAT,"r");
    if(statfp==NULL){
        // 如果打开文件失败，输出错误信息并退出
        io_error();
        exit(1);
    }
    // 定义一个缓存数组，读取文件内容
    char buffer[MAX];
    // 忽略前两行内容
    fgets(buffer,MAX,statfp);
    fgets(buffer,MAX,statfp);
    // 读取并解析文件中的每一行，获取网络接口名称
    while(fgets(buffer,MAX,statfp)){
        // 查找冒号的位置，以便确定网络接口名称的位置
        char *tmp;
        tmp = strstr(buffer , ":"); // 返回冒号的指针
        // 用于存储网络接口名称的辅助结构体
        index_t tmp_index;
        // 计算网络接口名称的位置
        calculate_iface_name(&tmp_index , buffer , tmp);
        // 将网络接口名称存储到结构体中
        strncpy(iface_stat->iface_name,buffer+tmp_index.l_index,(tmp_index.u_index-tmp_index.l_index));
        // 将结构体指针指向下一个节点
        iface_stat=iface_stat->next;
    }
    fclose(statfp);// 关闭文件指针
}

// 定义一个函数，获取网络接口的数据信息
void get_iface_data(iface_stat_t *iface_stat){

    // 打开文件指针，读取网络接口信息
    FILE *statfp;
    statfp=fopen(STAT,"r");
    if(statfp==NULL){
        // 如果打开文件失败，输出错误信息并退出
        io_error();
        exit(1);
    }
    // 定义一个缓存数组，用于读取文件内容
    char buffer[MAX];
    char iface_tmp_data[MAX];
    // 忽略前两行内容
    fgets(buffer,MAX,statfp);
    fgets(buffer,MAX,statfp);
    // 读取并解析文件中的每一行，获取网络接口的数据信息
    while(fgets(buffer,MAX,statfp)){
        // 查找冒号的位置，以便确定数据信息的位置
        char *tmp;
        tmp = strstr(buffer , ":");
        index_t tmp_index;
        // 计算数据信息的位置
        calculate_iface_name(&tmp_index , buffer , tmp);
        // 将辅助结构体中的起始位置设置为结束位置，用于计算接下来的数据信息位置
        tmp_index.l_index=tmp_index.u_index;
        // 计算接收数据信息的位置
        calculate_iface_data(&tmp_index , buffer , 1);
        // 从缓存数组中读取接收数据信息
        memset(iface_tmp_data , 0 , 255);
        strncpy(iface_tmp_data,buffer+tmp_index.l_index,(tmp_index.u_index-tmp_index.l_index));
        iface_stat->iface_data_recv=atoll(iface_tmp_data);
        // 计算发送数据信息的位置
        tmp_index.l_index=tmp_index.u_index;
        calculate_iface_data(&tmp_index , buffer , 8);
        // 从缓存数组中读取发送数据信息
        memset(iface_tmp_data , 0 , 255);
        strncpy(iface_tmp_data,buffer+tmp_index.l_index,(tmp_index.u_index-tmp_index.l_index));
        iface_stat->iface_data_send=atoll(iface_tmp_data);
        // 将结构体指针指向下一个节点
        iface_stat=iface_stat->next;
    }

    // 关闭文件指针
    fclose(statfp);
}
void io_error(void){
	// Something bad happened
	perror("Error opening stat file\n");
	}
void calculate_iface_name(index_t *tmp_index , char *buffer, char *tmp){ //计算接口名称函数, 接收一个结构体指针，一个字符数组指针和一个字符指针参数
	tmp_index->u_index=-1; //初始化结构体中的u_index成员为-1
	if(tmp) { //判断字符指针参数是否为空
		tmp_index->u_index=tmp-buffer; //如果不为空，计算出字符指针指向的字符在字符数组中的索引位置，并存储到结构体中的u_index成员中
	}
	else { //如果字符指针参数为空
		exit(1);
	}
	tmp_index->l_index=tmp_index->u_index; //将结构体中的u_index成员的值赋给l_index成员
	if(tmp_index->u_index >0){ //判断u_index成员的值是否大于0
		while( tmp_index->l_index >=0) { //如果大于0，则从u_index成员的值开始向前遍历字符数组，直到找到第一个空格字符
			if(buffer[tmp_index->l_index] != ' '){
				tmp_index->l_index--;
			}
			else {
				break;
			}
		}
		tmp_index->l_index++; //将找到的空格字符的下一个位置存储到结构体的l_index成员中
	}
}

void calculate_iface_data(index_t *tmp_index , char *buffer, int test_case){
	while(test_case > 0) { //循环遍历每个测试用例
		tmp_index->l_index=tmp_index->u_index; //将上一个测试用例的结束位置赋给l_index
		(tmp_index->l_index)++; //将l_index成员的值加1，指向当前测试用例的接口名称的第一个字符
		while( tmp_index->l_index >0) { //从l_index成员所指向的位置向前遍历字符数组，直到找到第一个非空格字符
			if(buffer[tmp_index->l_index] == ' '){
				(tmp_index->l_index)++;
			}
			else {
				break;
			}
		}
		tmp_index->u_index=tmp_index->l_index; //将找到的第一个非空格字符的位置存储到结构体的u_index成员中
		while(tmp_index->u_index>0){ //从u_index成员所指向的位置向后遍历字符数组，直到找到下一个空格字符
			if(buffer[tmp_index->u_index]!=' '){
				tmp_index->u_index++;
				}
			else {
				break;
				}
		}
		test_case--; //将测试用例数减1，继续处理下一个测试用例
	}
}