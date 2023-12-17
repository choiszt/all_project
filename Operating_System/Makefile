obj-m += switch.o
switch-objs := process_file_info.o read_clock.o read.o prog2task.o start_stop.o check_inform.o relation12.o record_cpu_time.o process_ip.o file_process_info.o proc_cpu_usage.o hash.o parse2structs.o
 
PWD := $(CURDIR) 

KBUILD_EXTRA_SYMBOLS += $(CURDIR)/write_proc/Module.symvers

export KBUILD_EXTRA_SYMBOLS

ccflags-y := -std=gnu99 -Wno-declaration-after-statement

all: 
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules
 
clean: 
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean
