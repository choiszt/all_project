obj-m += jiffies.o 
switch-objs := jiffies_up.o mod_timer.o jiffies_relative.o jiffies_relative_up.o setup_timer.o onstack.o time_addns.o time_equal.o timecompare.o task_pid.o writeproc.o fortest.o
PWD := $(CURDIR) 
all: 
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules
	gcc -g -W -Wall -std=c99 -o bmon main.c
clean: 
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean
