obj-m += writeproc.o 
# test_write-objs := writeproc.o fortest.o
PWD := $(CURDIR) 

ccflags-y := -std=gnu99 -Wno-declaration-after-statement

all: 
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules
 
clean: 
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean
