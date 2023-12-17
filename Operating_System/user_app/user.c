#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include "../start_stop.h"

#define DEVICE "/dev/test_device"
#define IOC_MAGIC '\x66'

// #define IOCTL_VALSET _IOW(IOC_MAGIC, 0, struct ioctl_arg)
// #define IOCTL_VALGET _IOR(IOC_MAGIC, 1, struct ioctl_arg)
#define IOCTL_START_CLOCK _IOR(IOC_MAGIC, 0, int)
#define IOCTL_STOP_CLOCK _IOR(IOC_MAGIC, 1, int)
#define IOCTL_RELATION _IOR(IOC_MAGIC, 2, int)


int main(int argc, char **argv)
{
    int fd;
    char ch, write_buf[100], read_buf[100];

    fd = open(DEVICE, O_RDONLY); // open for reading and writing

    if (fd == -1) {
        printf("File %s either does not exist or has been locked by another process\n", DEVICE);
        exit(-1);
    }

    // printf("r = read from device\nw = write to device\nEnter command: ");
    // scanf("%c", &ch);

    // switch (ch) {
    //     case 'w':
    //         printf("Enter data: ");
    //         scanf(" %[^\n]", write_buf);
    //         write(fd, write_buf, sizeof(write_buf));
    //         break;
    //     case 'r':
    //         read(fd, read_buf, sizeof(read_buf));
    //         printf("Device: %s\n", read_buf);
    //         break;
    //     default:
    //         printf("Command not recognized\n");
    //         break;
    // }
    if (argc == 2) {
        if (argv[1][0] == 's') {
            printf("Sending start command to device.\n");
            ioctl(fd, IOCTL_START_CLOCK, START);
        }
        else if (argv[1][0] == 't') {
            printf("Sending stop command to device.\n");
            ioctl(fd, IOCTL_STOP_CLOCK, STOP);
        }
        else if (argv[1][0] == 'r') {
            printf("Show relations between processes.\n");
            ioctl(fd, IOCTL_RELATION, START);
        }
        else {
            printf("Command not recognized\n");
        }
    }
    else {
        printf("Command not recognized\n");
    }

    close(fd);

    return 0;
}
