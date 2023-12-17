# compile user user.c

CC = gcc
CFLAGS = -Wall -g
# LDFLAGS = -lpthread

all: user

user: user.o
	$(CC) $(CFLAGS) -o user.out user.o

user.o: user.c
	$(CC) $(CFLAGS) -c user.c

clean:
	rm -f *.o user.out