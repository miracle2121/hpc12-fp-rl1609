EXECUTABLE = clfft
CC = gcc
CFLAGS = -std=c99 -lrt -D_XOPEN_SOURCE=500 -lOpenCL -lm

all: $(EXECUTABLE)

clfft: main.c cl-helper.o ppm.o clfft.c
	$(CC) $(CFLAGS) -o$@ $^

%.o: %.c %.h
	$(CC) -c -std=c99 $<

clean:
	rm -f $(EXECUTABLE) *.o
