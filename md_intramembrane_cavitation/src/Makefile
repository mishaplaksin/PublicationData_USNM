CC      = gcc
CFLAGS  = -O3 -march=native -ffast-math -fopenmp -Wall -Wno-unused-variable
LDFLAGS = -lm -fopenmp

all: cgmd

cgmd: cgmd.c
	$(CC) $(CFLAGS) -o cgmd cgmd.c $(LDFLAGS)

clean:
	rm -f cgmd
