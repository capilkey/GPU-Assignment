# Makefile for a2
#
GCC_VERSION = 4.7.1
PREFIX = /usr/local/gcc/${GCC_VERSION}/bin/
CC = ${PREFIX}gcc
CPP = ${PREFIX}g++

all: a2

a2: main.o besselj.o utils.o
	${CPP} -pg -oa2  main.o besselj.o utils.o

main.o: main.cpp
	${CPP} -c -O2 -g -pg -std=c++0x main.cpp

program.o: program.cpp
	${CPP} -c -O2 -g -pg -std=c++0x besselj.cpp

utils.o: utils.cpp
	${CPP} -c -O2 -g -pg -std=c++0x utils.cpp

clean: rm -rf *o 
