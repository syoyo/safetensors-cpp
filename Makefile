all:
	clang++ -fno-rtti -fno-exceptions -O1 -g -o example example.cc
	#clang -O1 -g -o example-c example-c.c
