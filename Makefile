all:
	clang++ -fno-rtti -fno-exceptions -O1 -g -o example example.cc
	#clang -O1 -g -o example-c example-c.c

# C11 version
example-c: csafetensors.c csafetensors.h
	$(CC) -std=c11 -O2 -Wall -Wextra -o $@ csafetensors.c -DCSAFETENSORS_EXAMPLE

# Unit tests for C11 version
test-c: csafetensors.c csafetensors.h test/unit/test_csafetensors.c
	$(CC) -std=c11 -O2 -Wall csafetensors.c test/unit/test_csafetensors.c -o test/unit/test_csafetensors -lm
	./test/unit/test_csafetensors

.PHONY: all test-c clean-c

clean-c:
	rm -f example-c test/unit/test_csafetensors
