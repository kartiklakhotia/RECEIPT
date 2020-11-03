CC      = g++
CPPFLAGS= -O3 -c -std=c++14 -fopenmp -mavx -I./include/ 
LDFLAGS = -fopenmp -m64 -lpthread 
SOURCES = src/main.cpp 
SOURCES_SEQ = baselines/seq.cpp
OBJECTS = $(SOURCES:.cpp=.o)
OBJECTS_SEQ = $(SOURCES_SEQ:.cpp=.o)

all: decomposePar decomposeSeq

decomposePar : $(OBJECTS)  

	$(CC) $(OBJECTS) $(LDFLAGS) -o $@


decomposeSeq : $(OBJECTS_SEQ)  

	$(CC) $(OBJECTS_SEQ) $(LDFLAGS) -o $@


.cpp.o : 
	$(CC) $(CPPFLAGS) $< -o $@

clean:
	rm -f *.o ./src/*.o baselines/*.o decompose* dump*

