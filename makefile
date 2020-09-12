CC      = g++
CPPFLAGS= -O3 -c -std=c++14 -fopenmp -mavx -I./include/ 
LDFLAGS = -fopenmp -m64 -lpthread 
SOURCES = src/main.cpp 
OBJECTS = $(SOURCES:.cpp=.o)

all: $(SOURCES) decompose

decompose : $(OBJECTS)  

	$(CC) $(OBJECTS) $(LDFLAGS) -o $@

.cpp.o : 
	$(CC) $(CPPFLAGS) $< -o $@

clean:
	rm -f *.o ./src/*.o decompose dump*

