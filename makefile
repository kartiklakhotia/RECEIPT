CC      = g++
CPPFLAGS= -O3 -c -std=c++14 -fopenmp -mavx -I./include/ 
LDFLAGS = -fopenmp -m64 -lpthread 

SOURCES_TIP = src/tip.cpp 
SOURCES_SEQ_TIP = baselines/seq_tip.cpp

SOURCES_WING = src/wing.cpp
SOURCES_SEQ_WING = baselines/seq_wing.cpp
SOURCES_PC_WING = baselines/pc_wing.cpp

OBJECTS_TIP = $(SOURCES_TIP:.cpp=.o)
OBJECTS_SEQ_TIP = $(SOURCES_SEQ_TIP:.cpp=.o)

OBJECTS_WING = $(SOURCES_WING:.cpp=.o)
OBJECTS_SEQ_WING = $(SOURCES_SEQ_WING:.cpp=.o)
OBJECTS_PC_WING = $(SOURCES_PC_WING:.cpp=.o)

all: decomposeParTip decomposeSeqTip decomposeParWing decomposeSeqWing decomposePCWing

decomposeParTip : $(OBJECTS_TIP)  

	$(CC) $(OBJECTS_TIP) $(LDFLAGS) -o $@


decomposeSeqTip : $(OBJECTS_SEQ_TIP)  

	$(CC) $(OBJECTS_SEQ_TIP) $(LDFLAGS) -o $@




decomposeParWing : $(OBJECTS_WING)  

	$(CC) $(OBJECTS_WING) $(LDFLAGS) -o $@

decomposeSeqWing : $(OBJECTS_SEQ_WING)  

	$(CC) $(OBJECTS_SEQ_WING) $(LDFLAGS) -o $@

decomposePCWing : $(OBJECTS_PC_WING)  

	$(CC) $(OBJECTS_PC_WING) $(LDFLAGS) -o $@



.cpp.o : 
	$(CC) $(CPPFLAGS) $< -o $@

clean:
	rm -f *.o ./src/*.o baselines/*.o decompose* dump*

