CUDA_INSTALL_PATH ?= /usr/local/cuda
VER =

CXX := /usr/bin/g++$(VER)
CC := /usr/bin/gcc$(VER)
LINK := /usr/bin/g++$(VER) -fPIC
CCPATH := ./gcc$(VER)
NVCC  := $(CUDA_INSTALL_PATH)/bin/nvcc -ccbin $(CCPATH)

# Includes
INCLUDES = -I. -I$(CUDA_INSTALL_PATH)/include

# Libraries
LIBS := -lcuda -lpthread -lX11


# Options
NVCCOPTIONS = -arch sm_20 -ptx

# Common flags
COMMONFLAGS += $(INCLUDES)
NVCCFLAGS += $(COMMONFLAGS) $(NVCCOPTIONS)
CXXFLAGS += $(COMMONFLAGS)
CFLAGS += $(COMMONFLAGS)



CUDA_OBJS = filter-noise.ptx 
OBJS = demo.cpp.o filter-noise.cpp.o
TARGET = solution.x
LINKLINE = $(LINK) -o $(TARGET) $(OBJS) $(LIBS)

.SUFFIXES:	.c	.cpp	.cu	.o	
%.c.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.ptx: %.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

%.cpp.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): prepare $(OBJS) $(CUDA_OBJS)
	$(LINKLINE)

clean:
	rm -rf $(TARGET) *.o *.ptx

prepare:
	rm -rf $(CCPATH);\
	mkdir -p $(CCPATH);\
	ln -s $(CXX) $(CCPATH)/g++;\
	ln -s $(CC) $(CCPATH)/gcc;

