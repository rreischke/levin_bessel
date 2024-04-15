#CC	= g++

SRC_PATH = ./src


LDFLAGS = -shared
CXXFLAGS = -O3 -pedantic -Wall -fPIC -m64 -Xpreprocessor -fopenmp -std=c++14 -I$(SRC_PATH)

LIBDIR	= ./lib
INCDIR	= ./include

STATIC	= liblevin.a


classes = \
	$(SRC_PATH)/levin_power.cpp
	
objects	= $(classes:.cpp=.o)
headers	= $(classes:.cpp=.h)

$(STATIC): $(objects)
	ar rv $(STATIC) $(objects)

$(SRC_PATH)/%.o: %.cpp %.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

.PHONY: clean distclean doc
.DEFAULT_GOAL: static

static: $(STATIC)

doc:
	doxygen

clean:
	$(RM) *.o *~
distclean: clean
	$(RM) Makefile $(STATIC)

install: $(STATIC)
	mv $(STATIC) $(LIBDIR); \
	$(RM) -r doc
