CXXFLAGS=-g
CFLAGS=-g 
CC=nvcc
CXX=nvcc



programs := hello-world threads-2d sum-array mmpcuda


all: $(programs)

hello-world: hello-world.cu
	$(CXX) -o $@ $(CXXFLAGS) $<

threads-2d: threads-2d.cu
	$(CXX) -o $@ $(CXXFLAGS) $<

sum-array: sum-array.cu timing.o
	$(CXX) -o $@ $(CXXFLAGS) $^

mmpcuda: mmpcuda.o timing.o
	$(CXX) -o $@ $(CXXFLAGS) $^

%.o: %.c
	$(CXX) -o $@ $(CXXFLAGS) -c $^
%.o: %.cu
	$(CXX) -o $@ $(CXXFLAGS) -c $^

clean:
	-rm -f $(programs) *.o
