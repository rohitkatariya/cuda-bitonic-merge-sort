hellomake: sortcu.cu
	nvcc -c sortcu.cu -o sortcu.o --std=c++11
	ar cr libsortcu.a sortcu.o 

my:sortcu.cu a5.cpp
	nvcc -c sortcu.cu -o sortcu.o --std=c++11
	ar cr libsortcu.a sortcu.o 
	nvcc -c a5.cpp -o a5.o --std=c++11
	nvcc a5.o libsortcu.a -o a.out --std=c++11

file:sortcu.cu a5.cpp
	nvcc -c sortcu.cu -o sortcu.o --std=c++11
	ar cr libsortcu.a sortcu.o 
	nvcc -c a5file.cpp -o a5file.o --std=c++11
	nvcc a5file.o libsortcu.a -o afile.out --std=c++11

gen:data_creator.cpp
	nvcc -c sortcu.cu -o sortcu.o --std=c++11
	ar cr libsortcu.a sortcu.o 
	nvcc -c data_creator.cpp -o data_creator.o --std=c++11
	nvcc data_creator.o libsortcu.a -o data_creator.out --std=c++11

read:psort.cpp reading.cpp
	mpic++ -c psort.cpp -o psort.o -fopenmp
	ar cr libpsort.a psort.o
	mpic++ -c reading.cpp -o reading.o -fopenmp
	mpic++ reading.o libpsort.a -o a.out -fopenmp
	# rm -rf input_dir/*.txt output_dir/*.txt output_dir/*.csv output_dir/*.mpi
	
