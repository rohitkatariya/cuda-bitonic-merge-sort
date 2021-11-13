hellomake: sortcu.cu
	nvcc -c sortcu.cu -o sortcu.o --std=c++11
	ar cr libsortcu.a sortcu.o 

my:sortcu.cu a5.cpp
	nvcc -c sortcu.cu -o sortcu.o --std=c++11
	ar cr libsortcu.a sortcu.o 
	nvcc -c a5.cpp -o a5.o --std=c++11
	nvcc a5.o libsortcu.a -o a.out --std=c++11
	# ./a.out 121
	#rm -rf input_dir/*.txt output_dir/*.txt output_dir/*.csv output_dir/*.mpi
	# time mpirun -np 4 ./a.out
	#mpirun -np 6 ./a.out
	#cat output_dir/inp_0.txt output_dir/inp_1.txt output_dir/inp_2.txt output_dir/inp_3.txt
	#cat outputy_dir/out_0.txt output_dir/out_1.txt output_dir/out_2.txt output_dir/out_3.txt
gen:gen.cpp
	mpic++ gen.cpp -o gen.out
	mpirun -np 1 ./gen.out input_dir/inputfile20 20
	mpirun -np 1 ./gen.out input_dir/inputfile100 100
	mpirun -np 1 ./gen.out input_dir/inputfile1000 1000
	mpirun -np 1 ./gen.out input_dir/inputfile100000 100000
	mpirun -np 1 ./gen.out input_dir/inputfile10000000 10000000

read:psort.cpp reading.cpp
	mpic++ -c psort.cpp -o psort.o -fopenmp
	ar cr libpsort.a psort.o
	mpic++ -c reading.cpp -o reading.o -fopenmp
	mpic++ reading.o libpsort.a -o a.out -fopenmp
	# rm -rf input_dir/*.txt output_dir/*.txt output_dir/*.csv output_dir/*.mpi
	
