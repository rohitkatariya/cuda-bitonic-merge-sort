#!/bin/sh
### Set the job name (for your reference)
#PBS -N rohitcol730
### Set the project name, your department code by default
#PBS -P cse
### Request email when job begins and ends
#PBS -m bea
### Specify email address to use for notification.
#PBS -M csz208844@iitd.ac.in
####
###PBS -l select=16:ncpus=3:mpiprocs=1
#PBS -l  select=1:ncpus=1:ngpus=1
### Specify "wallclock time" required for this job, hhh:mm:ss
#PBS -l walltime=00:10:00
#PBS -P col730.csz208844  
#PBS -o jobOutput.txt
#PBS -l place=scatter
###PBS -l software=OpenMPI
# After job starts, must goto working directory. 
# $PBS_O_WORKDIR is the directory from where the job is fired. 
#echo "==============================="
echo $PBS_JOBID
echo "PBS_NTASKS:".$PBS_NTASKS
#cat $PBS_NODEFILE
#echo "==============================="
echo "omp num threads echo ".$OMP_NUM_THREADS
cd $PBS_O_WORKDIR
#job 
#time -p ls 
module load compiler/cuda/9.2/compilervars
#mpirun -n 4 a.out 
#time -p mpirun -np $PBS_NTASKS a.out /home/cse/phd/csz208844/file_transfer/chhavi/out_file> $PBS_JOBID
#export OMP_NUM_THREADS=4
TIMEFORMAT=%R
#time -p mpirun -np 3 ./a.out input_dir/inputfile100000 1> $PBS_JOBID
#make my
#time -p ./a.out 33554432 > $PBS_JOBID
#time -p ./a.out 67108864 > $PBS_JOBID
#time -p ./a.out 134217728 > $PBS_JOBID
time -p ./a.out 268435456 > $PBS_JOBID
#time -p ./a.out 536870912 > $PBS_JOBID
#time -p ./a.out 1073741824 > $PBS_JOBID
#time -p ./a.out 2147483648 > $PBS_JOBID
#time -p ./a.out 4294967296 > $PBS_JOBID


#NOTE
# The job line is an example : users need to change it to suit their applications
# The PBS select statement picks n nodes each having m free processors
# OpenMPI needs more options such as $PBS_NODEFILE
