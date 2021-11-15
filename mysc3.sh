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
TIMEFORMAT=%R
#make my
for i in 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 ; do  time ./a.out $i;  done > myjobres3 2>&1
for i in 65536 131072 262144 524288 1048576 2097152 4194304 8388608 ; do  time ./a.out $i;  done > myjobres4 2>&1
for i in  16777216 33554432 67108864 134217728 268435456 536870912 1073741824 ; do  time ./a.out $i;  done > myjobres5 2>&1

#NOTE
# The job line is an example : users need to change it to suit their applications
# The PBS select statement picks n nodes each having m free processors
# OpenMPI needs more options such as $PBS_NODEFILE
