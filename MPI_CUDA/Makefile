#ssh -i cmc-supercomp-private edu-cmc-skmodel24-622-11@polus.hpc.cs.msu.ru
#mpirun -n 1 gdb ./cuda 101 1
#mpirun -n 1 cuda-gdb ./cuda 100 1

#mpirun -n 1 cuda-gdb -x commands ./cuda


sources = Sequence.h Array3D.h main.cpp  Array3D.cpp Sequence.cu utils.cpp

.PHONY: cuda

all: cuda 


gdb: $(sources)
	nvcc  -fmad=false -std=c++11 -O0 -std=c++11 -g -G -I/opt/ibm/spectrum_mpi/include -L/opt/ibm/spectrum_mpi/lib -lmpi_ibm -lmpiprofilesupport -Xcompiler -fopenmp   Sequence.cu utils.cpp main.cpp Array3D.cpp -o cuda 


ptx: $(sources)
	nvcc -ptx -std=c++11 Sequence.cu utils.cpp main.cpp 
	ptxas -v *.ptx

cuda: 
	nvcc  -fmad=false -std=c++11 -O3 -std=c++11  -I/opt/ibm/spectrum_mpi/include -L/opt/ibm/spectrum_mpi/lib -lmpi_ibm -lmpiprofilesupport -Xcompiler -fopenmp   Sequence.cu utils.cpp main.cpp -o cuda
	mpirun -n 1 ./cuda 128 1

polus:
	scp -i cmc-supercomp-private  commands *.cpp *.h  *.cu Makefile edu-cmc-skmodel24-622-11@polus.hpc.cs.msu.ru:~/MPI_CUDA
	



makefile:
	scp -i cmc-supercomp-private  Makefile edu-cmc-skmodel24-622-11@polus.hpc.cs.msu.ru:~/MPI_CUDA

clean:
	rm -rf *

send_commands:  commands
	scp -i cmc-supercomp-private  commands edu-cmc-skmodel24-622-11@polus.hpc.cs.msu.ru:~/MPI_CUDA



#scp -r -i cmc-supercomp-private edu-cmc-skmodel24-622-11@polus.hpc.cs.msu.ru:~/SuperPc-OpenMp-MPI .
#scp -r  -i cmc-supercomp-private ./SuperPc-OpenMp-MPI edu-cmc-skmodel24-622-11@polus.hpc.cs.msu.ru:~/ 
#mpicxx -fopenmp -O3 -std=c++11 -o prog Sequence.cpp utils.cpp main.cpp


#OMP_NUM_THREADS=32 bsub -x -n 2  -R "span[hosts=1] affinity[core(40)]" -W 00:05 -q normal -oo "out.txt" -eo "err.txt"  OMP_NUM_THREADS=32 mpiexec ./prog 512 1
#OMP_NUM_THREADS=80 bsub -x -n 2  -R "span[hosts=1] affinity[core(40)]" -W 00:05 -q normal -oo "out.txt" -eo "err.txt"  OMP_NUM_THREADS=80 mpiexec ./prog 512 1
#bsub -x -n 1 -gpu "num=1:mode=exclusive_process" -R "span[hosts=1]" -W 00:05 -q normal -o "cuda_out.txt" -e "cuda_err.txt"  mpiexec ./cuda 512 1
#bsub -n 1 -gpu "num=1:mode=exclusive_process" -R "span[hosts=1]" -W 00:05 -q normal -o "cuda_out.txt" -e "cuda_err.txt"  mpiexec ./cuda 512 1

#OMP_NUM_THREADS=4 bsub -n 8  -R "span[ptile=1] affinity[core(4)]" -W 00:15 -q normal -o "out.txt" -e "err.txt"  OMP_NUM_THREADS=4 mpiexec ./prog 512 1