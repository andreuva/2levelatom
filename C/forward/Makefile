################################################################
#         MAKEFILE FOR THE FORWARD SOLVER WITH GPUS            #
#     AUTHOR: ANDRES VICENTE AREVALO    DATE: 16-3-2021        #
################################################################

forward_solver : routines_cpu.o routines_gpu.o main.o
	gcc -o forward_solver routines_gpu.o routines_cpu.o main.o -lm -L /usr/local/cuda-11.2/lib64/ -lcudart -lstdc++
routines_cpu.o : routines_cpu.c params.h
	gcc -c routines_cpu.c -o routines_cpu.o -I /usr/local/cuda-11.2/include/. -lstdc++
main.o : main.c integratives.h params.h
	gcc -c main.c -o main.o -I /usr/local/cuda-11.2/include/. -lstdc++
routines_gpu.o : routines_gpu.cu cuda_extension.h params.h
	nvcc -c routines_gpu.cu -o routines_gpu.o

.PHONY : clean

clean :
	rm -f forward_solver routines_gpu.o routines_cpu.o main.o
