################################################################
#         MAKEFILE FOR THE FORWARD SOLVER WITH GPUS            #
#     AUTHOR: ANDRES VICENTE AREVALO    DATE: 16-3-2021        #
################################################################

ifeq ($(CUDA),1)
inversion_gpu : forward_solver_gpu.o inversion_GD.o integrals_quadratures.o inversion_routines.o
	gcc -o inversion_gpu forward_solver_gpu.o integrals_quadratures.o inversion_GD.o inversion_routines.o -lm -L /usr/local/cuda-11.2/lib64/ -lcudart -lstdc++

forward_solver_gpu.o : forward_solver_gpu.cu cuda_extension.h params.h includes_definitions.h
	nvcc -c forward_solver_gpu.cu -o forward_solver_gpu.o
else
inversion : forward_solver.o inversion_GD.o integrals_quadratures.o inversion_routines.o
	gcc -o inversion forward_solver.o integrals_quadratures.o inversion_GD.o inversion_routines.o -lm -L /usr/local/cuda-11.2/lib64/ -lstdc++

forward_solver.o : forward_solver.c params.h includes_definitions.h
	gcc -c forward_solver.c -o forward_solver.o -I /usr/local/cuda-11.2/include/. -lstdc++
endif

integrals_quadratures.o : integrals_quadratures.c includes_definitions.h
	gcc -c integrals_quadratures.c -o integrals_quadratures.o -I /usr/local/cuda-11.2/include/. -lstdc++

inversion_GD.o : inversion_GD.c params.h includes_definitions.h
	gcc -c inversion_GD.c -o inversion_GD.o -I /usr/local/cuda-11.2/include/. -lstdc++

inversion_routines.o : inversion_routines.c params.h includes_definitions.h
	gcc -c inversion_routines.c -o inversion_routines.o -I /usr/local/cuda-11.2/include/. -lstdc++

.PHONY : clean

clean :
	rm -f inversion forward_solver.o integrals_quadratures.o inversion_GD.o inversion_routines.o
	rm -f inversion_gpu forward_solver_gpu.o integrals_quadratures.o inversion_GD.o inversion_routines.o
