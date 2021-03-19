################################################################
#         MAKEFILE FOR THE FORWARD SOLVER WITH GPUS            #
#     AUTHOR: ANDRES VICENTE AREVALO    DATE: 16-3-2021        #
################################################################

inversion : forward_solver.o inversion_GD.o integrals_quadratures.o inversion_routines.o # routines_gpu.o
	# gcc -o inversion routines_gpu.o forward_solver.o integrals_quadratures.o inversion_GD.o inversion_routines.o -lm -L /usr/local/cuda-11.2/lib64/ -lcudart -lstdc++
	gcc -o inversion forward_solver.o integrals_quadratures.o inversion_GD.o inversion_routines.o -lm -L /usr/local/cuda-11.2/lib64/ -lstdc++

integrals_quadratures.o : integrals_quadratures.c includes_definitions.h
	gcc -c integrals_quadratures.c -o integrals_quadratures.o -I /usr/local/cuda-11.2/include/. -lstdc++

inversion_GD.o : inversion_GD.c params.h includes_definitions.h
	gcc -c inversion_GD.c -o inversion_GD.o -I /usr/local/cuda-11.2/include/. -lstdc++

forward_solver.o : forward_solver.c params.h includes_definitions.h
	gcc -c forward_solver.c -o forward_solver.o -I /usr/local/cuda-11.2/include/. -lstdc++

inversion_routines.o : inversion_routines.c params.h includes_definitions.h
	gcc -c inversion_routines.c -o inversion_routines.o -I /usr/local/cuda-11.2/include/. -lstdc++

routines_gpu.o : routines_gpu.cu cuda_extension.h params.h includes_definitions.h
	nvcc -c routines_gpu.cu -o routines_gpu.o

.PHONY : clean

clean :
	rm -f inversion forward_solver.o inversion_GD.o routines_gpu.o integrals_quadratures.o inversion_routines.o
