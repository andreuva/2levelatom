################################################################
#         MAKEFILE FOR THE INVERSION CODE WITH GPUS            #
#     AUTHOR: ANDRES VICENTE AREVALO    DATE: 16-3-2021        #
################################################################

CUR_DIR = $(shell pwd)
BIN_DIR = $(CUR_DIR)/bin
FIG_DIR = $(CUR_DIR)/figures

CC = gcc
NVCC = nvcc

CCFLAGS = -lstdc++ -lm -I /usr/local/cuda-11.2/include/.
NVCCFLAGS = -lm -L /usr/local/cuda-11.2/lib64/ -lcudart -lstdc++

ifeq ($(CUDA),1)
SRC_DIR = $(CUR_DIR)/src/cuda
SRC_CUDA = $(wildcard $(SRC_DIR)/*.cu)
CUDA_OBJECTS = $(patsubst $(SRC_DIR)/%.cu, $(BIN_DIR)/%.o, $(SRC_CUDA))
else

ifeq ($(JKQ),1)
SRC_DIR = $(CUR_DIR)/src/jkq
else
SRC_DIR = $(CUR_DIR)/src
endif

endif

SRC = $(wildcard $(SRC_DIR)/*.c)
OBJECTS = $(SRC:$(SRC_DIR)/%.c=$(BIN_DIR)/%.o)

HEADERS = $(wildcard $(SRC_DIR)/*.h)

ifeq ($(CUDA),1)

inversion_gpu : dirs $(CUDA_OBJECTS) $(OBJECTS)
	$(CC) -o inversion_gpu $(OBJECTS) $(CUDA_OBJECTS) $(NVCCFLAGS)

# Compilation comands
$(BIN_DIR)/%.o : $(SRC_DIR)/%.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BIN_DIR)/%.o : $(SRC_DIR)/%.c $(HEADERS)
	$(CC) $(CCFLAGS) -c $< -o $@

else
# Link comand
inversion : dirs $(OBJECTS)
	$(CC) -o inversion $(OBJECTS) $(CCFLAGS)

$(BIN_DIR)/%.o : $(SRC_DIR)/%.c $(HEADERS)
	$(CC) $(CCFLAGS) -c $< -o $@

endif

# Create build directory if it doesn't exists
.PHONY: dirs
dirs: ${BIN_DIR} ${FIG_DIR}
${BIN_DIR}:
	mkdir -p ${BIN_DIR}
${FIG_DIR}:
	mkdir -p ${FIG_DIR}

# Create clean comand to remove the last build
.PHONY : clean
clean :
	rm -f -r bin
	rm -f inversion*

