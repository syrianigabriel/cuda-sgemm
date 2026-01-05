NVCC = nvcc
CFLAGS = -std=c++17 -O2
INCLUDES = -Iinclude

SRC = src/main.cu src/sgemm.cu kernels/block_tiled_sgemm.cu kernels/register_tiled_sgemm.cu kernels/uncoalesced_naive_sgemm.cu kernels/coalesced_naive_sgemm.cu
OBJ = $(SRC:.cu=.o)
TARGET = sgemm

all: $(TARGET)

# Link step
$(TARGET): $(OBJ)
	$(NVCC) $(OBJ) -o $(TARGET)

# Compile each .cu into .o
%.o: %.cu
	$(NVCC) $(CFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)