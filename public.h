#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

#define CHECK(call)                                   \
do {                                                  \
    const cudaError_t error = call;                   \
    if (error != cudaSuccess) {                       \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason:%s\n", error,         \
                cudaGetErrorString(error));           \
        exit(1);                                      \
    }                                                 \
} while(0)