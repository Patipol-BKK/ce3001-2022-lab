#include <stdio.h>

__global__ void hello_GPU(int id) {
	int i = threadIdx.x;
	printf("Hello from GPU%d[%d]!\n", id, i);
}


int main(void) {
	printf("Hello from CPU!\n");
	hello_GPU<<<1,4>>>(1);
	hello_GPU<<<1,6>>>(2);
	cudaDeviceSynchronize();
	return 0;
}