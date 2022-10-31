#include <stdio.h>

__global__ void cudaVectorDotProduct(int *vecC, int *vecA, int *vecB, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int i;
	vecC[idx] = vecA[idx] * vecB[idx];
	__syncthreads();
	if(!idx) {
		for(i = 1; i < n; i++) {
			vecC[idx] += vecC[i];
		}
	}
}

int main() {
	int n = 4;
	int vecA[n] = {22, 13, 16, 5};
	int vecB[n] = {5, 22, 17, 37};
	int vecC[n];

	int *dVecA, *dVecB, *dVecC;
	cudaMalloc((void**)&dVecA, sizeof(int)*n);
	cudaMalloc((void**)&dVecB, sizeof(int)*n);
	cudaMalloc((void**)&dVecC, sizeof(int)*n);

	cudaMemcpy(dVecA, vecA, sizeof(int)*n, cudaMemcpyHostToDevice);
	cudaMemcpy(dVecB, vecB, sizeof(int)*n, cudaMemcpyHostToDevice);

	cudaVectorDotProduct<<<2, 4>>>(dVecC, dVecA, dVecB, n);
	cudaDeviceSynchronize();
	cudaMemcpy(vecC, dVecC, sizeof(int)*n, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cudaFree(dVecA);
	cudaFree(dVecB);
	cudaFree(dVecC);

	int i;
	printf("A   ");
	for(i = 0; i < n; i++) {
		printf("%2d ", vecA[i]);
	}
	printf("\n");

	printf("B   ");
	for(i = 0; i < n; i++) {
		printf("%2d ", vecB[i]);
	}
	printf("\n");

	printf("Answer = %d\n", vecC[0]);
	
	return 0;
}