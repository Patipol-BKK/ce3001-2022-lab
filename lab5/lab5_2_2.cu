#include <stdio.h>

__global__ void cudaVectorAddition(int *vecC, int *vecA, int *vecB, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	vecC[idx] = vecA[idx] + vecB[idx];
}

int main() {
	int n = 4;
	int vecA[n] = {1, 2, 4, 2};
	int vecB[n] = {7, 1, 3, 5};
	int vecC[n];

	int *dVecA, *dVecB, *dVecC;
	cudaMalloc((void**)&dVecA, sizeof(int)*n);
	cudaMalloc((void**)&dVecB, sizeof(int)*n);
	cudaMalloc((void**)&dVecC, sizeof(int)*n);

	cudaMemcpy(dVecA, vecA, sizeof(int)*n, cudaMemcpyHostToDevice);
	cudaMemcpy(dVecB, vecB, sizeof(int)*n, cudaMemcpyHostToDevice);

	cudaVectorAddition<<<2, 4>>>(dVecC, dVecA, dVecB, n);
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

	printf("C   ");
	for(i = 0; i < n; i++) {
		printf("%2d ", vecC[i]);
	}
	printf("\n");

	return 0;
}