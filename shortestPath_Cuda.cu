/*
 *  Copyright (C) 2009 by Vitsios Dimitrios
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in
 *  all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 */

/**
 * Shortest path,
 * parallel implementation
 * using CUDA
 */


#include <stdio.h>
#include <sys/types.h>
#include <cuda.h>
#include <time.h>
#include <math.h>
#include <float.h>
#define N 512


__global__ void shortest_path( float *c, float *C , int k, int iter, int *path, int counter, int L, int K)
{

__shared__ float C_th[N]; 
__shared__ int MIN_IDX[N];

  unsigned int s;

  int index = blockIdx.x * N + threadIdx.x;

  MIN_IDX[threadIdx.x] = threadIdx.x;

  int offset = (k*k+iter)*N*N;

  C_th[threadIdx.x] = C[threadIdx.x] + c[index + offset];
  
  __syncthreads();

  
  for(s=blockDim.x/2;s>32;s>>=1){
	if(threadIdx.x < s){
		if( C_th[MIN_IDX[threadIdx.x]] > C_th[MIN_IDX[threadIdx.x + s]]){
			MIN_IDX[threadIdx.x] = MIN_IDX[threadIdx.x + s];
		}
	}
	__syncthreads();
  }	

  if(threadIdx.x < 32){
	if(N > 32){	
		if( C_th[MIN_IDX[threadIdx.x]] > C_th[MIN_IDX[threadIdx.x + 32]]){
				MIN_IDX[threadIdx.x] = MIN_IDX[threadIdx.x + 32];
			}
	}
	if(N > 16){	
		if( C_th[MIN_IDX[threadIdx.x]] > C_th[MIN_IDX[threadIdx.x + 16]]){
				MIN_IDX[threadIdx.x] = MIN_IDX[threadIdx.x + 16];
			}
 	}
	if(N > 8){	
		if( C_th[MIN_IDX[threadIdx.x]] > C_th[MIN_IDX[threadIdx.x + 8]]){
				MIN_IDX[threadIdx.x] = MIN_IDX[threadIdx.x + 8];
			}
	}
	if(N > 4){	
		if( C_th[MIN_IDX[threadIdx.x]] > C_th[MIN_IDX[threadIdx.x + 4]]){
				MIN_IDX[threadIdx.x] = MIN_IDX[threadIdx.x + 4];
			}
 	}
	if(N > 2){	
		if( C_th[MIN_IDX[threadIdx.x]] > C_th[MIN_IDX[threadIdx.x + 2]]){
				MIN_IDX[threadIdx.x] = MIN_IDX[threadIdx.x + 2];
			}
	}
		if( C_th[MIN_IDX[threadIdx.x]] > C_th[MIN_IDX[threadIdx.x + 1]]){
				MIN_IDX[threadIdx.x] = MIN_IDX[threadIdx.x + 1];
			}
  }

  __syncthreads();

  if(threadIdx.x == 0){
  	C[blockIdx.x] = C_th[MIN_IDX[0]]; 
	path[blockIdx.x*L + counter] = MIN_IDX[0];	
  }

}


int main(int argc, char* argv[])
{

FILE *f_path;
clock_t start, end;
int i,j,r=0, k=1, iter, counter=0, *path_host, *path, L, K;
float *c_host, *c, *C_host, *C_host_L, *C;  

printf("Type the number of levels of the graph (L):       [ < 2048]\n\n");
scanf("%d",&L);

printf("Type the value of 'K':       \n\n");
scanf("%d",&K);

cudaStream_t stream[2];
cudaStreamCreate(&stream[0]);
cudaStreamCreate(&stream[1]);


C_host = (float *)malloc( N * sizeof(float) );
C_host_L = (float *)malloc( N * sizeof(float) );
path_host = (int *)malloc( L * N * sizeof(int) );
path = (int *)malloc( L * N * sizeof(int) );


int size_c = L * N * N * sizeof(float);
cudaMallocHost((void**)&c_host, size_c);

srand(5);

	//initialiZe c_host[][] matrix...
	for(i=0; i<L*N*N; i++)
             c_host[i] = (float)(rand() % 1000 + 10)/100; //supposing that edges have costs from  0.1 to 109

	//initialiZe C_host[][] matrix...		 
	for(i=0; i<N; i++)
             C_host[i] = (float)(rand() % 1000 + 10)/100;
	//initialiZe random C_host matrix for the last but one level
	for(i=0; i<N; i++)
             C_host_L[i] = (float)(rand() % 1000 + 10)/100;

start=clock();

int size1 = N * N * sizeof(float);
cudaMalloc((void**)&c,2*K*size1);
cudaMemcpy( c, c_host, K*size1, cudaMemcpyHostToDevice );

int size2 = N * sizeof(float);
cudaMalloc((void**)&C,size2);
cudaMemcpy( C, C_host, size2, cudaMemcpyHostToDevice );

int size3 = N * L* sizeof(int);
cudaMalloc((void**)&path,size3);
cudaMemcpy( path, path_host, size3, cudaMemcpyHostToDevice );
    
printf("GPU computing started!\n");

for(r=1; r<(L/K); r++){
		
		
	cudaMemcpyAsync( c+k*K*N*N, c_host+r*K*N*N, K*size1, cudaMemcpyHostToDevice, stream[0] );

	for(iter=0;iter<K;iter++){
		shortest_path<<< N, N, 0, stream[1] >>>( c, C , !k, iter, path, counter, L, K);
		counter++;
	} 

	cudaThreadSynchronize();	 		
	k == 0 ? k = 1 : k = 0;

}

for(iter=0;iter<K;iter++){
	
	shortest_path<<< N, N, 0, stream[0] >>>( c, C , !k, iter, path, counter, L, K);
	counter++;
}

cudaMemcpy(C_host, C, size2, cudaMemcpyDeviceToHost);
cudaMemcpy(path_host, path, size3, cudaMemcpyDeviceToHost);


end=clock();




float total_min = C_host[0]+C_host_L[0];
int total_min_idx = 0;
for(i=1; i<N; i++){
	if( C_host[i] + C_host_L[i]< total_min){
		total_min = C_host[i]+ C_host_L[i];
		total_min_idx = i;
	}
}

printf("\nTotal min = %f", total_min);
printf("\nTotal min INDEX = %d", total_min_idx);

f_path = fopen("path.txt","w");
             
printf("\n\n*** path_host ***\n");
	for(j=0; j<L; j++){
             printf("%d ", path_host[ total_min_idx*L + j ]);             
	     	 fprintf(f_path,"%d", path_host[ total_min_idx*L + j ]);
	}


    printf("\n\n*******************************************************************************");
    printf("\nTotal time elapsed for transfering the data and computing in GPU: %d ms",(end-start)*1000/CLOCKS_PER_SEC);	


scanf("%d",&i);
return EXIT_SUCCESS;
}
