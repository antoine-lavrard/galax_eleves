#ifdef GALAX_MODEL_GPU
#include <iostream>
#include "cuda.h"
#include "kernel.cuh"
#define DIFF_T (0.1f)
#define EPS (1.0f)

__device__ float3 operator-(const float3 &a, const float3 &b) {

  return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);

}

__device__ float3 operator+(const float3 &a, const float3 &b) {

  return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);

}



__device__ float3 operator*(const float3 &a, const float3 &b) {

  return make_float3(a.x*b.x, a.y*b.y, a.z*b.z);

}

__device__ float3 operator*(const float3 &a, const float b) {

  return make_float3(a.x*b, a.y*b, a.z*b);

}

__device__ float sum(const float3 &a) {

  return a.x + a.y + a.z;
}


__device__ void compute_difference(float3 &posi, float3 &posj, float3 &diff, float &dij)
{
	diff= posj - posi;
	dij=sum(diff*diff);
}




__device__ void compute_forces(const float &mi,const float &mj,float &dij, float &dij_mi,float &dij_mj){
    if (dij > 1)
    {
		dij=rsqrt(dij*dij*dij);
		dij_mi = dij * mj;
        dij_mj = dij * mi;
        
    }
    else
    {
		dij_mi = mj;
        dij_mj = mi;
    }
}

__device__ void compute_forces(const float &mj,float &dij, float &dij_mi){
    if (dij > 1)
    {
		dij=rsqrt(dij*dij*dij);
        dij_mi = dij * mj;
    }
    else
    {
        dij_mi = mj;
    }
}


template<const int number_unrolling>
__global__ void update_acc(float3 * positionsGPU, float3 * velocitiesGPU,\
	 float3 * accelerationsGPU,float* massesGPU,const int n_particles)
{	

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (i >= n_particles) return;
	
	accelerationsGPU[i] = make_float3(0.0f, 0.0f, 0.0f);
	
	for(int j =0; j<n_particles; j+=number_unrolling){ 

		float3 diff[number_unrolling];
		float dij[number_unrolling];
		float dij_mi[number_unrolling];
		// can be unrolled since number_unrolling is a constant
		for (int k=0; k<number_unrolling; k++){
			compute_difference(positionsGPU[i], positionsGPU[j+k], diff[k], dij[k]);
			compute_forces(massesGPU[j+k], dij[k], dij_mi[k]);

			accelerationsGPU[i] =accelerationsGPU[i] + diff[k] * dij_mi[k];
		}

	}

	// missing values in the last loop
	float3 diff;
	float dij;
	float dij_mi;
	int number_loop=n_particles/number_unrolling;
	int next_value=number_loop*number_unrolling;
	for(int j =next_value; j<n_particles; j++){ 
		
		compute_difference(positionsGPU[i], positionsGPU[j], diff, dij);
		compute_forces(massesGPU[j], dij, dij_mi);
		accelerationsGPU[i] =accelerationsGPU[i] + diff * dij_mi;
	}

}


// __global__ void maj_pos(float3 * positionsGPU, float3 * velocitiesGPU, float3 * accelerationsGPU)
// {
// 	// unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

// }

static inline int divup(int a, int b) {
	// how many blocks of size b should we use to represent a block of size a
	return (a + b - 1)/b;
}



void update_position_cu(float3* positionsGPU,float3* velocitiesGPU, \
	float3* accelerationsGPU, float* massesGPU, int n_particles)
{
	int nthreads = 128;
	int nblocks = divup(n_particles, nthreads);

	update_acc<3> <<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU,\
	 massesGPU, n_particles);

	

}


#endif // GALAX_MODEL_GPU