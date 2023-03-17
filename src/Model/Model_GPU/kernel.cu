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
	// 9 FLOPS
	// 
	diff= posj - posi;
	dij=sum(diff*diff);
}




__device__ void compute_forces(const float &mi,const float &mj,float &dij, float &dij_mi,float &dij_mj){
	// deux interaction
    // 4 FLOPS
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

__device__ void compute_forces(const float &mj,float &dij, float &dij_mj){
	// 1 iteraction
    // 3 FLOPS
	if (dij > 1)
    {
		dij=rsqrt(dij*dij*dij);
        dij_mj = dij * mj;
    }
    else
    {
        dij_mj = mj;
    }
}

template<const int number_unrolling>
__device__ void update_acc_at_i_between_j(float3 &position_at_i,float3 &accelerationsGPU_at_i,\
	float3 * positionsGPU,float* massesGPU,\
	 const int min_j, const int max_j){
	

	for(int j =min_j; j+number_unrolling<=max_j; j+=number_unrolling){ 

		float3 diff[number_unrolling];
		float dij[number_unrolling];
		float dij_mj[number_unrolling];
		// can be unrolled since number_unrolling is a constant
		
		// registers for j/number_unrolling
		// 6 register for position and acceleration
		
		// 3* K +2*K register acess for writing
		// 4*K loading register
		
		float3* positionGPUJ = &positionsGPU[j];
		float* massesGPUJ = &massesGPU[j];

		// 

		// 
		//#pragma unrolled
		for (int k=0; k<number_unrolling; k++){
			compute_difference(position_at_i,positionGPUJ[k],diff[k], dij[k]); //positionsGPU[j+k], diff[k], dij[k]);
			compute_forces(massesGPUJ[k], dij[k], dij_mj[k]);
			// accelerationsGPU_at_i =accelerationsGPU_at_i + diff[k] * dij_mj[k];
		}

		// for (int k=0; k<number_unrolling; k++){
		// 	compute_forces(massesGPUJ[k], dij[k], dij_mj[k]);//massesGPU[j+k], dij[k], dij_mj[k]);
		// }
		for (int k=0; k<number_unrolling; k++){
			// 6 flops
			accelerationsGPU_at_i =accelerationsGPU_at_i + diff[k] * dij_mj[k];
		}
	}

	// additional values are put in a loop
	float3 diff;
	float dij;
	float dij_mj;

	
	int number_loop=(max_j-min_j)/number_unrolling;
	int next_value=min_j+number_loop*number_unrolling;
	
	// if (threadIdx.x==0 and blockIdx.x==0){
	// 	printf("number loop : %d, from %d to %d and then to %d \n",number_loop,min_j,max_j,next_value);
	// }
	
	for(int j =next_value; j<max_j; j++){

		compute_difference(position_at_i, positionsGPU[j], diff, dij);
		compute_forces(massesGPU[j], dij, dij_mj);
		accelerationsGPU_at_i =accelerationsGPU_at_i + diff * dij_mj;
	}

		
}



template<const int block_dim,const int number_unrolling, const int number_particles_to_load_per_thread>
__global__ void update_acc(float4 * positionAndMassGpu, float3 * velocitiesGPU,\
	 float3 * accelerationsGPU,const int n_particles)
{	


	const int number_particles_shared_memory = block_dim * number_particles_to_load_per_thread;

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	
	int thread_id=threadIdx.x;
	//__shared__ float3 p[block_dim*number_particles_shared_memory];

	
	if (i >= n_particles) return;
	//should be useless, except for debug purpose
	accelerationsGPU[i] = make_float3(0.0f, 0.0f, 0.0f);


	// initialisation of the values : in shared memory
	__shared__ float3 positions_shared[number_particles_shared_memory];
	__shared__ float masses_shared[number_particles_shared_memory];
	
	float3 acceleration_at_i= make_float3(0.0f, 0.0f, 0.0f);

	float3 position_at_i;
	position_at_i.x=positionAndMassGpu[i].x;
	position_at_i.y=positionAndMassGpu[i].y;
	position_at_i.z=positionAndMassGpu[i].z;

	// __shared__ float3 acceleration_at_i[number_threads];
	// __shared__ float3 position_at_i[number_threads];

	// acceleration_at_i[thread_id]= make_float3(0.0f, 0.0f, 0.0f);
	// position_at_i[thread_id]=positionsGPU[i];



	//note that acceleration is not shared since it is acess once per thread
	// this is also the case for the position at index i

	int number_mem_load=(n_particles+number_particles_shared_memory-1)/number_particles_shared_memory;
	for (int mem_load_id=0; mem_load_id<number_mem_load;mem_load_id++){
		
		// copy the data in the shared memory
		int j=mem_load_id*number_particles_shared_memory;
		

		for(int k=0; k<number_particles_to_load_per_thread;k++){
			const int load_to = k*block_dim + threadIdx.x;
			const int load_from = k*block_dim + threadIdx.x + j;

			if (load_from < n_particles){
				positions_shared[load_to].x=positionAndMassGpu[load_from].x;
				positions_shared[load_to].y=positionAndMassGpu[load_from].y;
				positions_shared[load_to].z=positionAndMassGpu[load_from].z;
				masses_shared[load_to]=positionAndMassGpu[load_from].w;

			}


		}

		
		__syncthreads();
		// update the acceleration
		int min_j= 0;
		int max_j= number_particles_shared_memory;
		
		if (j+max_j>n_particles) max_j=n_particles-j;

		// if (thread_id==0 and blockIdx.x==0){
		// 	printf("%d load id : %d , %d, %d  \n",number_mem_load,j,min_j,max_j);
		// }
		update_acc_at_i_between_j<number_unrolling>(
			position_at_i,acceleration_at_i,//position_at_i[thread_id], acceleration_at_i[thread_id],//accelerationsGPU[i],//acc_shared[threadIdx.x],
			positions_shared,  masses_shared,\
		min_j, max_j);
		__syncthreads();

	}
	accelerationsGPU[i]=acceleration_at_i;//[thread_id];
	// update_acc_at_i_between_j<number_unrolling>(positionsGPU, velocitiesGPU, accelerationsGPU, massesGPU,\
	// i, 0, n_particles);
	
	

}


// __global__ void maj_pos(float3 * positionsGPU, float3 * velocitiesGPU, float3 * accelerationsGPU)
// {
// 	// unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

// }

static inline int divup(int a, int b) {
	// how many blocks of size b should we use to represent a block of size a
	return (a + b - 1)/b;
}



void update_position_cu(float4* positionAndMassGpu,float3* velocitiesGPU, \
	float3* accelerationsGPU, int n_particles)
{
	const int nthreads = 64;
	int nblocks = divup(n_particles, nthreads);

	// fps x N² iteration x 18 flops flops/s
	// register : 32*10*k <4Kb
	update_acc<nthreads,10,16> <<<nblocks, nthreads>>>(positionAndMassGpu, velocitiesGPU, accelerationsGPU,\
	 n_particles);


}


#endif // GALAX_MODEL_GPU