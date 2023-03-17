#ifdef GALAX_MODEL_GPU

#include <cmath>
#include <iostream>
#include "Model_GPU.hpp"
#include "kernel.cuh"


inline bool cuda_malloc(void ** devPtr, size_t size)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaMalloc(devPtr, size);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "error: unable to allocate buffer" << std::endl;
		return false;
	}
	return true;
}

inline bool cuda_memcpy(void * dst, const void * src, size_t count, enum cudaMemcpyKind kind)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaMemcpy(dst, src, count, kind);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "error: unable to copy buffer" << std::endl;
		return false;
	}

	return true;
}

void update_position_gpu(float4* positionsAndMassGPU, float3* velocitiesGPU, float3* accelerationsGPU, int n_particles)
{
	update_position_cu(positionsAndMassGPU, velocitiesGPU, accelerationsGPU, n_particles);

	cudaError_t cudaStatus;
	cudaStatus = cudaDeviceSynchronize();
	
	if (cudaStatus != cudaSuccess)
		std::cout << "error: unable to synchronize threads" << std::endl;
}


Model_GPU
::Model_GPU(const Initstate& initstate, Particles& particles)
: Model(initstate, particles),
  positionsAndMassf4    (n_particles),
  velocitiesf3   (n_particles),
  accelerationsf3(n_particles)
{
	// init cuda
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
		std::cout << "error: unable to setup cuda device" << std::endl;


	for (int i = 0; i < n_particles; i++)
	{
		positionsAndMassf4[i].x     = initstate.positionsx [i];
		positionsAndMassf4[i].y     = initstate.positionsy [i];
		positionsAndMassf4[i].z     = initstate.positionsz [i];
		positionsAndMassf4[i].w     = initstate.masses [i];

		velocitiesf3[i].x    = initstate.velocitiesx[i];
		velocitiesf3[i].y    = initstate.velocitiesy[i];
		velocitiesf3[i].z    = initstate.velocitiesz[i];
		accelerationsf3[i].x = 0;
		accelerationsf3[i].y = 0;
		accelerationsf3[i].z = 0;
	}

	cuda_malloc((void**)&positionsAndMassGPU,     n_particles * sizeof(float4));
	cuda_malloc((void**)&velocitiesGPU,     n_particles * sizeof(float3));
	cuda_malloc((void**)&accelerationsGPU,     n_particles * sizeof(float3));
	//cuda_malloc((void**)&massesGPU,     n_particles * sizeof(float));

	cuda_memcpy(positionsAndMassGPU,  positionsAndMassf4.data(), n_particles * sizeof(float4), cudaMemcpyHostToDevice);
	cuda_memcpy(velocitiesGPU,  velocitiesf3.data(), n_particles * sizeof(float3), cudaMemcpyHostToDevice);
	cuda_memcpy(accelerationsGPU,  accelerationsf3.data(), n_particles * sizeof(float3), cudaMemcpyHostToDevice);
	//cuda_memcpy(massesGPU,  initstate.masses.data(), n_particles * sizeof(float), cudaMemcpyHostToDevice);
	


}

Model_GPU
::~Model_GPU()
{
	cudaFree((void**)&positionsAndMassGPU);
}

void Model_GPU
::step()
{


	

	cuda_memcpy(positionsAndMassGPU,  positionsAndMassf4.data(), n_particles * sizeof(float4), cudaMemcpyHostToDevice);


	update_position_gpu(positionsAndMassGPU,velocitiesGPU,accelerationsGPU,n_particles);
	

	cuda_memcpy(accelerationsf3.data(),accelerationsGPU, n_particles * sizeof(float3), cudaMemcpyDeviceToHost);
	
	

	const float G = 10;
	for (int i = 0; i < n_particles; i++)
	{

		accelerationsf3[i].x = accelerationsf3[i].x * G;
		accelerationsf3[i].y = accelerationsf3[i].y * G;
		accelerationsf3[i].z = accelerationsf3[i].z * G;
		velocitiesf3[i].x += accelerationsf3[i].x * 2.0f;
		velocitiesf3[i].y += accelerationsf3[i].y * 2.0f;
		velocitiesf3[i].z += accelerationsf3[i].z * 2.0f;
	    positionsAndMassf4[i].x += velocitiesf3[i].x * 0.1f;
		positionsAndMassf4[i].y += velocitiesf3[i].y * 0.1f;
		positionsAndMassf4[i].z += velocitiesf3[i].z * 0.1f;
		particles.x[i] = positionsAndMassf4[i].x;
		particles.y[i] = positionsAndMassf4[i].y;
		particles.z[i] = positionsAndMassf4[i].z;
		
		
	}
	// printf("first pos : \n");

	// printf("position and mass x  %f \n",positionsAndMassf4[0].x);
	// printf("position and mass y  %f \n",positionsAndMassf4[0].y);
	// printf("position and mass z  %f \n",positionsAndMassf4[0].z);
	// printf("position and mass w  %f \n",positionsAndMassf4[0].w);
	// printf("particles %f \n",particles.x[0]);


}

#endif // GALAX_MODEL_GPU
