#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <chrono>
#include <iostream>

#include <cuda_runtime.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>


//      ↑
//  A[id-pitch]
//        |
// A[id-1] — A[id] — A[id+1]
//        |
//  A[id+pitch]
//      ↓

// __global__ void jacobi_step(int imax, int jmax,
//                             const double* __restrict__ A,
//                             double* __restrict__ Anew,
//                             double* __restrict__ err)
// {
//     // Hilo <-> celda interior: i=1..imax, j=1..jmax
//     int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
//     int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
//     if (i > imax || j > jmax) return;

//     int pitch = imax + 2;          // ancho con halo
//     int id = j * pitch + i;        // acceso fila-continua

//     double newv = 0.25 * ( A[id+1] + A[id-1] + A[id-pitch] + A[id+pitch] ); // 4
//     Anew[id] = newv;
//     err[id]  = fabs(newv - A[id]); // error local para reducción // 1
// }

// //FLOATS per point 
// // double newv = 0.25 * ( A[id+1] + A[id-1] + A[id-pitch] + A[id+pitch] ) - 4 operations
// // fabs(newv - A[id]);  substracts operation, absolute value - 2
// // -------- FLOATS_PER_POINT = 6 -------------


//use of __ldg
__global__ void jacobi_step(int imax, int jmax,
                            const double* __restrict__ A,
                            double* __restrict__ Anew,
                            double* __restrict__ err)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i > imax || j > jmax) return;

    const int pitch = imax + 2;
    const int id = j * pitch + i;

    // usar __ldg para las lecturas de A
    double Ai   = __ldg(&A[id]);
    double newv = 0.25 * ( __ldg(&A[id+1]) + __ldg(&A[id-1])
                         + __ldg(&A[id-pitch]) + __ldg(&A[id+pitch]) );
    Anew[id] = newv;
    err[id]  = fabs(newv - Ai);
}




int main(int argc, const char** argv)
{
  //Size along y
  int jmax = 4096;
  //Size along x
  int imax = 4096;
  //Size along x
  int iter_max = 100;

  double pi  = 2.0 * asin(1.0);
  //tolerance
  const double tol = 1.0e-6;
  double error     = 1.0;


  //matriz settings
  double * A = new double[(imax+2) * (jmax+2)];
  double * Anew = new double[(imax+2) * (jmax+2)];
  // filling matriz A with zeros
  memset(A, 0, (imax+2) * (jmax+2) * sizeof(double));


  //

  // set boundary conditions
  for (int i = 0; i < imax+2; i++) //top 
    A[(0)*(imax+2)+i]   = 0.0;

  for (int i = 0; i < imax+2; i++) //bottom
    A[(jmax+1)*(imax+2)+i] = 0.0;

  for (int j = 0; j < jmax+2; j++) //left border
  {
    A[(j)*(imax+2)+0] = sin(pi * j / (jmax+1));
  }

  for (int j = 0; j < imax+2; j++) //right border
  {
    A[(j)*(imax+2)+imax+1] = sin(pi * j / (jmax+1))*exp(-pi);
  }

  printf("Jacobi relaxation Calculation: %d x %d mesh\n", imax+2, jmax+2);

  int iter = 0;

  for (int i = 1; i < imax+2; i++)
    Anew[(0)*(imax+2)+i]   = 0.0;

  for (int i = 1; i < imax+2; i++)
    Anew[(jmax+1)*(imax+2)+i] = 0.0;

  for (int j = 1; j < jmax+2; j++)
    Anew[(j)*(imax+2)+0]   = sin(pi * j / (jmax+1));

  for (int j = 1; j < jmax+2; j++)
    Anew[(j)*(imax+2)+jmax+1] = sin(pi * j / (jmax+1))*expf(-pi);


  // memory GPU
  // ====== memoria en GPU (device) ======
  size_t N = (size_t)(imax+2) * (jmax+2);

  double *d_A    = nullptr;
  double *d_Anew = nullptr;
  double *d_err  = nullptr;

  cudaMalloc(&d_A,    N * sizeof(double));
  cudaMalloc(&d_Anew, N * sizeof(double));
  cudaMalloc(&d_err,  N * sizeof(double));

  // copiar lo inicializado en CPU -> GPU
  cudaMemcpy(d_A,    A,    N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_Anew, Anew, N * sizeof(double), cudaMemcpyHostToDevice);


  // Configuración de ejecución
  dim3 block(32,32); // nxn should be a mult of 32 , less than 1024 , try tot he the real shape size 
  
  dim3 grid( (imax + block.x - 1) / block.x,
            (jmax + block.y - 1) / block.y ); 
          

  // Asegura err = 0 al inicio (bordes no escritos por el kernel)
  cudaMemset(d_err, 0, N * sizeof(double));

  auto t1 = std::chrono::high_resolution_clock::now();

  // while (error > tol && iter < iter_max)
  // {
     
  //     // 1) Jacobi + error local en GPU
  //     jacobi_step<<<grid, block>>>(imax, jmax, d_A, d_Anew, d_err); //one measurement 
  //     // cudaDeviceSynchronize();  redundant 

  //     // 2) Reducción: máximo del error (en toda la malla, bordes son 0)
  //     error = thrust::reduce(
  //         thrust::device_ptr<double>(d_err),
  //         thrust::device_ptr<double>(d_err + N),
  //         0.0, thrust::maximum<double>()); //how much data this move - extra read - second measurement

  //     // 3) Intercambio de buffers (evita copiar)
  //     std::swap(d_A, d_Anew); //not move data

  //     if (iter % 10 == 0) printf("%5d, %0.6f\n", iter, error);
  //     ++iter;
  // }

  const int K = 5;
  int k = 0;

  while (error > tol && iter < iter_max) {
      jacobi_step<<<grid, block>>>(imax, jmax, d_A, d_Anew, d_err);
      std::swap(d_A, d_Anew);
      ++iter; ++k;

      if (k == K || iter == iter_max) {
          error = thrust::reduce(thrust::device_pointer_cast(d_err),
                                thrust::device_pointer_cast(d_err + N),
                                0.0, thrust::maximum<double>());
          k = 0;
      }
  }

  
  // cudaMemcpy(A, d_A, N * sizeof(double), cudaMemcpyDeviceToHost);

  auto t2 = std::chrono::high_resolution_clock::now();


  printf("%5d, %0.6f\n", iter, error);

  double err_diff = fabs((100.0*(error/2.421354960840227e-03))-100.0);
  printf("Total error is within %3.15E %% of the expected error\n",err_diff);
  if(err_diff < 0.001)
    printf("This run is considered PASSED\n");
  else
    printf("This test is considered FAILED\n");

  std::chrono::duration<double, std::milli> ms_double = t2 - t1;
  std::cout << ms_double.count() << "ms\n";

    // GFLOPS: FLOPs por punto * puntos interiores * iteraciones / tiempo
  const double FLOPS_PER_POINT = 6.0; // uquantity of operations
  // const double points = static_cast<double>(imax+2) * static_cast<double>(jmax+2); // 
  const double internal_points = static_cast<double>(imax) * static_cast<double>(jmax); // 

  const double points = static_cast<double>(imax+2) * static_cast<double>(jmax+2); // 
  const double iters  = static_cast<double>(iter);
  const double seconds = ms_double.count() / 1000.0;

  const double total_flops = FLOPS_PER_POINT * points * iters;
  const double gflops = total_flops / (seconds * 1e9);

  std::cout << "GFLOPS/s ("
            << FLOPS_PER_POINT << " flops/pt): "
            << gflops << " points "
            << points << " iterations "
            << iters <<
            std::endl; 

            
  // ===== Bandwidth (GB/s) =====

  const double BYTES_PER_POINT = 5.0 * sizeof(double); // error reduce , error, reading A , write A new (4) ------ 4 ---------
  const double total_bytes = BYTES_PER_POINT * points * iters;
  const double bandwidth_GBps = total_bytes / (seconds * 1e9);

  std::cout << "Bandwidth (approx): " << bandwidth_GBps << " GB/s" << std::endl;


  cudaFree(d_A);
  cudaFree(d_Anew);
  cudaFree(d_err);


  return 0;
}
