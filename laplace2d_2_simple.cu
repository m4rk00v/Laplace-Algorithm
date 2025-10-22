// #include <math.h>
// #include <string.h>
// #include <stdlib.h>
// #include <stdio.h>
// #include <chrono>
// #include <iostream>
// #include <fstream>
// #include <sys/stat.h>

// #include <cuda_runtime.h>
// #include <thrust/reduce.h>
// #include <thrust/device_ptr.h>



// // define tipo base
// #ifndef REAL_T
// #define REAL_T float
// #endif

// //      ↑
// //  A[id-pitch]
// //        |
// // A[id-1] — A[id] — A[id+1]
// //        |
// //  A[id+pitch]
// //      ↓

// // Configuration metadata to make GLOBAL_BEST comparable
// static constexpr double FLOPS_PER_POINT = 6.0; // 4 adds + 1 mul + 1 sub
// static constexpr const char* DTYPE = "fp64";
// static constexpr const char* POINTS_MODE = "padded"; // counting (imax+2)*(jmax+2)


// // Jacobi step using __ldg() for cached reads
// __device__ float atomicMaxFloat(float* addr, float val){
//     int* ai = reinterpret_cast<int*>(addr);
//     int old = __float_as_int(*addr), assumed;
//     if (__int_as_float(old) >= val) return __int_as_float(old);
//     do {
//         assumed = old;
//         old = atomicCAS(ai, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
//     } while (old != assumed);
//     return __int_as_float(old);
// }

// template<int BX, int BY>
// __global__ void jacobi_step(int imax, int jmax,
//                                  const float* __restrict__ A,
//                                  float* __restrict__ Anew,
//                                  float* __restrict__ g_err_max, // 1 valor global
//                                  int do_err) // 1 cada K iter
// {
//     __shared__ float smax;
//     if (threadIdx.x==0 && threadIdx.y==0) smax = 0.0f;
//     __syncthreads();

//     int i = blockIdx.x*BX + threadIdx.x + 1;
//     int j = blockIdx.y*BY + threadIdx.y + 1;
//     if (i>imax || j>jmax) return;

//     int pitch = imax+2, id = j*pitch + i;
//     float c = __ldg(&A[id]);
//     float n = __ldg(&A[id+1]);
//     float s = __ldg(&A[id-1]);
//     float e = __ldg(&A[id+pitch]);
//     float w = __ldg(&A[id-pitch]);

//     float v = 0.25f*(n+s+e+w);
//     Anew[id] = v;

//     if (do_err){
//         float loc = fabsf(v - c);
//         // reducción por bloque
//         atomicMaxFloat(&smax, loc);
//     }
//     __syncthreads();

//     if (do_err && threadIdx.x==0 && threadIdx.y==0){
//         atomicMaxFloat(g_err_max, smax);
//     }
// }





// int main(int argc, const char** argv)
// {
//   int jmax = 4096;
//   int imax = 4096;
//   int iter_max = 100;

//   int bx = 32, by = 8;
//   for (int i = 1; i < argc; ++i) {
//     if (strncmp(argv[i], "--bx=", 5) == 0) bx = atoi(argv[i] + 5);
//     else if (strncmp(argv[i], "--by=", 5) == 0) by = atoi(argv[i] + 5);
//   }
//   if (bx <= 0 || by <= 0 || 1LL * bx * by > 1024) {
//     fprintf(stderr, "Error: invalid block dims. Require bx>0, by>0 and bx*by ≤ 1024.\n");
//     return 1;
//   }

//   REAL_T pi  = static_cast<REAL_T>(2.0 * asin(1.0));
//   const REAL_T tol = static_cast<REAL_T>(1.0e-6);
//   REAL_T error     = static_cast<REAL_T>(1.0);

//   // Host matrices
//   REAL_T *A    = new REAL_T[(size_t)(imax+2) * (jmax+2)];
//   REAL_T *Anew = new REAL_T[(size_t)(imax+2) * (jmax+2)];
//   memset(A, 0, (size_t)(imax+2) * (jmax+2) * sizeof(REAL_T));

//   for (int i = 0; i < imax+2; i++) A[(0)*(imax+2)+i]   = 0.0;
//   for (int i = 0; i < imax+2; i++) A[(jmax+1)*(imax+2)+i] = 0.0;
//   for (int j = 0; j < jmax+2; j++) A[(j)*(imax+2)+0] = sin(pi * j / (jmax+1));
//   for (int j = 0; j < imax+2; j++)  A[(j)*(imax+2)+imax+1] = sin(pi * j / (jmax+1)) * exp(-pi);

//   printf("Jacobi relaxation Calculation: %d x %d mesh\n", imax+2, jmax+2);

//   for (int i = 1; i < imax+2; i++) Anew[(0)*(imax+2)+i]   = 0.0;
//   for (int i = 1; i < imax+2; i++) Anew[(jmax+1)*(imax+2)+i] = 0.0;
//   for (int j = 1; j < jmax+2; j++) Anew[(j)*(imax+2)+0]   = sin(pi * j / (jmax+1));
//   for (int j = 1; j < jmax+2; j++) Anew[(j)*(imax+2)+jmax+1] = sin(pi * j / (jmax+1)) * exp(-pi);

//   size_t N = (size_t)(imax+2) * (jmax+2);
//   REAL_T *d_A = nullptr, *d_Anew = nullptr, *d_err = nullptr;
//   cudaMalloc(&d_A,    N * sizeof(REAL_T));
//   cudaMalloc(&d_Anew, N * sizeof(REAL_T));
//   cudaMalloc(&d_err,  N * sizeof(REAL_T));

//   cudaMemcpy(d_A,    A,    N * sizeof(REAL_T), cudaMemcpyHostToDevice);
//   cudaMemcpy(d_Anew, Anew, N * sizeof(REAL_T), cudaMemcpyHostToDevice);

//   dim3 block(bx, by);
//   // dim3 grid( (imax + block.x - 1) / block.x,
//   //            (jmax + block.y - 1) / block.y );

  
//   dim3 grid( ((imax/2) + block.x - 1) / block.x,
//            (jmax     + block.y - 1) / block.y );

  

//   cudaMemset(d_err, 0, N * sizeof(REAL_T));

//   auto t1 = std::chrono::high_resolution_clock::now();

//   const int K = 5;
//   int k = 0, iter = 0;

//   while (error > tol && iter < iter_max) {
//       jacobi_step<<<grid, block>>>(imax, jmax, d_A, d_Anew, d_err);
//       std::swap(d_A, d_Anew);
//       ++iter; ++k;

//       if (k == K || iter == iter_max) {
//           error = thrust::reduce(thrust::device_pointer_cast(d_err),
//                                  thrust::device_pointer_cast(d_err + N),
//                                  static_cast<REAL_T>(0.0),
//                                  thrust::maximum<REAL_T>());
//           k = 0;
//       }
//   }

//   cudaDeviceSynchronize();
//   auto t2 = std::chrono::high_resolution_clock::now();

//   printf("%5d, %0.6f\n", iter, (double)error);

//   REAL_T err_diff = fabs((100.0 * (error / 2.421354960840227e-03)) - 100.0);
//   printf("Total error is within %3.15E %% of the expected error\n", (double)err_diff);
//   if (err_diff < 0.001) printf("This run is considered PASSED\n");
//   else                  printf("This test is considered FAILED\n");

//   std::chrono::duration<double, std::milli> ms_double = t2 - t1;
//   const double time_ms = ms_double.count();

//   const double points  = static_cast<double>(imax+2) * static_cast<double>(jmax+2);
//   const double iters   = static_cast<double>(iter);
//   const double seconds = time_ms / 1000.0;

//   const double total_flops = FLOPS_PER_POINT * points * iters;
//   const double gflops = total_flops / (seconds * 1e9);

//   std::cout << "GFLOPS/s (" << FLOPS_PER_POINT << " flops/pt): "
//             << gflops << "  points " << points
//             << "  iterations " << iters << std::endl;

//   const double BYTES_PER_POINT = 5.0 * sizeof(REAL_T);
//   const double total_bytes = BYTES_PER_POINT * points * iters;
//   const double bandwidth_GBps = total_bytes / (seconds * 1e9);

//   std::cout << "Bandwidth (approx): " << bandwidth_GBps << " GB/s" << std::endl;

//   const double avg_iter_ms = (iter > 0) ? (time_ms / static_cast<double>(iter)) : 0.0;
//   printf("SUMMARY imax=%d jmax=%d dtype=%s flops_per_pt=%.0f points=%s bx=%d by=%d time_ms=%.3f iters=%d gflops=%.6f bandwidth_GBps=%.6f avg_iter_ms=%.6f\n",
//          imax, jmax, DTYPE, FLOPS_PER_POINT, POINTS_MODE,
//          bx, by, time_ms, iter, gflops, bandwidth_GBps, avg_iter_ms);

//   std::cout << "--------------------------------------------------------------------------------";

//   const char* LOG_PATH = "jacobi_log.txt";
//   {
//       std::ofstream out(LOG_PATH, std::ios::app);
//       if (out) {
//           out << "SUMMARY "
//               << "imax=" << imax << " "
//               << "jmax=" << jmax << " "
//               << "dtype=" << DTYPE << " "
//               << "flops_per_pt=" << FLOPS_PER_POINT << " "
//               << "points=" << POINTS_MODE << " "
//               << "bx=" << bx << " "
//               << "by=" << by << " "
//               << "time_ms=" << time_ms << " "
//               << "iters=" << iter << " "
//               << "gflops=" << gflops << " "
//               << "bandwidth_GBps=" << bandwidth_GBps << " "
//               << "avg_iter_ms=" << avg_iter_ms
//               << "\n";
//       }
//   }

//   // =====================================================
// // Human-readable name for REAL_T (for logging)
// // =====================================================



//   cudaFree(d_A);
//   cudaFree(d_Anew);
//   cudaFree(d_err);
//   delete[] A;
//   delete[] Anew;

//   return 0;
// }



// laplace2d_2.cu
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <type_traits>

#include <cuda_runtime.h>

// ===================== Config & helpers =====================

// Tipo base desde compilación: -DREAL_T=float o -DREAL_T=double
#ifndef REAL_T
#define REAL_T float
#endif

static_assert(std::is_same<REAL_T,float>::value || std::is_same<REAL_T,double>::value,
              "REAL_T must be float or double");

// FLOPs por punto (Jacobi 5-pt: 4 sumas + 1 mul + 1 resta)
static constexpr double FLOPS_PER_POINT = 6.0;
// Para logging (puntos contados como (imax+2)*(jmax+2))
static constexpr const char* POINTS_MODE = "padded";

// Nombre humano del dtype
__host__ __device__ static inline const char* dtype_name() {
  if constexpr (std::is_same<REAL_T,double>::value) return "float64";
  else                                              return "float32";
}

// CUDA error check
#define CUDA_CHECK(call) do { \
  cudaError_t __e = (call); \
  if (__e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__, cudaGetErrorString(__e)); \
    exit(1); \
  } \
} while(0)

// atomicMax para float/double
__device__ inline float atomicMaxFloat(float* addr, float val){
  int* ai = reinterpret_cast<int*>(addr);
  int old = __float_as_int(*addr), assumed;
  while (__int_as_float(old) < val) {
    assumed = old;
    old = atomicCAS(ai, assumed, __float_as_int(val));
  }
  return __int_as_float(old);
}

__device__ inline double atomicMaxDouble(double* addr, double val){
  unsigned long long* ull = reinterpret_cast<unsigned long long*>(addr);
  unsigned long long old = *ull, assumed;
  while (__longlong_as_double(old) < val) {
    assumed = old;
    old = atomicCAS(ull, assumed, __double_as_longlong(val));
  }
  return __longlong_as_double(old);
}

__device__ inline void atomicMaxReal(float*  a, float  v){ atomicMaxFloat(a,  v); }
__device__ inline void atomicMaxReal(double* a, double v){ atomicMaxDouble(a, v); }

// ===================== Kernel =====================
// Jacobi 5-pt; calcula error máximo por bloque cuando do_err==1
__global__ void jacobi_step(int imax, int jmax,
                            const REAL_T* __restrict__ A,
                            REAL_T* __restrict__ Anew,
                            REAL_T* __restrict__ g_err_max,  // escalar global (1 valor)
                            int do_err)                       // 0/1: medir error esta iter
{
  __shared__ REAL_T smax; // reducción por bloque
  if (threadIdx.x==0 && threadIdx.y==0) smax = (REAL_T)0;
  __syncthreads();

  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (i > imax || j > jmax) return;

  const int pitch = imax + 2;
  const int id = j * pitch + i;

  // 5 lecturas (c, n, s, e, w)
  REAL_T c = __ldg(&A[id]);
  REAL_T n = __ldg(&A[id+1]);
  REAL_T s = __ldg(&A[id-1]);
  REAL_T e = __ldg(&A[id+pitch]);
  REAL_T w = __ldg(&A[id-pitch]);

  REAL_T v = (REAL_T)0.25 * (n + s + e + w);
  Anew[id] = v;  // 1 escritura

  if (do_err){
    REAL_T loc = fabs(v - c);
    // max local en shared usando atomicMax (simple/robusto)
    atomicMaxReal(&smax, loc);
  }
  __syncthreads();

  if (do_err && threadIdx.x==0 && threadIdx.y==0){
    // un atómico global por bloque
    atomicMaxReal(g_err_max, smax);
  }
}

// ===================== Main =====================
int main(int argc, const char** argv)
{
  // Dominio y control
  int jmax = 4096;     // Y
  int imax = 4096;     // X
  int iter_max = 100;

  // Bloques desde CLI
  int bx = 128, by = 2;
  for (int i = 1; i < argc; ++i) {
    if (strncmp(argv[i], "--bx=", 5) == 0) bx = atoi(argv[i] + 5);
    else if (strncmp(argv[i], "--by=", 5) == 0) by = atoi(argv[i] + 5);
  }
  if (bx <= 0 || by <= 0 || 1LL * bx * by > 1024) {
    fprintf(stderr, "Error: invalid block dims. Require bx>0, by>0 and bx*by ≤ 1024.\n");
    return 1;
  }

  // Constantes
  REAL_T pi  = (REAL_T)(2.0 * asin(1.0));
  const REAL_T tol = (REAL_T)1.0e-6;
  REAL_T error     = (REAL_T)1.0;

  // Host buffers
  size_t N = (size_t)(imax+2) * (jmax+2);
  REAL_T *A    = new REAL_T[N];
  REAL_T *Anew = new REAL_T[N];
  if (!A || !Anew) {
    fprintf(stderr, "Host allocation failed\n");
    return 1;
  }
  memset(A, 0, N * sizeof(REAL_T));

  // Condiciones de borde (eval en double y casteo)
  for (int i = 0; i < imax+2; i++) A[(0)*(imax+2)+i]   = (REAL_T)0.0;
  for (int i = 0; i < imax+2; i++) A[(jmax+1)*(imax+2)+i] = (REAL_T)0.0;
  for (int j = 0; j < jmax+2; j++) A[(j)*(imax+2)+0] = (REAL_T)sin((double)pi * j / (jmax+1));
  for (int j = 0; j < imax+2; j++)  A[(j)*(imax+2)+imax+1] =
      (REAL_T)( sin((double)pi * j / (jmax+1)) * exp(-(double)pi) );

  printf("Jacobi relaxation Calculation: %d x %d mesh | dtype=%s\n",
         imax+2, jmax+2, dtype_name());

  // Anew bordes
  for (int i = 1; i < imax+2; i++) Anew[(0)*(imax+2)+i]   = (REAL_T)0.0;
  for (int i = 1; i < imax+2; i++) Anew[(jmax+1)*(imax+2)+i] = (REAL_T)0.0;
  for (int j = 1; j < jmax+2; j++) Anew[(j)*(imax+2)+0]   = (REAL_T)sin((double)pi * j / (jmax+1));
  for (int j = 1; j < jmax+2; j++) Anew[(j)*(imax+2)+jmax+1] =
      (REAL_T)( sin((double)pi * j / (jmax+1)) * exp(-(double)pi) );

  // Device buffers
  REAL_T *d_A = nullptr, *d_Anew = nullptr, *d_err_max = nullptr;
  CUDA_CHECK(cudaMalloc(&d_A,     N * sizeof(REAL_T)));
  CUDA_CHECK(cudaMalloc(&d_Anew,  N * sizeof(REAL_T)));
  CUDA_CHECK(cudaMalloc(&d_err_max, sizeof(REAL_T)));

  CUDA_CHECK(cudaMemcpy(d_A,    A,    N * sizeof(REAL_T), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_Anew, Anew, N * sizeof(REAL_T), cudaMemcpyHostToDevice));

  // Launch config
  dim3 block(bx, by);
  dim3 grid( (imax + block.x - 1) / block.x,
             (jmax + block.y - 1) / block.y );

  // Bucle principal
  auto t1 = std::chrono::high_resolution_clock::now();
  const int K = 5; // medir error cada K
  int k = 0, iter = 0;

  while (error > tol && iter < iter_max) {
    int do_err = (k == 0) ? 1 : 0;          // midamos al cerrar el bloque K
    if (do_err) CUDA_CHECK(cudaMemset(d_err_max, 0, sizeof(REAL_T)));

    jacobi_step<<<grid, block>>>(imax, jmax, d_A, d_Anew, d_err_max, do_err);
    CUDA_CHECK(cudaGetLastError());

    std::swap(d_A, d_Anew);
    ++iter; ++k;

    if (k == K || iter == iter_max) {
      CUDA_CHECK(cudaMemcpy(&error, d_err_max, sizeof(REAL_T), cudaMemcpyDeviceToHost));
      k = 0;
    }
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  auto t2 = std::chrono::high_resolution_clock::now();

  // Reporte de convergencia
  printf("%5d, %0.6f\n", iter, (double)error);
  REAL_T err_diff = (REAL_T)fabs((100.0 * ((double)error / 2.421354960840227e-03)) - 100.0);
  printf("Total error is within %3.15E %% of the expected error\n", (double)err_diff);
  if (err_diff < (REAL_T)0.001) printf("This run is considered PASSED\n");
  else                          printf("This test is considered FAILED\n");

  // Métricas
  std::chrono::duration<double, std::milli> ms_double = t2 - t1;
  const double time_ms = ms_double.count();
  const double points  = (double)(imax+2) * (double)(jmax+2);
  const double iters   = (double)(iter);
  const double seconds = time_ms / 1000.0;

  const double total_flops = FLOPS_PER_POINT * points * iters;
  const double gflops = total_flops / (seconds * 1e9);

  // Bytes por punto aprox: 5 lecturas (c, n, s, e, w) + 1 escritura Anew = 6 * sizeof(REAL_T)
  const double BYTES_PER_POINT = 6.0 * sizeof(REAL_T);
  const double total_bytes = BYTES_PER_POINT * points * iters;
  const double bandwidth_GBps = total_bytes / (seconds * 1e9);

  std::cout << "GFLOPS/s (" << FLOPS_PER_POINT << " flops/pt): "
            << gflops << "  points " << points
            << "  iterations " << iters << std::endl;

  std::cout << "Bandwidth (approx): " << bandwidth_GBps << " GB/s" << std::endl;

  const double avg_iter_ms = (iter > 0) ? (time_ms / (double)iter) : 0.0;

  printf("SUMMARY imax=%d jmax=%d dtype=%s flops_per_pt=%.0f points=%s bx=%d by=%d time_ms=%.3f iters=%d gflops=%.6f bandwidth_GBps=%.6f avg_iter_ms=%.6f\n",
         imax, jmax, dtype_name(), FLOPS_PER_POINT, POINTS_MODE,
         bx, by, time_ms, iter, gflops, bandwidth_GBps, avg_iter_ms);

  // (Opcional) Append a log TXT
  const char* LOG_PATH = "jacobi_log.txt";
  {
    std::ofstream out(LOG_PATH, std::ios::app);
    if (out) {
      out << "SUMMARY "
          << "imax=" << imax << " "
          << "jmax=" << jmax << " "
          << "dtype=" << dtype_name() << " "
          << "flops_per_pt=" << FLOPS_PER_POINT << " "
          << "points=" << POINTS_MODE << " "
          << "bx=" << bx << " "
          << "by=" << by << " "
          << "time_ms=" << time_ms << " "
          << "iters=" << iter << " "
          << "gflops=" << gflops << " "
          << "bandwidth_GBps=" << bandwidth_GBps << " "
          << "avg_iter_ms=" << avg_iter_ms
          << "\n";
    }
  }

  // Limpieza
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_Anew));
  CUDA_CHECK(cudaFree(d_err_max));
  delete[] A;
  delete[] Anew;

  return 0;
}
