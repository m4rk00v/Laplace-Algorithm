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
// #define REAL_T double
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
// __global__ void jacobi_step(int imax, int jmax,
//                             const REAL_T* __restrict__ A,
//                             REAL_T* __restrict__ Anew,
//                             REAL_T* __restrict__ err)
// {
//     int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
//     int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
//     if (i > imax || j > jmax) return;

//     const int pitch = imax + 2;
//     const int id = j * pitch + i;

//     REAL_T Ai   = __ldg(&A[id]);
//     REAL_T newv = static_cast<REAL_T>(0.25) * ( __ldg(&A[id+1]) + __ldg(&A[id-1])
//                        + __ldg(&A[id-pitch]) + __ldg(&A[id+pitch]) );
//     Anew[id] = newv;
//     err[id]  = fabs(newv - Ai);
// }

// int main(int argc, const char** argv)
// {
//   int jmax = 4096;
//   int imax = 4096;
//   int iter_max = 100;

//   int bx = 1024, by = 1;
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
//   dim3 grid( (imax + block.x - 1) / block.x,
//              (jmax + block.y - 1) / block.y );

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


// #include <math.h>
// #include <string.h>
// #include <stdlib.h>
// #include <stdio.h>
// #include <chrono>
// #include <iostream>
// #include <fstream>
// #include <iomanip>
// #include <sys/stat.h>

// #include <cuda_runtime.h>
// #include <thrust/reduce.h>
// #include <thrust/device_ptr.h>

// #if defined(HAS_HALF)
//   #include <cuda_fp16.h>
// #endif

// // Base type
// #ifndef REAL_T
// #define REAL_T double
// #endif

// // Human-readable dtype string (set from the build with -DDTYPE_STR="\"fp64\"" etc.)
// #ifndef DTYPE_STR
// #define DTYPE_STR "fp64"
// #endif
// static constexpr const char* DTYPE = DTYPE_STR;

// //      ↑
// //  A[id-pitch]
// //        |
// // A[id-1] — A[id] — A[id+1]
// //        |
// //  A[id+pitch]
// //      ↓

// // Configuration metadata to make GLOBAL_BEST comparable
// static constexpr double FLOPS_PER_POINT = 6.0; // 4 adds + 1 mul + 1 sub
// static constexpr const char* POINTS_MODE = "padded"; // counting (imax+2)*(jmax+2)

// // -----------------------------------------------------------------------------
// // Helpers to make code work uniformly for double/float/__half
// // -----------------------------------------------------------------------------

// // Host-side: convert a double to REAL_T safely
// #if defined(HAS_HALF)
// __host__ __device__ inline __half real_from_double(double x) { return __double2half(x); }
// #else
// __host__ __device__ inline REAL_T real_from_double(double x) { return static_cast<REAL_T>(x); }
// #endif

// // Device-side: abs and subtract for REAL_T (handles __half properly)
// #if defined(HAS_HALF)
// __device__ __forceinline__ __half real_sub(__half a, __half b) { return __hsub(a, b); }
// __device__ __forceinline__ __half real_abs(__half x)           { return __habs(x); }
// #else
// template <typename T>
// __device__ __forceinline__ T real_sub(T a, T b) { return a - b; }

// template <typename T>
// __device__ __forceinline__ T real_abs(T x) { return fabs(x); }
// #endif

// // Jacobi step using __ldg() for cached reads
// __global__ void jacobi_step(int imax, int jmax,
//                             const REAL_T* __restrict__ A,
//                             REAL_T* __restrict__ Anew,
//                             REAL_T* __restrict__ err)
// {
//     int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
//     int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
//     if (i > imax || j > jmax) return;

//     const int pitch = imax + 2;
//     const int id = j * pitch + i;

//     REAL_T Ai   = __ldg(&A[id]);
//     REAL_T newv = static_cast<REAL_T>(0.25) * ( __ldg(&A[id+1]) + __ldg(&A[id-1])
//                        + __ldg(&A[id-pitch]) + __ldg(&A[id+pitch]) );
//     Anew[id] = newv;

//     // Works for float/double/__half
//     err[id] = real_abs( real_sub(newv, Ai) );
// }

// int main(int argc, const char** argv)
// {
//   int jmax = 4096;
//   int imax = 4096;
//   int iter_max = 100;

//   int bx = 1024, by = 1;
//   for (int i = 1; i < argc; ++i) {
//     if (strncmp(argv[i], "--bx=", 5) == 0) bx = atoi(argv[i] + 5);
//     else if (strncmp(argv[i], "--by=", 5) == 0) by = atoi(argv[i] + 5);
//   }
//   if (bx <= 0 || by <= 0 || 1LL * bx * by > 1024) {
//     fprintf(stderr, "Error: invalid block dims. Require bx>0, by>0 and bx*by ≤ 1024.\n");
//     return 1;
//   }

//   // Use double for host math to avoid __half overload issues, cast at assignment
//   const double pi_d  = 2.0 * asin(1.0);
//   const double tol_d = 1.0e-6;

//   // We'll keep error as REAL_T (device reduction writes into it), but we avoid
//   // using host fabs() directly on REAL_T when REAL_T=__half.
//   REAL_T error = real_from_double(1.0);

//   // Host matrices
//   REAL_T *A    = new REAL_T[(size_t)(imax+2) * (jmax+2)];
//   REAL_T *Anew = new REAL_T[(size_t)(imax+2) * (jmax+2)];
//   memset(A, 0, (size_t)(imax+2) * (jmax+2) * sizeof(REAL_T));
//   memset(Anew, 0, (size_t)(imax+2) * (jmax+2) * sizeof(REAL_T));

//   // Top/bottom boundaries
//   for (int i = 0; i < imax+2; i++) A[(0)*(imax+2)+i]             = real_from_double(0.0);
//   for (int i = 0; i < imax+2; i++) A[(jmax+1)*(imax+2)+i]        = real_from_double(0.0);

//   // Left boundary
//   for (int j = 0; j < jmax+2; j++) {
//       double v = sin(pi_d * (double)j / (double)(jmax+1));
//       A[(j)*(imax+2)+0] = real_from_double(v);
//   }
//   // Right boundary
//   for (int j = 0; j < jmax+2; j++) {
//       double v = sin(pi_d * (double)j / (double)(jmax+1)) * exp(-pi_d);
//       A[(j)*(imax+2)+imax+1] = real_from_double(v);
//   }

//   printf("Jacobi relaxation Calculation: %d x %d mesh\n", imax+2, jmax+2);

//   // Initialize Anew boundaries similarly
//   for (int i = 1; i < imax+2; i++) Anew[(0)*(imax+2)+i]             = real_from_double(0.0);
//   for (int i = 1; i < imax+2; i++) Anew[(jmax+1)*(imax+2)+i]        = real_from_double(0.0);
//   for (int j = 1; j < jmax+2; j++) {
//       double v = sin(pi_d * (double)j / (double)(jmax+1));
//       Anew[(j)*(imax+2)+0] = real_from_double(v);
//   }
//   for (int j = 1; j < jmax+2; j++) {
//       double v = sin(pi_d * (double)j / (double)(jmax+1)) * exp(-pi_d);
//       Anew[(j)*(imax+2)+jmax+1] = real_from_double(v);
//   }

//   size_t N = (size_t)(imax+2) * (jmax+2);
//   REAL_T *d_A = nullptr, *d_Anew = nullptr, *d_err = nullptr;
//   cudaMalloc(&d_A,    N * sizeof(REAL_T));
//   cudaMalloc(&d_Anew, N * sizeof(REAL_T));
//   cudaMalloc(&d_err,  N * sizeof(REAL_T));

//   cudaMemcpy(d_A,    A,    N * sizeof(REAL_T), cudaMemcpyHostToDevice);
//   cudaMemcpy(d_Anew, Anew, N * sizeof(REAL_T), cudaMemcpyHostToDevice);

//   dim3 block(bx, by);
//   dim3 grid( (imax + block.x - 1) / block.x,
//              (jmax + block.y - 1) / block.y );

//   cudaMemset(d_err, 0, N * sizeof(REAL_T));

//   auto t1 = std::chrono::high_resolution_clock::now();

//   const int K = 5;
//   int k = 0, iter = 0;

//   // Stopping tolerance in device type
//   const REAL_T tol = real_from_double(tol_d);

//   while ( ( (double)error > tol_d ) && iter < iter_max) {
//       jacobi_step<<<grid, block>>>(imax, jmax, d_A, d_Anew, d_err);
//       std::swap(d_A, d_Anew);
//       ++iter; ++k;

//       if (k == K || iter == iter_max) {
//           error = thrust::reduce(thrust::device_pointer_cast(d_err),
//                                  thrust::device_pointer_cast(d_err + N),
//                                  real_from_double(0.0),
//                                  thrust::maximum<REAL_T>());
//           k = 0;
//       }
//   }

//   cudaDeviceSynchronize();
//   auto t2 = std::chrono::high_resolution_clock::now();

//   printf("%5d, %0.6f\n", iter, (double)error);

//   // Use double on host to compute and print the error difference safely
//   const double ref_err = 2.421354960840227e-03;
//   const double err_diff_d = std::fabs( (100.0 * ((double)error / ref_err)) - 100.0 );
//   printf("Total error is within %3.15E %% of the expected error\n", err_diff_d);
//   if (err_diff_d < 0.001) printf("This run is considered PASSED\n");
//   else                    printf("This test is considered FAILED\n");

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

//   // Append machine-readable SUMMARY to a text log
//   {
//       const char* LOG_PATH = "jacobi_log.txt";
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
//   // CSV output: bx,by,dtype,gflops,bandwidth_GBps,iters
//   // =====================================================
//   {
//       const char* CSV_PATH = "tune.csv";
//       struct stat stcsv{};
//       bool write_header = (stat(CSV_PATH, &stcsv) != 0) || (stcsv.st_size == 0);

//       std::ofstream csv(CSV_PATH, std::ios::app);
//       if (csv) {
//           if (write_header) {
//               csv << "bx,by,dtype,gflops,bandwidth_GBps,iters\n";
//           }
//           csv << bx << ","
//               << by << ","
//               << DTYPE << ","
//               << std::setprecision(10) << gflops << ","
//               << std::setprecision(10) << bandwidth_GBps << ","
//               << iter
//               << "\n";
//       }
//   }

//   cudaFree(d_A);
//   cudaFree(d_Anew);
//   cudaFree(d_err);
//   delete[] A;
//   delete[] Anew;

//   return 0;
// }

// #include <math.h>
// #include <string.h>
// #include <stdlib.h>
// #include <stdio.h>
// #include <chrono>
// #include <iostream>
// #include <fstream>
// #include <iomanip>
// #include <sys/stat.h>

// #include <cuda_runtime.h>
// #include <thrust/reduce.h>
// #include <thrust/device_ptr.h>

// #if defined(HAS_HALF)
//   #include <cuda_fp16.h>
// #endif

// // Base type
// #ifndef REAL_T
// #define REAL_T double
// #endif

// // Human-readable dtype string (set from the build with -DDTYPE_STR="\"fp64\"" etc.)
// #ifndef DTYPE_STR
// #define DTYPE_STR "fp64"
// #endif
// static constexpr const char* DTYPE = DTYPE_STR;

// //      ↑
// //  A[id-pitch]
// //        |
// // A[id-1] — A[id] — A[id+1]
// //        |
// //  A[id+pitch]
// //      ↓

// // Configuration metadata to make GLOBAL_BEST comparable
// static constexpr double FLOPS_PER_POINT = 6.0; // 4 adds + 1 mul + 1 sub
// static constexpr const char* POINTS_MODE = "padded"; // counting (imax+2)*(jmax+2)

// // -----------------------------------------------------------------------------
// // Helpers to make code work uniformly for double/float/__half
// // -----------------------------------------------------------------------------

// // Host-side: convert a double to REAL_T safely
// #if defined(HAS_HALF)
// __host__ __device__ inline __half real_from_double(double x) { return __double2half(x); }
// #else
// __host__ __device__ inline REAL_T real_from_double(double x) { return static_cast<REAL_T>(x); }
// #endif

// // Device-side: abs and subtract for REAL_T (handles __half properly)
// #if defined(HAS_HALF)
// __device__ __forceinline__ __half real_sub(__half a, __half b) { return __hsub(a, b); }
// __device__ __forceinline__ __half real_abs(__half x)           { return __habs(x); }
// #else
// template <typename T>
// __device__ __forceinline__ T real_sub(T a, T b) { return a - b; }

// template <typename T>
// __device__ __forceinline__ T real_abs(T x) { return fabs(x); }
// #endif

// // Error accumulation type: for FP16 we keep per-point error in float
// #if defined(HAS_HALF)
// using ERR_T = float;
// #else
// using ERR_T = REAL_T;
// #endif

// // Jacobi step using __ldg() for cached reads
// __global__ void jacobi_step(int imax, int jmax,
//                             const REAL_T* __restrict__ A,
//                             REAL_T* __restrict__ Anew,
//                             ERR_T*  __restrict__ err)   // NOTE: ERR_T here
// {
//     int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
//     int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
//     if (i > imax || j > jmax) return;

//     const int pitch = imax + 2;
//     const int id = j * pitch + i;

// #if defined(HAS_HALF)
//     // Promote to float for compute & error; store back to half
//     float Ai = __half2float(A[id]);
//     float sum =
//         __half2float(A[id+1]) + __half2float(A[id-1]) +
//         __half2float(A[id-pitch]) + __half2float(A[id+pitch]);
//     float newv_f = 0.25f * sum;
//     Anew[id] = __float2half(newv_f);
//     err[id]  = fabsf(newv_f - Ai);
// #else
//     // Native path for float/double
//     REAL_T Ai   = __ldg(&A[id]);
//     REAL_T newv = static_cast<REAL_T>(0.25) * ( __ldg(&A[id+1]) + __ldg(&A[id-1])
//                        + __ldg(&A[id-pitch]) + __ldg(&A[id+pitch]) );
//     Anew[id] = newv;
//     err[id]  = real_abs( real_sub(newv, Ai) );
// #endif
// }

// int main(int argc, const char** argv)
// {
//   int jmax = 4096;
//   int imax = 4096;
//   int iter_max = 100;

//   int bx = 1024, by = 1;
//   for (int i = 1; i < argc; ++i) {
//     if (strncmp(argv[i], "--bx=", 5) == 0) bx = atoi(argv[i] + 5);
//     else if (strncmp(argv[i], "--by=", 5) == 0) by = atoi(argv[i] + 5);
//   }
//   if (bx <= 0 || by <= 0 || 1LL * bx * by > 1024) {
//     fprintf(stderr, "Error: invalid block dims. Require bx>0, by>0 and bx*by ≤ 1024.\n");
//     return 1;
//   }

//   // === Query kernel attributes and occupancy-based suggestion ===
//   int suggested_block_threads = 0;
//   int min_grid_size = 0;
//   cudaError_t occErr = cudaOccupancyMaxPotentialBlockSize(
//       &min_grid_size, &suggested_block_threads,
//       jacobi_step,  /* kernel */
//       0,            /* dynamic smem per block */
//       0             /* no block-size limit */
//   );
//   if (occErr != cudaSuccess) {
//       fprintf(stderr, "cudaOccupancyMaxPotentialBlockSize failed: %s\n",
//               cudaGetErrorString(occErr));
//   }

//   // CUDA API: exact registers per thread, static smem
//   cudaFuncAttributes fattr{};
//   cudaError_t attrErr = cudaFuncGetAttributes(&fattr, jacobi_step);
//   if (attrErr != cudaSuccess) {
//       fprintf(stderr, "cudaFuncGetAttributes failed: %s\n",
//               cudaGetErrorString(attrErr));
//   }
//   const int num_regs_per_thread = fattr.numRegs;
//   const int static_smem_bytes   = fattr.sharedSizeBytes;

//   // Use double for host math to avoid __half overload issues, cast at assignment
//   const double pi_d =
//       2.0 * asin(1.0);  // pi with double accuracy for boundary setup
// #if defined(HAS_HALF)
//   const double tol_d = 5e-3;    // looser tolerance for FP16
// #else
//   const double tol_d = 1.0e-6;
// #endif

//   // We'll keep error as REAL_T (store the reduced value), computed from ERR_T
//   REAL_T error = real_from_double(1.0);

//   // Host matrices
//   REAL_T *A    = new REAL_T[(size_t)(imax+2) * (jmax+2)];
//   REAL_T *Anew = new REAL_T[(size_t)(imax+2) * (jmax+2)];
//   memset(A, 0, (size_t)(imax+2) * (jmax+2) * sizeof(REAL_T));
//   memset(Anew, 0, (size_t)(imax+2) * (jmax+2) * sizeof(REAL_T));

//   // Top/bottom boundaries
//   for (int i = 0; i < imax+2; i++) A[(0)*(imax+2)+i]             = real_from_double(0.0);
//   for (int i = 0; i < imax+2; i++) A[(jmax+1)*(imax+2)+i]        = real_from_double(0.0);

//   // Left boundary
//   for (int j = 0; j < jmax+2; j++) {
//       double v = sin(pi_d * (double)j / (double)(jmax+1));
//       A[(j)*(imax+2)+0] = real_from_double(v);
//   }
//   // Right boundary
//   for (int j = 0; j < jmax+2; j++) {
//       double v = sin(pi_d * (double)j / (double)(jmax+1)) * exp(-pi_d);
//       A[(j)*(imax+2)+imax+1] = real_from_double(v);
//   }

//   printf("Jacobi relaxation Calculation: %d x %d mesh\n", imax+2, jmax+2);
//   printf("Suggested block size (threads): %d; Min grid size: %d; Kernel registers/thread: %d; Static smem: %d B\n",
//          suggested_block_threads, min_grid_size, num_regs_per_thread, static_smem_bytes);

//   // Initialize Anew boundaries similarly
//   for (int i = 1; i < imax+2; i++) Anew[(0)*(imax+2)+i]             = real_from_double(0.0);
//   for (int i = 1; i < imax+2; i++) Anew[(jmax+1)*(imax+2)+i]        = real_from_double(0.0);
//   for (int j = 1; j < jmax+2; j++) {
//       double v = sin(pi_d * (double)j / (double)(jmax+1));
//       Anew[(j)*(imax+2)+0] = real_from_double(v);
//   }
//   for (int j = 1; j < jmax+2; j++) {
//       double v = sin(pi_d * (double)j / (double)(jmax+1)) * exp(-pi_d);
//       Anew[(j)*(imax+2)+jmax+1] = real_from_double(v);
//   }

//   size_t N = (size_t)(imax+2) * (jmax+2);
//   REAL_T *d_A = nullptr, *d_Anew = nullptr;
//   ERR_T  *d_err = nullptr;  // NOTE: ERR_T here
//   cudaMalloc(&d_A,    N * sizeof(REAL_T));
//   cudaMalloc(&d_Anew, N * sizeof(REAL_T));
//   cudaMalloc(&d_err,  N * sizeof(ERR_T));

//   cudaMemcpy(d_A,    A,    N * sizeof(REAL_T), cudaMemcpyHostToDevice);
//   cudaMemcpy(d_Anew, Anew, N * sizeof(REAL_T), cudaMemcpyHostToDevice);

//   dim3 block(bx, by);
//   dim3 grid( (imax + block.x - 1) / block.x,
//              (jmax + block.y - 1) / block.y );

//   cudaMemset(d_err, 0, N * sizeof(ERR_T));

//   auto t1 = std::chrono::high_resolution_clock::now();

//   const int K = 5;
//   int k = 0, iter = 0;

//   // Stopping tolerance in device type
//   const REAL_T tol = real_from_double(tol_d);

//   while ( ( (double)error > tol_d ) && iter < iter_max) {
//       jacobi_step<<<grid, block>>>(imax, jmax, d_A, d_Anew, d_err);
//       std::swap(d_A, d_Anew);
//       ++iter; ++k;

//       if (k == K || iter == iter_max) {
//           // Reduce max error in ERR_T, then store into REAL_T 'error'
//           ERR_T max_err = thrust::reduce(
//               thrust::device_pointer_cast(d_err),
//               thrust::device_pointer_cast(d_err + N),
//               (ERR_T)0,
//               thrust::maximum<ERR_T>());
//           error = real_from_double((double)max_err);
//           k = 0;
//       }
//   }

//   cudaDeviceSynchronize();
//   auto t2 = std::chrono::high_resolution_clock::now();

//   printf("%5d, %0.6f\n", iter, (double)error);

//   // Reference error check (dtype-aware threshold)
//   double ref_err = 2.421354960840227e-03; // default (double baseline)
//   #if defined(HAS_HALF)
//     // Load an FP16 baseline from env or use a safer fallback
//     const char* env_ref = std::getenv("REF_ERR_FP16");
//     if (env_ref) ref_err = atof(env_ref);
//     else         ref_err = 5.0e-3; // conservative fallback for FP16
//   #endif

//   const double err_diff_d = std::fabs( (100.0 * ((double)error / ref_err)) - 100.0 );
// #if defined(HAS_HALF)
//   double pass_threshold_pct = 1.0;      // 1% for FP16 (tune as needed)
// #else
//   double pass_threshold_pct = 0.001;    // 0.001% for FP32/64
// #endif
//   printf("Total error is within %3.15E %% of the expected error\n", err_diff_d);
//   if (err_diff_d < pass_threshold_pct) printf("This run is considered PASSED\n");
//   else                                 printf("This test is considered FAILED\n");

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
//   printf("SUMMARY imax=%d jmax=%d dtype=%s flops_per_pt=%.0f points=%s bx=%d by=%d time_ms=%.3f iters=%d gflops=%.6f bandwidth_GBps=%.6f avg_iter_ms=%.6f suggested_block_threads=%d num_regs=%d min_grid_size=%d static_smem_B=%d\n",
//          imax, jmax, DTYPE, FLOPS_PER_POINT, POINTS_MODE,
//          bx, by, time_ms, iter, gflops, bandwidth_GBps, avg_iter_ms,
//          suggested_block_threads, num_regs_per_thread, min_grid_size, static_smem_bytes);

//   std::cout << "--------------------------------------------------------------------------------";

//   // Append machine-readable SUMMARY to a text log
//   {
//       const char* LOG_PATH = "jacobi_log.txt";
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
//               << "avg_iter_ms=" << avg_iter_ms << " "
//               << "suggested_block_threads=" << suggested_block_threads << " "
//               << "num_regs=" << num_regs_per_thread << " "
//               << "min_grid_size=" << min_grid_size << " "
//               << "static_smem_B=" << static_smem_bytes
//               << "\n";
//       }
//   }

//   // =====================================================
//   // CSV output: bx,by,dtype,gflops,bandwidth_GBps,iters,suggested_block_threads,num_regs
//   // =====================================================
//   {
//       const char* CSV_PATH = "tune.csv";
//       struct stat stcsv{};
//       bool write_header = (stat(CSV_PATH, &stcsv) != 0) || (stcsv.st_size == 0);

//       std::ofstream csv(CSV_PATH, std::ios::app);
//       if (csv) {
//           if (write_header) {
//               csv << "bx,by,dtype,gflops,bandwidth_GBps,iters,suggested_block_threads,num_regs,min_grid_size,static_smem_B\n";
//           }
//           csv << bx << ","
//               << by << ","
//               << DTYPE << ","
//               << std::setprecision(10) << gflops << ","
//               << std::setprecision(10) << bandwidth_GBps << ","
//               << iter << ","
//               << suggested_block_threads << ","
//               << num_regs_per_thread << ","
//               << min_grid_size << ","
//               << static_smem_bytes
//               << "\n";
//       }
//   }

//   cudaFree(d_A);
//   cudaFree(d_Anew);
//   cudaFree(d_err);
//   delete[] A;
//   delete[] Anew;

//   return 0;
// }


#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sys/stat.h>
#include <sstream>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>

#if defined(HAS_HALF)
  #include <cuda_fp16.h>
#endif

// ====== Bloque fijo opcional (compile-time) ======
#ifndef BX_DEF
#define BX_DEF 256
#endif
#ifndef BY_DEF
#define BY_DEF 2
#endif
#ifndef USE_FIXED_BLOCK
#define USE_FIXED_BLOCK 0
#endif

// ====== Tipo base ======
#ifndef REAL_T
#define REAL_T double
#endif

#ifndef DTYPE_STR
#define DTYPE_STR "fp64"
#endif
static constexpr const char* DTYPE = DTYPE_STR;

// ====== Métricas ======
static constexpr double FLOPS_PER_POINT = 6.0;
static constexpr const char* POINTS_MODE = "padded";   // (imax+2)*(jmax+2)

// ====== helpers dtype ======
#if defined(HAS_HALF)
__host__ __device__ inline __half real_from_double(double x) { return __double2half(x); }
#else
__host__ __device__ inline REAL_T real_from_double(double x) { return static_cast<REAL_T>(x); }
#endif

#if defined(HAS_HALF)
__device__ __forceinline__ __half real_sub(__half a, __half b) { return __hsub(a, b); }
__device__ __forceinline__ __half real_abs(__half x)           { return __habs(x); }
#else
template <typename T>
__device__ __forceinline__ T real_sub(T a, T b) { return a - b; }
template <typename T>
__device__ __forceinline__ T real_abs(T x) { return fabs(x); }
#endif

#if defined(HAS_HALF)
using ERR_T = float;
#else
using ERR_T = REAL_T;
#endif

// ====== kernel ======
__global__ void jacobi_step(int imax, int jmax,
                            const REAL_T* __restrict__ A,
                            REAL_T* __restrict__ Anew,
                            ERR_T*  __restrict__ err)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i > imax || j > jmax) return;

    const int pitch = imax + 2;
    const int id = j * pitch + i;

#if defined(HAS_HALF)
    float Ai = __half2float(A[id]);
    float sum =
        __half2float(A[id+1]) + __half2float(A[id-1]) +
        __half2float(A[id-pitch]) + __half2float(A[id+pitch]);
    float newv_f = 0.25f * sum;
    Anew[id] = __float2half(newv_f);
    err[id]  = fabsf(newv_f - Ai);
#else
    REAL_T Ai   = __ldg(&A[id]);
    REAL_T newv = static_cast<REAL_T>(0.25) * ( __ldg(&A[id+1]) + __ldg(&A[id-1])
                       + __ldg(&A[id-pitch]) + __ldg(&A[id+pitch]) );
    Anew[id] = newv;
    err[id]  = real_abs( real_sub(newv, Ai) );
#endif
}

// ====== main ======
int main(int argc, const char** argv)
{
  int jmax = 4096;
  int imax = 4096;
  int iter_max = 100;

  int bx = 1024, by = 1;
  const char* bx_list_arg = nullptr;
  const char* by_list_arg = nullptr;

#if USE_FIXED_BLOCK
  (void)argc; (void)argv;
  bx = BX_DEF; by = BY_DEF;
#else
  for (int i = 1; i < argc; ++i) {
    if      (strncmp(argv[i], "--bx=", 5) == 0)        bx = atoi(argv[i] + 5);
    else if (strncmp(argv[i], "--by=", 5) == 0)        by = atoi(argv[i] + 5);
    else if (strncmp(argv[i], "--bx-list=", 10) == 0)  bx_list_arg = argv[i] + 10;
    else if (strncmp(argv[i], "--by-list=", 10) == 0)  by_list_arg = argv[i] + 10;
  }
#endif

  auto split_csv = [](const char* s){
      std::vector<int> v;
      if (!s) return v;
      std::stringstream ss(s); std::string tok;
      while (std::getline(ss, tok, ',')) if (!tok.empty()) v.push_back(std::stoi(tok));
      return v;
  };
  std::vector<int> bx_list = split_csv(bx_list_arg);
  std::vector<int> by_list = split_csv(by_list_arg);
  bool do_sweep = (!bx_list.empty() && !by_list.empty());

  // Ocupancia y atributos
  int suggested_block_threads = 0;
  int min_grid_size = 0;
  cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &suggested_block_threads, jacobi_step, 0, 0);
  cudaFuncAttributes fattr{};
  cudaFuncGetAttributes(&fattr, jacobi_step);
  const int num_regs_per_thread = fattr.numRegs;
  const int static_smem_bytes   = fattr.sharedSizeBytes;

  // Setup frontera
  const double pi_d  = 2.0 * asin(1.0);
  const double tol_d = 1.0e-6;

  REAL_T *A    = new REAL_T[(size_t)(imax+2) * (jmax+2)];
  REAL_T *Anew = new REAL_T[(size_t)(imax+2) * (jmax+2)];
  memset(A,    0, (size_t)(imax+2) * (jmax+2) * sizeof(REAL_T));
  memset(Anew, 0, (size_t)(imax+2) * (jmax+2) * sizeof(REAL_T));

  for (int i = 0; i < imax+2; i++) {
      A[(0)*(imax+2)+i]           = real_from_double(0.0);
      A[(jmax+1)*(imax+2)+i]      = real_from_double(0.0);
  }
  for (int j = 0; j < jmax+2; j++) {
      double vL = sin(pi_d * j / (double)(jmax+1));
      double vR = vL * exp(-pi_d);
      A[j*(imax+2)+0]      = real_from_double(vL);
      A[j*(imax+2)+imax+1] = real_from_double(vR);
  }
  memcpy(Anew, A, (size_t)(imax+2)*(jmax+2)*sizeof(REAL_T));

  printf("Jacobi relaxation Calculation: %d x %d mesh\n", imax+2, jmax+2);
  printf("Suggested block size (threads): %d; Min grid size: %d; Kernel registers/thread: %d; Static smem: %d B\n",
         suggested_block_threads, min_grid_size, num_regs_per_thread, static_smem_bytes);

  // Memoria en device
  size_t N = (size_t)(imax+2)*(jmax+2);
  REAL_T *d_A, *d_Anew, *d_A_init, *d_Anew_init;
  ERR_T *d_err;
  cudaMalloc(&d_A, N*sizeof(REAL_T));
  cudaMalloc(&d_Anew, N*sizeof(REAL_T));
  cudaMalloc(&d_A_init, N*sizeof(REAL_T));
  cudaMalloc(&d_Anew_init, N*sizeof(REAL_T));
  cudaMalloc(&d_err, N*sizeof(ERR_T));

  cudaMemcpy(d_A, A, N*sizeof(REAL_T), cudaMemcpyHostToDevice);
  cudaMemcpy(d_Anew, Anew, N*sizeof(REAL_T), cudaMemcpyHostToDevice);
  cudaMemcpy(d_A_init, d_A, N*sizeof(REAL_T), cudaMemcpyDeviceToDevice);
  cudaMemcpy(d_Anew_init, d_Anew, N*sizeof(REAL_T), cudaMemcpyDeviceToDevice);

  struct Result { int bx, by; double gflops, bw, time_ms; };
  std::vector<Result> results;

  auto run_once = [&](int bx_, int by_) {
      if (bx_<=0 || by_<=0 || 1LL*bx_*by_>1024) return;
      cudaMemcpy(d_A, d_A_init, N*sizeof(REAL_T), cudaMemcpyDeviceToDevice);
      cudaMemcpy(d_Anew, d_Anew_init, N*sizeof(REAL_T), cudaMemcpyDeviceToDevice);
      cudaMemset(d_err,0,N*sizeof(ERR_T));

      dim3 block(bx_,by_);
      dim3 grid((imax+block.x-1)/block.x,(jmax+block.y-1)/block.y);

      REAL_T error = real_from_double(1.0);
      int iter=0,k=0;
      auto t0=std::chrono::high_resolution_clock::now();
      while((double)error>tol_d && iter<100){
          jacobi_step<<<grid,block>>>(imax,jmax,d_A,d_Anew,d_err);
          std::swap(d_A,d_Anew); ++iter;++k;
          if(k==5||iter==100){
              ERR_T max_err=thrust::reduce(thrust::device_pointer_cast(d_err),
                                           thrust::device_pointer_cast(d_err+N),
                                           (ERR_T)0,thrust::maximum<ERR_T>());
              error=real_from_double((double)max_err);k=0;
          }
      }
      cudaDeviceSynchronize();
      auto t1=std::chrono::high_resolution_clock::now();
      double time_ms=std::chrono::duration<double,std::milli>(t1-t0).count();
      double points=(imax+2.0)*(jmax+2.0);
      double iters=iter,sec=time_ms/1000.0;
      double gflops=FLOPS_PER_POINT*points*iters/(sec*1e9);
      double BW=5.0*sizeof(REAL_T)*points*iters/(sec*1e9);
      results.push_back({bx_,by_,gflops,BW,time_ms});
      printf("SUMMARY bx=%d by=%d gflops=%.3f BW=%.3f GB/s time_ms=%.2f\n",bx_,by_,gflops,BW,time_ms);

      // log
      std::ofstream out("jacobi_log.txt",std::ios::app);
      if(out) out<<"SUMMARY bx="<<bx_<<" by="<<by_<<" gflops="<<gflops<<" BW="<<BW<<" time_ms="<<time_ms<<"\n";
  };

  if(do_sweep){
      for(int bx_:bx_list) for(int by_:by_list)
          if(1LL*bx_*by_<=1024 && bx_>0 && by_>0) run_once(bx_,by_);
  }else run_once(bx,by);

  if(!results.empty()){
      auto best=results[0];
      for(auto&r:results) if(r.gflops>best.gflops) best=r;
      printf("\nGLOBAL_BEST bx=%d by=%d gflops=%.3f BW=%.3f time_ms=%.2f\n",
             best.bx,best.by,best.gflops,best.bw,best.time_ms);
      std::ofstream out("jacobi_log.txt",std::ios::app);
      if(out) out<<"GLOBAL_BEST bx="<<best.bx<<" by="<<best.by<<" gflops="<<best.gflops<<" BW="<<best.bw<<" time_ms="<<best.time_ms<<"\n";
  }

#if USE_FIXED_BLOCK
  printf("Using FIXED block compile-time bx=%d by=%d\n",bx,by);
#endif
}
