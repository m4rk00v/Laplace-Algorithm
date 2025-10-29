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
#include <limits>
#include <functional>
#include <algorithm>

#include <cuda_runtime.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>

#if defined(HAS_HALF)
  #include <cuda_fp16.h>
#endif

// ====== Optional fixed block (compile-time) ======
#ifndef BX_DEF
#define BX_DEF 256
#endif
#ifndef BY_DEF
#define BY_DEF 2
#endif
#ifndef USE_FIXED_BLOCK
#define USE_FIXED_BLOCK 0
#endif

// ====== Base type ======
#ifndef REAL_T
#define REAL_T double
#endif

#ifndef DTYPE_STR
#define DTYPE_STR "fp64"
#endif

#define KERNEL_TAG /* #[kernel] */

static constexpr const char* DTYPE = DTYPE_STR;

// ====== Metrics ======
static constexpr double FLOPS_PER_POINT = 6.0;         // Jacobi 5-pt stencil
static constexpr const char* POINTS_MODE = "padded";   // (imax+2)*(jmax+2)

// ====== dtype helpers ======
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
KERNEL_TAG
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

// ====== helpers ======
static inline bool is_valid_block(int bx, int by){
    long long t = 1LL * bx * by;
    if (bx <= 0 || by <= 0) return false;
    if (t > 1024) return false;
    if ((t % 32) != 0) return false;   // full warps
    return true;
}

int main(int argc, const char** argv)
{
  // Domain / iterations
  int jmax = 4096;
  int imax = 4096;
  int iter_max = 100;

  // CLI
  int bx = 1024, by = 1;
  int check_every = 5;
  const char* bx_list_arg = nullptr;
  const char* by_list_arg = nullptr;

#if USE_FIXED_BLOCK
  (void)argc; (void)argv;
  bx = BX_DEF; by = BY_DEF;
#else
  for (int i = 1; i < argc; ++i) {
    if      (strncmp(argv[i], "--bx=", 5) == 0)            bx = atoi(argv[i] + 5);
    else if (strncmp(argv[i], "--by=", 5) == 0)            by = atoi(argv[i] + 5);
    else if (strncmp(argv[i], "--bx-list=", 10) == 0)      bx_list_arg = argv[i] + 10;
    else if (strncmp(argv[i], "--by-list=", 10) == 0)      by_list_arg = argv[i] + 10;
    else if (strncmp(argv[i], "--check-every=", 14) == 0)  check_every = atoi(argv[i] + 14);
  }
#endif

  // Parse CSV lists
  auto split_csv = [](const char* s){
      std::vector<int> v;
      if (!s) return v;
      std::stringstream ss(s); std::string tok;
      while (std::getline(ss, tok, ',')) if (!tok.empty()) v.push_back(std::stoi(tok));
      return v;
  };
  std::vector<int> bx_list = split_csv(bx_list_arg);
  std::vector<int> by_list = split_csv(by_list_arg);

  // Validate compile-time block if fixed
#if USE_FIXED_BLOCK
  if (!is_valid_block(bx, by)) {
      fprintf(stderr,
              "ERROR: compile-time block (%d,%d) invalid. "
              "Must satisfy bx>0,by>0, bx*by<=1024 and (bx*by)%%32==0.\n", bx, by);
      return 1;
  }
#endif

  // Occupancy & attributes (informational)
  int suggested_block_threads = 0;
  int min_grid_size = 0;
  cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &suggested_block_threads, jacobi_step, 0, 0);
  cudaFuncAttributes fattr{};
  cudaFuncGetAttributes(&fattr, jacobi_step);
  const int num_regs_per_thread = fattr.numRegs;
  const int static_smem_bytes   = fattr.sharedSizeBytes;

  // Boundary setup
  const double pi_d  = 2.0 * asin(1.0);
  const double tol_d = 2.5e-3;

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

  // Device memory
  size_t N = (size_t)(imax+2)*(jmax+2);
  REAL_T *d_A, *d_Anew;
  ERR_T *d_err;
  cudaMalloc(&d_A, N*sizeof(REAL_T));
  cudaMalloc(&d_Anew, N*sizeof(REAL_T));
  cudaMalloc(&d_err, N*sizeof(ERR_T));

  cudaMemcpy(d_A, A, N*sizeof(REAL_T), cudaMemcpyHostToDevice);
  cudaMemcpy(d_Anew, Anew, N*sizeof(REAL_T), cudaMemcpyHostToDevice);
  cudaMemset(d_err, 0, N*sizeof(ERR_T));

  // ====== Build candidate list (multiple-of-32 total threads) ======
  std::vector<std::pair<int,int>> cand;
  if (!bx_list.empty() && !by_list.empty()){
      for (int bx_ : bx_list)
          for (int by_ : by_list)
              if (is_valid_block(bx_, by_)) cand.push_back({bx_, by_});
  } else {
      if (is_valid_block(bx, by)) cand.push_back({bx, by});
      else cand.push_back({256, 4}); // fallback = 1024 threads
  }
  if (cand.empty()) cand.push_back({256, 4});

  // ====== Online tuning across first |cand| iterations (exactly 1 kernel per iteration) ======
  struct Acc { double time_ms=0.0; double gflops=0.0; double bw=0.0; };
  std::vector<Acc> acc(cand.size(), Acc{});

  int cur_cand = 0;
  bool tuned = (cand.size() == 1);
  int best_i = 0;

  // Start with first candidate
  int cur_bx = cand[0].first;
  int cur_by = cand[0].second;
  dim3 block(cur_bx, cur_by);
  dim3 grid((imax+block.x-1)/block.x, (jmax+block.y-1)/block.y);

  REAL_T error_val = real_from_double(1.0);
  int iter=0, k=0;
  bool printed_e0 = false;

  std::vector<double> err_samples_iter;
  std::vector<double> err_samples_value;

  auto t0=std::chrono::high_resolution_clock::now();
  while ((double)error_val > tol_d && iter < iter_max) {

      // If still tuning, set block from current candidate
      if (!tuned) {
        cur_bx = cand[cur_cand].first;
        cur_by = cand[cur_cand].second;
        block  = dim3(cur_bx, cur_by);
        grid   = dim3((imax+block.x-1)/block.x, (jmax+block.y-1)/block.y);
      }

      // ---- Exactly ONE kernel launch this iteration; measure it with events ----
      cudaEvent_t ev_start, ev_stop;
      cudaEventCreate(&ev_start);
      cudaEventCreate(&ev_stop);
      cudaEventRecord(ev_start, 0);

      jacobi_step<<<grid, block>>>(imax, jmax, d_A, d_Anew, d_err);
      std::swap(d_A, d_Anew);

      cudaEventRecord(ev_stop, 0);
      cudaEventSynchronize(ev_stop);
      float iter_ms = 0.f;
      cudaEventElapsedTime(&iter_ms, ev_start, ev_stop);
      cudaEventDestroy(ev_start);
      cudaEventDestroy(ev_stop);

      ++iter; ++k;

      // If tuning, record performance for this candidate (based on this single iteration)
      if (!tuned) {
          double points = (imax + 2.0) * (jmax + 2.0);
          double sec    = iter_ms / 1000.0;
          double gflops = FLOPS_PER_POINT * points / (sec * 1e9);
          double BW     = 5.0 * sizeof(REAL_T) * points / (sec * 1e9);
          acc[cur_cand].time_ms = iter_ms;
          acc[cur_cand].gflops  = gflops;
          acc[cur_cand].bw      = BW;

          printf("TUNE iter=%d  cand=%d bx=%d by=%d  time_ms=%.3f  gflops=%.3f  BW=%.3f GB/s\n",
                 iter, cur_cand, cur_bx, cur_by, iter_ms, gflops, BW);

          // Advance candidate and finalize if done
          ++cur_cand;
          if (cur_cand >= (int)cand.size()) {
              // Select best by GFLOPS, tie by lower time
              best_i = 0;
              for (int i=1; i<(int)cand.size(); ++i) {
                  if (acc[i].gflops > acc[best_i].gflops ||
                      (fabs(acc[i].gflops - acc[best_i].gflops) < 1e-9 &&
                       acc[i].time_ms < acc[best_i].time_ms)) best_i = i;
              }
              cur_bx = cand[best_i].first;
              cur_by = cand[best_i].second;
              block  = dim3(cur_bx, cur_by);
              grid   = dim3((imax+block.x-1)/block.x, (jmax+block.y-1)/block.y);
              tuned  = true;

              printf("\nGLOBAL_BEST (online) bx=%d by=%d  gflops=%.3f  BW=%.3f GB/s  time_ms=%.3f\n\n",
                     cur_bx, cur_by, acc[best_i].gflops, acc[best_i].bw, acc[best_i].time_ms);
          }
      }

      // Print "0, ..." once after the first iteration (same behavior as original)
      if (iter == 1 && !printed_e0){
          ERR_T e0 = thrust::reduce(thrust::device_pointer_cast(d_err),
                                    thrust::device_pointer_cast(d_err+N),
                                    (ERR_T)0, thrust::maximum<ERR_T>());
          printf("    %d, %.6f\n", 0, (double)e0);
          printed_e0 = true;
      }

      // Error check cadence (unchanged)
      if (k == check_every || iter == iter_max){
          ERR_T max_err = thrust::reduce(thrust::device_pointer_cast(d_err),
                                         thrust::device_pointer_cast(d_err+N),
                                         (ERR_T)0, thrust::maximum<ERR_T>());
          error_val = real_from_double((double)max_err);
          k = 0;

          err_samples_iter.push_back((double)iter);
          err_samples_value.push_back((double)error_val);

          if (iter % 10 == 0 || iter == iter_max){
              printf("%5d, %.6f\n", iter, (double)error_val);
          }
      }
  }


  cudaDeviceSynchronize();
  auto t1=std::chrono::high_resolution_clock::now();
  double time_ms=std::chrono::duration<double,std::milli>(t1-t0).count();

  // Final metrics (include tuning iterations)
  double points=(imax+2.0)*(jmax+2.0);
  double gflops=FLOPS_PER_POINT*points*iter/((time_ms/1000.0)*1e9);
  double BW=5.0*sizeof(REAL_T)*points*iter/((time_ms/1000.0)*1e9);

  printf("FINAL bx=%d by=%d iters=%d gflops=%.3f BW=%.3f GB/s time_ms=%.2f\n",
         cur_bx, cur_by, iter, gflops, BW, time_ms);

  bool converged = ((double)error_val <= tol_d);

  // Optional: expected error via log-linear fit ln(err) = a + b*iter
  auto fit_log_linear = [&](const std::vector<double>& xs, const std::vector<double>& ys, double& a, double& b){
      size_t n = xs.size();
      if (n < 2) { a = std::log(ys.empty()?1.0:ys.back()); b = 0.0; return false; }
      double Sx=0, Sy=0, Sxx=0, Sxy=0;
      for (size_t i=0;i<n;++i){
          double x = xs[i];
          double ly = std::log(std::max(ys[i], 1e-300));
          Sx += x; Sy += ly; Sxx += x*x; Sxy += x*ly;
      }
      double denom = n*Sxx - Sx*Sx;
      if (fabs(denom) < 1e-30) { a = Sy/n; b = 0.0; return false; }
      b = (n*Sxy - Sx*Sy)/denom;
      a = (Sy - b*Sx)/n;
      return true;
  };

  double a_fit=0.0, b_fit=0.0;
  bool ok_fit = fit_log_linear(err_samples_iter, err_samples_value, a_fit, b_fit);
  double expected_final = 0.0;
  if (ok_fit){
      expected_final = std::exp(a_fit + b_fit * (double)iter);
  } else {
      expected_final = (err_samples_value.empty()? (double)error_val : err_samples_value.back());
  }

  double percent_of_expected = (expected_final > 0.0)
                             ? 100.0 * fabs((double)error_val - expected_final) / expected_final
                             : 0.0;

  printf("Total error is within %.15E %% of the expected error\n", percent_of_expected);
  printf("This run is considered %s\n", converged ? "PASSED" : "FAILED");

  // Log FINAL + CONVERGENCE
  {
      std::ofstream out("jacobi_log.txt",std::ios::app);
      if(out){
          out<<std::fixed<<std::setprecision(3)
             <<"FINAL bx="<<cur_bx<<" by="<<cur_by<<" iters="<<iter
             <<" gflops="<<gflops<<" BW="<<BW<<" time_ms="<<time_ms<<"\n";
          out<<std::setprecision(9)
             <<"expected_final="<<expected_final<<" error_final="<<(double)error_val<<"\n";
          out<<std::setprecision(15)
             <<"Total error is within "<< (percent_of_expected) <<" % of the expected error\n";
          out<<"This run is considered "<< (converged ? "PASSED" : "FAILED") <<"\n";
      }
  }

#if USE_FIXED_BLOCK
  printf("Using FIXED block (compile-time) bx=%d by=%d\n", cur_bx, cur_by);
#endif

  // Cleanup
  cudaFree(d_A); cudaFree(d_Anew); cudaFree(d_err);
  delete[] A; delete[] Anew;

  return 0;
}
