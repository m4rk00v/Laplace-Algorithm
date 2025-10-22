#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <chrono>
#include <iostream>


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
  auto t1 = std::chrono::high_resolution_clock::now();
  while ( error > tol && iter < iter_max )
  {
    error = 0.0;
    #pragma omp parallel for reduction(max:error)
    for( int j = 1; j < jmax+1; j++ )
    {
      for( int i = 1; i < imax+1; i++)
      {
        Anew[(j)*(imax+2)+i] = 0.25f * ( A[(j)*(imax+2)+i+1] + A[(j)*(imax+2)+i-1]
            + A[(j-1)*(imax+2)+i] + A[(j+1)*(imax+2)+i]);
        error = fmax( error, fabs(Anew[(j)*(imax+2)+i]-A[(j)*(imax+2)+i]));
      }
    }
    #pragma omp parallel for
    for( int j = 1; j < jmax+1; j++ )
    {
      for( int i = 1; i < imax+1; i++)
      {
        A[(j)*(imax+2)+i] = Anew[(j)*(imax+2)+i];
      }
    }
    if(iter % 10 == 0) printf("%5d, %0.6f\n", iter, error);
    iter++;
  }
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
  const double FLOPS_PER_POINT = 5.0; // usa 6.0 si quieres contar fabs también
  const double points = static_cast<double>(imax) * static_cast<double>(jmax); // interior
  const double iters  = static_cast<double>(iter);
  const double seconds = ms_double.count() / 1000.0;

  const double total_flops = FLOPS_PER_POINT * points * iters;
  const double gflops = total_flops / (seconds * 1e9);

  std::cout << "GFLOPS/s ("
            << FLOPS_PER_POINT << " flops/pt): "
            << gflops << std::endl;


  // BADNWITH

    const bool COUNT_ERROR_OPS = true; // set to false if you don't want to count them
    const int adds  = 3;               // summing 4 values → 3 additions
    const int muls  = 1;               // multiply by 0.25
    const int subs  = COUNT_ERROR_OPS ? 1 : 0; // Anew - A
    const int abss  = COUNT_ERROR_OPS ? 1 : 0; // fabs
    const int maxop = COUNT_ERROR_OPS ? 1 : 0; // fmax

    const double BYTES_PER_POINT = 32 * sizeof(double); // = 64 bytes for doubles
    const double total_bytes = BYTES_PER_POINT * points * iters;
    const double bandwidth_GBps = total_bytes / (seconds * 1e9);

    std::cout << "Bandwidth: " << bandwidth_GBps << " GB/s" << std::endl;


  return 0;
}
