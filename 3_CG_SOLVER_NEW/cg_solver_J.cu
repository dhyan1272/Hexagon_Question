#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <fstream>
#include <string>
#include <chrono>
#include <cuda.h>

#define threadsPerBlock 128

template <typename T>
std::vector<T> load_csv(std::string path) {
  std::vector<T> data;
  std::ifstream file(path);
  std::string line;
  while (std::getline(file, line)) {
    if(!line.empty()) data.push_back(std::stod(line));
  }
  return data;
}

// Sparse Matrix-Vector Multiplication: y = A * x
__global__ void spmv_kernel(int N, const int* rowptr, const int* col, const double* val, const double* x, double* y) {
  
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  double sum = 0.0;
  for (int k = rowptr[i]; k < rowptr[i+1]; ++k) {
    sum += val[k] * x[col[k]];
  }
  y[i] = sum;
}

// Jacobi Preconditioner: z = invM * r
__global__ void apply_jacobi_kernel(int N, const double* invM, const double* r, double* z) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  z[i] = invM[i] * r[i];
}

// Vector-Vector Ops: x = x + alpha * d  AND r = r - alpha * Ad
__global__ void update_x_r_kernel(int N, double alpha, const double* d, const double* Ad, double* x, double* r) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
    x[i] += alpha * d[i];
    r[i] -= alpha * Ad[i];
}

// Update Direction: d = z + beta * d
__global__ void update_direction_kernel(int N, double beta, const double* z, double* d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  d[i] = z[i] + beta * d[i];
}


// Dot Product with Warp Reduction (Partial Sums)
__global__ void dot_product_kernel(int N, const double* a, const double* b, double* result) {
 
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ double shared_data[threadsPerBlock];
  int tid = threadIdx.x;

  shared_data[tid] = (i<N) ? a[i] * b[i]: 0.0;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) shared_data[tid] += shared_data[tid + s];
    __syncthreads();
  }
  if (tid == 0) atomicAdd(result, shared_data[0]);
}

double dot_product(const std::vector<double>& a, const std::vector<double>& b) {
 double res = 0.0;
 for (size_t i = 0; i < a.size(); ++i) res += a[i] * b[i];
 return res;
}



int main(int argc, char* argv[]) {

  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <directory_path>" << std::endl;
    exit(-1);
  }

  std::string dir = argv[1];
  //Convergence parameters
  double tolerance  = 1e-8;
  int iterations = 1000;
 
  //ReadInputs & Calculate N
  std::vector<int> rowptr = load_csv<int>(dir + "rowptr.csv");
  std::vector<int> col = load_csv<int>(dir + "col.csv");
  std::vector<double> val = load_csv<double>(dir + "val.csv");
  int N = rowptr.size() - 1;
  
  //GPU kernel launch parameters
  int nBlocks = (N + threadsPerBlock - 1)/threadsPerBlock;
 
  //Device Pointers
  int* device_rowptr, *device_col;
  double* device_val, *device_x, *device_Adir, *device_invM, *device_r, *device_z, *device_dir,
          *device_rz, *device_dirAdir, *device_rnorm;
  
  //Memory Allocation
  cudaMalloc(&device_rowptr, (N + 1) * sizeof(int));
  cudaMalloc(&device_col, col.size() * sizeof(int));
  cudaMalloc(&device_val, val.size() * sizeof(double));
  cudaMalloc(&device_x, N*sizeof(double));
  cudaMalloc(&device_Adir, N*sizeof(double));    //Matrix Vector multiplication
  std::vector<double> b (N, 1.0);              //RHS
  cudaMalloc(&device_invM, N*sizeof(double));
  cudaMalloc(&device_r, N*sizeof(double));   //Residula 
  cudaMalloc(&device_z, N*sizeof(double));   //Preconditoned residual
  cudaMalloc(&device_dir, N*sizeof(double));   //Search direction
  cudaMalloc(&device_rz, sizeof(double));   //Search direction 
  cudaMalloc(&device_dirAdir, sizeof(double));   //Search direction 
  cudaMalloc(&device_rnorm, sizeof(double));   //Search direction 
  
  //Extract Diagonal for Jacobi Preconditioner (Done in CPU as done once)
  std::vector<double> invM(N);
  for (int i = 0; i < N; ++i) {
    for (int j = rowptr[i]; j < rowptr[i + 1]; ++j) {
      if (col[j] == i) {
        invM[i] = 1.0 / val[j];
        break;
      }
    }   
  }

  //Copying from host to device
  cudaMemcpy(device_rowptr, rowptr.data(), (N + 1) * sizeof(int), cudaMemcpyHostToDevice);    //matrix
  cudaMemcpy(device_col, col.data(), col.size() * sizeof(int), cudaMemcpyHostToDevice);       //matrix
  cudaMemcpy(device_val, val.data(), val.size() * sizeof(double), cudaMemcpyHostToDevice);    //matrix
  cudaMemset(device_x, 0, N * sizeof(double));                                           //x vector
  cudaMemcpy(device_invM, invM.data(), N * sizeof(double), cudaMemcpyHostToDevice);      //Jacobi preconditoner
  cudaMemcpy(device_r, b.data(), N * sizeof(double), cudaMemcpyHostToDevice);      //Initiasl residual Ax-b=b as x=0
 
  //Launch precondiotion kernel giving z from r
  apply_jacobi_kernel<<<nBlocks, threadsPerBlock>>>(N, device_invM, device_r, device_z);
  //Initially copy direction from z
  cudaMemcpy(device_dir, device_z, N * sizeof(double), cudaMemcpyDeviceToDevice);
  
  //Do once in CPU norm of b
  double b_norm = sqrt(dot_product(b, b));

  //Dot Product of r and z
  dot_product_kernel<<<nBlocks, threadsPerBlock>>>(N, device_r, device_z, device_rz);
  double r_old=0.0;
  cudaMemcpy(&r_old, device_rz, sizeof(double), cudaMemcpyDeviceToHost);
  //std::cout<<"Initial rz: "<< r_old << std::endl;
  
  
  //Timers
  auto t0 = std::chrono::high_resolution_clock::now(); 
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float spmv_time=0.0f; 
  float dot_time=0.0f;
  float jacobi_time=0.0f;
  float update_xr_time =0.0f;
  float dir_update_time=0.0f;
    
  //Iterations
  for (int i = 0; i < iterations; i++){
  
    cudaMemset(device_dirAdir, 0, sizeof(double));
    cudaMemset(device_rz, 0, sizeof(double));
    cudaMemset(device_rnorm, 0, sizeof(double));
    
    cudaEventRecord(start);
    //A \times direction
    spmv_kernel<<<nBlocks, threadsPerBlock>>>(N, device_rowptr, device_col, device_val, device_dir, device_Adir);
    //Timing 
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    spmv_time += ms;     

    cudaEventRecord(start);
    //Dot product to update x and residual
    dot_product_kernel<<<nBlocks, threadsPerBlock>>>(N, device_Adir, device_dir, device_dirAdir);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    ms=0;
    cudaEventElapsedTime(&ms, start, stop);
    dot_time +=ms;

    double temp = 0;
    cudaMemcpy(&temp, device_dirAdir, sizeof(double), cudaMemcpyDeviceToHost);
    //Update x and residual
    double alpha=r_old/temp;
    
    cudaEventRecord(start);
    update_x_r_kernel<<<nBlocks, threadsPerBlock>>>(N, alpha, device_dir, device_Adir, device_x, device_r); 
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    ms=0;
    cudaEventElapsedTime(&ms, start, stop);
    update_xr_time +=ms;
    
    cudaEventRecord(start);
    //Check if new residual meets requirements
    dot_product_kernel<<<nBlocks, threadsPerBlock>>>(N, device_r, device_r, device_rnorm);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    ms=0;
    cudaEventElapsedTime(&ms, start, stop);
    dot_time +=ms;
    double r_norm = 0;
    cudaMemcpy(&r_norm, device_rnorm, sizeof(double), cudaMemcpyDeviceToHost);
    auto residual_norm = sqrt(r_norm)/b_norm;
    if (residual_norm  < tolerance) {
      std::cout << "Converged in " << i << " iterations with normalized residual: " << residual_norm << std::endl;
      break;
    }
 
    cudaEventRecord(start);
    //Apply Preconditioning to new residual
    apply_jacobi_kernel<<<nBlocks, threadsPerBlock>>>(N, device_invM, device_r, device_z);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    ms=0;
    cudaEventElapsedTime(&ms, start, stop);
    jacobi_time +=ms;
    

    cudaEventRecord(start);
    //Dot product to find r_(t+1)z_(t+1)
    //double r_new = gpu_dot(N, d_r, d_z, d_dot_temp);
    dot_product_kernel<<<nBlocks, threadsPerBlock>>>(N, device_r, device_z, device_rz);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    dot_time +=ms;
    double r_new =0;
    cudaMemcpy(&r_new, device_rz, sizeof(double), cudaMemcpyDeviceToHost);
    double beta = r_new / r_old;
    
    //Update direction and update rz
    cudaEventRecord(start);
    update_direction_kernel<<<nBlocks, threadsPerBlock>>>(N, beta, device_z, device_dir);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    dir_update_time +=ms;

    r_old = r_new;
    //std::cout<<"Iteration: "<< i <<"  "<<r_new<<std::endl;
  }
  
  cudaDeviceSynchronize();  // ensure all kernels finished
  std::cout << "Total SpMV kernel time: "<< spmv_time << " ms\n";
  std::cout << "Total Dot product time: "<< dot_time << " ms\n";
  std::cout << "Jacobi time: "<< jacobi_time << " ms\n";
  std::cout << "XR_update time: "<< update_xr_time << " ms\n";
  std::cout << "Dir_update time: "<< dir_update_time << " ms\n";

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  auto t1 = std::chrono::high_resolution_clock::now();
  double seconds = std::chrono::duration<double>(t1 - t0).count();
  std::cout << "Total wall time: " << 1000*seconds << "ms\n";

  cudaFree(device_rowptr);
  cudaFree(device_col);
  cudaFree(device_val);
  cudaFree(device_x);
  cudaFree(device_Adir);
  cudaFree(device_r);
  cudaFree(device_z);
  cudaFree(device_dir);

  // Free scalars
  cudaFree(device_rz);
  cudaFree(device_dirAdir);
  cudaFree(device_rnorm);

  return 0;
}
