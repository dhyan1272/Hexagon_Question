#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <fstream>
#include <string>
#include <cuda.h>

#define threadsPerBlock 256

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
  if(i>=N) return;

  __shared__ double shared_data[threadsPerBlock];
  int tid = threadIdx.x;

  double val = a[i] * b[i];
  shared_data[tid] = val;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) shared_data[tid] += shared_data[tid + s];
    __syncthreads();
  }
  if (tid == 0) atomicAdd(result, shared_data[0]);
}

int main(int argc, char* argv[]) {

  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <directory_path>" << std::endl;
    exit(-1);
  }
  std::string dir = argv[1];
  //Convergence parameters
  double tolerance  = 1e-8;
  double iterations = 10;
 
  //ReadInputs & Calculate N
  std::vector<int> rowptr = load_csv<int>(dir + "rowptr.csv");
  std::vector<int> col = load_csv<int>(dir + "col.csv");
  std::vector<double> val = load_csv<double>(dir + "val.csv");
  int N = rowptr.size() - 1;
  
  //GPU kernel launch parameters
  int nBlocks = (N + threadsPerBlock - 1)/threadsPerBlock;
 
  //Device Pointers
  int* device_rowptr, *device_col;
  double* device_val, *device_x, *device_Adir, *device_invM, *device_r, *device_z, *device_dir, *device_rz, *device_dirAdir;
  
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
  //Initially copy dire from z
  cudaMemcpy(device_dir, device_z, N * sizeof(double), cudaMemcpyDeviceToDevice);
  
  //Dot Product of r and z
  dot_product_kernel<<<nBlocks, threadsPerBlock>>>(N, device_r, device_z, device_rz);
  double r_old=0.0;
  cudaMemcpy(&r_old, device_rz, sizeof(double), cudaMemcpyDeviceToHost);
  std::cout<<"Initial rz: "<< r_old << std::endl;
  
  //Iterations
  for (int i = 0; i < iterations; i++){
    //A \times direction
    spmv_kernel<<<nBlocks, threadsPerBlock>>>(N, device_rowptr, device_col, device_val, device_dir, device_Adir);
     
    //Dot product to update x and residual
    dot_product_kernel<<<nBlocks, threadsPerBlock>>>(N, device_Adir, device_dir, device_dirAdir);
    double temp = 0;
    cudaMemcpy(&temp, device_dirAdir, sizeof(double), cudaMemcpyDeviceToHost);

    //Update x and residual
    double alpha=r_old/temp;
    update_x_r_kernel<<<nBlocks, threadsPerBlock>>>(N, alpha, device_dir, device_Adir, device_x, device_r); 
    std::cout<<"Iteration: "<< i <<"  "<<temp<<"  "<<alpha<<std::endl;
    
    //Apply Preconditioning to new residual
    apply_jacobi_kernel<<<nBlocks, threadsPerBlock>>>(N, device_invM, device_r, device_z);
    
    //Dot product to find r_(t+1)z_(t+1)
    //double r_new = gpu_dot(N, d_r, d_z, d_dot_temp);
    dot_product_kernel<<<nBlocks, threadsPerBlock>>>(N, device_r, device_z, device_rz);
    double r_new =0;
    cudaMemcpy(&r_new, device_rz, sizeof(double), cudaMemcpyDeviceToHost);
    double beta = r_new / r_old;
    
    //Update direction and update rz
    update_direction_kernel<<<nBlocks, threadsPerBlock>>>(N, beta, device_z, device_dir);
    r_old = r_new;
    std::cout<<"Iteration: "<< i <<"  "<<r_new<<std::endl;
  }
  
  //Debugging
  if(true)
    return;
  std::vector<double> y_host(N);
  cudaMemcpy(y_host.data(), device_Adir, N * sizeof(double), cudaMemcpyDeviceToHost);
  for (int k=0; k<N; k++) 
    if(y_host[k] !=0) std::cout<<"y=Ax: "<< y_host[k] << std::endl;
  return 0;
}
