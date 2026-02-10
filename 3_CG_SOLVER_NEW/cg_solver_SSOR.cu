#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <fstream>
#include <string>
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

double dot_product(const std::vector<double>& a, const std::vector<double>& b) {
  double res = 0.0;
  for (size_t i = 0; i < a.size(); ++i) res += a[i] * b[i];
  return res;
}

void computeLDependence(const std::vector<int>& rowptr, const std::vector<int>cols, const int N, std::vector<int>& level_offsets, 
     std::vector<int>& flat_rows, bool lower,  bool debug){

  // Step 1: compute L-dependencies for each row
  std::vector<std::vector<int>> Ldeps(N);
  for (int i = 0; i < N; i++) {
    for (int idx = rowptr[i]; idx < rowptr[i+1]; idx++) {
      int col = cols[idx];
      if (lower && col < i)
        Ldeps[i].push_back(col);    // L-dependencies
      else if (!lower && col > i)
        Ldeps[i].push_back(col);    // U-dependencies
    }
  }
  // Step 2: assign levels
  // Assign label for each row so that each row label can be fed to a different thread
  int label[N];          // output: level of each row
  bool assigned[N] = {false};
  int remaining = N;
  int current_level = 0;

  while (remaining > 0) {
    std::vector<int> ready;
    // scan using old assigned[]
    //e.g. for first of scan all rows (0 to n-1), everything will remain assigned=false
    //so only the first row whih has no dependency will be set current level
    for (int i = 0; i < N; i++) {
      if (!assigned[i]) {
        //Are all the lower-triangular dependencies of row i already assigned a level
        //If not wait for next pass
        bool deps_done = true;
        for (int d : Ldeps[i]) {
          if (!assigned[d]) {
            deps_done = false;
            break;
          }
        }
        if (deps_done)
          ready.push_back(i);
      }
    }
    // assign AFTER scan
    for (int i : ready) {
      assigned[i] = true;
      label[i] = current_level;
      remaining--;
    }
    current_level++;
  }

  // Each level has rows associated with it
  // Push rows for each level
  std::vector<std::vector<int>> rows_in_level(current_level);
  for (int i = 0; i < N; i++) {
    int lvl = label[i];
    rows_in_level[lvl].push_back(i);
  }
  //Cannot pass to GPU as vector of vector, so flattening it like a CSR array
  int num_levels = rows_in_level.size();
  level_offsets.resize(rows_in_level.size() + 1, 0);
  // Kind of like CSR i.e offset
  for (int l = 0; l < num_levels; l++)
    level_offsets[l+1] = level_offsets[l] + rows_in_level[l].size();
  int total = level_offsets[num_levels];
  //Actual data
  flat_rows.resize(total, 0);

  for (int l = 0; l < num_levels; l++) {
    int start = level_offsets[l];
    for (int k = 0; k < rows_in_level[l].size(); k++) {
      flat_rows[start + k] = rows_in_level[l][k];
    }
  }

  if(!debug) return;
  for (int lvl = 0; lvl < rows_in_level.size(); lvl++) {
    std::cout << "Level " << lvl << ": ";
    for (int r : rows_in_level[lvl]) {
        std::cout << r << " ";
    }
    std::cout << "\n";
  }
  std::cout<<"CSR indexing"<<std::endl;
  for (int k=0; k <level_offsets.size(); k++)
    std::cout<< level_offsets[k]<<std::endl;
  std::cout<<"Row data to operate on"<<std::endl; 
  for (int k=0;k <flat_rows.size(); k++)
    std::cout<< flat_rows[k]<<std::endl;
}

__global__ void ssor_forward_level_kernel(
    const int* __restrict__ flat_rows,
    int start, int end,
    const int* __restrict__ rowptr,
    const int* __restrict__ cols,
    const double* __restrict__ vals,
    const double* __restrict__ r,
    double* __restrict__ y,
    const double omega){

  int tid = blockIdx.x * blockDim.x + threadIdx.x;  //Current thread ID Thread Id
  int idx = start + tid;                            //Index in flatrow this thread should work on 
  if (idx >= end) return;
  int i = flat_rows[idx];                           //Actual row in flarRow this thread shouuld work on

  double sum_Ly=0.0;
  double d_ii=0.0;
  for (int k = rowptr[i]; k < rowptr[i+1]; k++) {
    if(cols[k]<i)      
      sum_Ly += vals[k] * y[cols[k]];
    else if(cols[k] == i)
      d_ii=vals[k];      
  }
  y[i] = (r[i] - omega*sum_Ly) / d_ii;
}

__global__ void ssor_backward_level_kernel(
    const int* __restrict__ flat_rows,
    int start, int end,
    const int* __restrict__ rowptr,
    const int* __restrict__ cols,
    const double* __restrict__ vals,
    const double* __restrict__ y,
    double* __restrict__ z,
    const double omega){

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = start + tid;
  if (idx >= end) return;
  int i = flat_rows[idx];

  double factor = omega*(2-omega);
  double d_ii = 0.0;
  double sum_Uz = 0.0;
  for (int k = rowptr[i]; k < rowptr[i+1]; k++){
    if (cols[k]>i)
      sum_Uz += vals[k]*z[cols[k]]; 
    else if (cols[k] == i) 
      d_ii = vals[k];
  }
  z[i] = (factor * d_ii * y[i]- omega*sum_Uz)/d_ii;
}

__global__ void spmv_kernel(int N, const int* rowptr, const int* col, const double* val, const double* x, double* y) {
  
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  double sum = 0.0;
  for (int k = rowptr[i]; k < rowptr[i+1]; ++k) {
    sum += val[k] * x[col[k]];
  }
  y[i] = sum;
}

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

void ssor_gpu( const int N,
  const int* device_rowptr,
  const int* device_cols,
  const double* device_vals,
  const double* device_r,
  double* device_z, //Output
  const int* device_flat_rows_low,
  const std::vector<int>& level_offsets_low,
  const int* device_flat_rows_u,
  const std::vector<int>& level_offsets_u,
  double omega){

  double *device_y;
  
  //Memory Allocation
  cudaMalloc(&device_y, N * sizeof(double));
  cudaMemset(device_y, 0, N * sizeof(double));              

  // Forward sweep
  int num_levels_low = level_offsets_low.size() - 1;
  for (int lvl = 0; lvl < num_levels_low; lvl++) {
    int start = level_offsets_low[lvl];
    int end   = level_offsets_low[lvl + 1];
    int nrows = end - start;
    int nBlocks = (nrows + threadsPerBlock - 1) / threadsPerBlock;
    ssor_forward_level_kernel<<<nBlocks, threadsPerBlock>>>(device_flat_rows_low, start, end, device_rowptr, device_cols,
                                                            device_vals, device_r, device_y, omega);
    cudaDeviceSynchronize(); // ensure this level finished
  }
  // Backward sweep
  int num_levels_u = level_offsets_u.size() - 1;
  for (int lvl = 0; lvl < num_levels_u; lvl++) {
    int start = level_offsets_u[lvl];
    int end   = level_offsets_u[lvl + 1];
    int nrows = end - start;
    int nBlocks = (nrows + threadsPerBlock - 1) / threadsPerBlock;
    ssor_backward_level_kernel<<<nBlocks, threadsPerBlock>>>(device_flat_rows_u, start, end, device_rowptr, device_cols, 
                                                             device_vals, device_y, device_z, omega);
    cudaDeviceSynchronize(); // ensure this level finished
  }
  cudaFree(device_y);
}

int main(int argc, char* argv[]) {
  
  //Program basic inputs
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <directory_path>" << std::endl;
    exit(-1);
  }
  std::string dir = argv[1];
  //Max Loops and tolerance 
  double tolerance  = 1e-8;
  int iterations = 1000;
  
  //Read Inputs and calulate N
  std::vector<int> rowptr = load_csv<int>(dir + "rowptr.csv");
  std::vector<int> cols = load_csv<int>(dir + "col.csv");
  std::vector<double> vals = load_csv<double>(dir + "val.csv");
  const int N = rowptr.size() - 1;
  int nBlocks = (N + threadsPerBlock - 1)/threadsPerBlock;
 
  //Device Pointers
  int* device_rowptr, *device_cols;
  double* device_vals, *device_x, *device_r, *device_Adir, *device_z, *device_dir;
  double* device_rz, *device_dirAdir, *device_rnorm;
  
  //Memory Allocation
  cudaMalloc(&device_rowptr, (N + 1) * sizeof(int));
  cudaMalloc(&device_cols, cols.size() * sizeof(int));
  cudaMalloc(&device_vals, vals.size() * sizeof(double));
  cudaMalloc(&device_x, N*sizeof(double));
  cudaMalloc(&device_Adir, N*sizeof(double));  
  cudaMalloc(&device_r, N * sizeof(double));
  cudaMalloc(&device_z, N * sizeof(double));
  cudaMalloc(&device_dir, N*sizeof(double));   //Search direction
  cudaMalloc(&device_rz, sizeof(double));      //Scalar
  cudaMalloc(&device_dirAdir, sizeof(double)); //Scalar
  cudaMalloc(&device_rnorm, sizeof(double));   //Scalar 

  //Copying from host to device
  cudaMemcpy(device_rowptr, rowptr.data(), (N + 1) * sizeof(int), cudaMemcpyHostToDevice);    //matrix
  cudaMemcpy(device_cols, cols.data(), cols.size() * sizeof(int), cudaMemcpyHostToDevice);      //matrix
  cudaMemcpy(device_vals, vals.data(), vals.size() * sizeof(double), cudaMemcpyHostToDevice);    //matrix
  cudaMemset(device_x, 0, N * sizeof(double));                                             //Initial x vector
  //Thus residual b-Ax = b
  //Set b and then the residual
  std::vector<double> b(N, 1.0);
  double b_norm = sqrt(dot_product(b, b));   //Done once
  cudaMemcpy(device_r, b.data(), N * sizeof(double), cudaMemcpyHostToDevice); 
  
  //Now need to apply preconditioning to residual and calculate z
  //Set initial direction as z instead of residual
  //If no precondioning, set initial direction as resiual as no concept of z
  cudaMemcpy(device_dir, device_z, N * sizeof(double), cudaMemcpyDeviceToDevice);
     
  //SSOR parameters and get z from r using SSOR
  double omega = 1.7; // Relaxation factor

  std::vector<int> level_offsets_low;
  std::vector<int> flat_rows_low;
  bool lower= true;
  computeLDependence(rowptr, cols, N, level_offsets_low, flat_rows_low, lower, false); 
  std::vector<int> level_offsets_u;
  std::vector<int> flat_rows_u;
  lower= false;
  computeLDependence(rowptr, cols, N, level_offsets_u, flat_rows_u, lower, false); 
  
  //Copy to device
  int *device_flat_rows_low, *device_flat_rows_u;
  cudaMalloc(&device_flat_rows_low, flat_rows_low.size()*sizeof(int));
  cudaMemcpy(device_flat_rows_low, flat_rows_low.data(), flat_rows_low.size()*sizeof(int), cudaMemcpyHostToDevice);
  cudaMalloc(&device_flat_rows_u, flat_rows_u.size()*sizeof(int));
  cudaMemcpy(device_flat_rows_u, flat_rows_u.data(), flat_rows_u.size()*sizeof(int), cudaMemcpyHostToDevice);

  ssor_gpu(N, device_rowptr, device_cols, device_vals, device_r, device_z, device_flat_rows_low, level_offsets_low, 
           device_flat_rows_u, level_offsets_u, omega);
  
  //Set init direction as z
  cudaMemcpy(device_dir, device_z, N * sizeof(double), cudaMemcpyDeviceToDevice);
  
  dot_product_kernel<<<nBlocks, threadsPerBlock>>>(N, device_r, device_z, device_rz);
  double r_old=0.0;
  cudaMemcpy(&r_old, device_rz, sizeof(double), cudaMemcpyDeviceToHost);

  for (int i = 0; i < iterations; i++){
    cudaMemset(device_dirAdir, 0, sizeof(double));
    cudaMemset(device_rz, 0, sizeof(double));
    cudaMemset(device_rnorm, 0, sizeof(double));
    
    //A \times direction
    spmv_kernel<<<nBlocks, threadsPerBlock>>>(N, device_rowptr, device_cols, device_vals, device_dir, device_Adir);
    
    //Dot product to update x and residual
    dot_product_kernel<<<nBlocks, threadsPerBlock>>>(N, device_Adir, device_dir, device_dirAdir);
    double temp = 0;
    cudaMemcpy(&temp, device_dirAdir, sizeof(double), cudaMemcpyDeviceToHost);
    //Update x and residual
    double alpha=r_old/temp; 
    update_x_r_kernel<<<nBlocks, threadsPerBlock>>>(N, alpha, device_dir, device_Adir, device_x, device_r); 
    //std::cout<<"Iteration temp alpha:   "<< i<<" "<<temp <<" " <<alpha<<std::endl;
   
    //Check if new residual meets requirements
    dot_product_kernel<<<nBlocks, threadsPerBlock>>>(N, device_r, device_r, device_rnorm);
    double r_norm = 0;
    cudaMemcpy(&r_norm, device_rnorm, sizeof(double), cudaMemcpyDeviceToHost);
    auto residual_norm = sqrt(r_norm)/b_norm;
    if (residual_norm  < tolerance) {
      std::cout << "Converged in " << i << " iterations with normalized residual: " << residual_norm << std::endl;
      break;
    }
 
    //Apply Preconditioning to new residual
    ssor_gpu(N, device_rowptr, device_cols, device_vals, device_r, device_z, device_flat_rows_low, level_offsets_low, 
           device_flat_rows_u, level_offsets_u, omega);
  
    //Dot product to find r_(t+1)z_(t+1)
    dot_product_kernel<<<nBlocks, threadsPerBlock>>>(N, device_r, device_z, device_rz);
    double r_new =0;
    cudaMemcpy(&r_new, device_rz, sizeof(double), cudaMemcpyDeviceToHost);
    double beta = r_new / r_old;
    
    //Update direction and update rz
    update_direction_kernel<<<nBlocks, threadsPerBlock>>>(N, beta, device_z, device_dir);

    r_old = r_new;
    //std::cout<<"Iteration: "<< i <<"  "<<r_new<<std::endl;
  }
  cudaFree(device_rowptr);
  cudaFree(device_cols);
  cudaFree(device_vals);
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
