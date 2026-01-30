#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <cuda.h>

//Load data
std::vector<double> load_csv(const std::string& path, int rows, int cols) {
  std::vector <double> data(rows*cols);
  std::ifstream file(path);
  std::string line, val;
  for (int i = 0; i < rows; i++) {
    getline(file, line);
    std::stringstream ss(line);
    for (int j = 0; j < cols; j++) {
      getline(ss, val, ',');
      data[i*cols+j] = stod(val);
    }
  }
  return data;
}

//Dot product for norm calculation
double dot_product(const std::vector<double>& a, const std::vector<double>& b) {
  double res = 0.0;
  for (size_t i = 0; i < a.size(); ++i) res += a[i] * b[i];
  return res;
}

__global__ void matrixXvec_kernel(const double* A, const double* x, double* y, int n_row, int n_col){
  //Each thread handles one row of the matrix
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= n_row) return;
  double sum = 0.0;
  for (int j = 0; j < n_col; j++)
    sum += A[i * n_col + j] * x[j];
  y[i] = sum;
}

__global__ void matrixXvec_kernel_coalesced(const double* A, const double* x, double* y, int nrows, int ncols) {
  // Each warp will handle one row
  int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  int lane_id = threadIdx.x % 32; // ID within the warp (0-31)
    
  if (warp_id >= nrows) return;

  double sum = 0.0;
    
  // All threads in the warp loop over the columns of row 'warp_id'
  // with a stride of 32
  for (int j = lane_id; j < ncols; j += 32) {
    sum += A[warp_id * ncols + j] * x[j];
  }

  // Parallel Reduction: Sum the values held by all 32 threads in the warp
  for (int offset = 16; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  }
  // Thread 0 of the warp writes the final result
  if (lane_id == 0) {
    y[warp_id] = sum;
  }
}


int main(int argc, char** argv) {
  
  if(argc < 7){
    std::cerr<<"Usage ./Exec /path/to/matrix_file /path/to/vector_file row_size col_size threadsPerBlock Option 1/2"<<std::endl;
    exit(-1);
  }
  std::string A_file = argv[1];
  std::string x_file = argv[2];
  int n_row = std::stoi(argv[3]);
  int n_col = std::stoi(argv[4]);
  int threadsPerBlock =  std::stoi(argv[5]);
  int option = std::stoi(argv[6]); 
 
  //Load data 
  auto A = load_csv(A_file, n_row, n_col);
  auto x = load_csv(x_file, n_col, 1);

  //Grid Calculation
  int gridSize = -1;
  if (option == 1)
    gridSize = (n_row + threadsPerBlock - 1) / threadsPerBlock;
  else if(option == 2)
    gridSize = (n_row*32 + threadsPerBlock - 1) / threadsPerBlock; 

  //Timing with CUDA events instead of CPU timers
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  double *A_device, *x_device, *y_device; 
  cudaMalloc (& A_device, n_row*n_col*sizeof(double));
  cudaMalloc (& x_device, n_col*sizeof(double));
  cudaMalloc (& y_device, n_row*sizeof(double));

  cudaMemcpy(A_device, A.data(), n_row*n_col*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(x_device, x.data(), n_col*sizeof(double), cudaMemcpyHostToDevice);
  
  cudaEventRecord(start);
  if (option == 1)
    matrixXvec_kernel<<<gridSize, threadsPerBlock>>>(A_device, x_device, y_device, n_row, n_col);
  else if (option ==2)
    matrixXvec_kernel_coalesced<<<gridSize, threadsPerBlock>>>(A_device, x_device, y_device, n_row, n_col);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("GPU-kernel time in milliseconds %f \n", milliseconds);

  std::vector<double> y_host(n_row);
  cudaMemcpy(y_host.data(), y_device, n_row*sizeof(double), cudaMemcpyDeviceToHost);

  //This is for verification with a CPU code 
  double y_norm = dot_product(y_host, y_host);
  //printf ("Norm of result %.15e \n", sqrt(y_norm));

  //Free CUDA memory
  cudaFree(A_device);
  cudaFree(x_device);
  cudaFree(y_device);
  
  return 0;
}
