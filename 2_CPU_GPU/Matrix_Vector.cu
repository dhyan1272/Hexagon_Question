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


int main(int argc, char** argv) {
  
  if(argc < 6){
    std::cerr<<"Usage ./Exec /path/to/matrix_file /path/to/vector_file row_size col_size blockSize"<<std::endl;
    exit(-1);
  }
  std::string A_file = argv[1];
  std::string x_file = argv[2];
  int n_row = std::stoi(argv[3]);
  int n_col = std::stoi(argv[4]);
 
  //Grid Calculation
  int blockSize =  std::stoi(argv[5]);
  int gridSize = (n_row + blockSize - 1) / blockSize;

  //Load data 
  auto A = load_csv(A_file, n_row, n_col);
  auto x = load_csv(x_file, n_col, 1);
  
  double *A_device, *x_device, *y_device; 
  cudaMalloc (& A_device, n_row*n_col*sizeof(double));
  cudaMalloc (& x_device, n_col*sizeof(double));
  cudaMalloc (& y_device, n_row*sizeof(double));

  cudaMemcpy(A_device, A.data(), n_row*n_col*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(x_device, x.data(), n_col*sizeof(double), cudaMemcpyHostToDevice);
  
  
  matrixXvec_kernel<<<gridSize, blockSize>>>(A_device, x_device, y_device, n_row, n_col);

  std::vector<double> y_host(n_row);
  cudaMemcpy(y_host.data(), y_device, n_row*sizeof(double), cudaMemcpyDeviceToHost);

  
  double y_norm = dot_product(y_host, y_host);
  printf ("Norm of result %.15e \n", sqrt(y_norm));

  //Free CUDA memory
  cudaFree(A_device);
  cudaFree(x_device);
  cudaFree(y_device);
  
  return 0;
}
