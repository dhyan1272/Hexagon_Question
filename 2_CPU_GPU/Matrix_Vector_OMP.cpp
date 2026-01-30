#include <omp.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <cmath>
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

int main(int argc, char** argv) {  
  if(argc < 5){
    std::cerr<<"Usage ./Exec /path/to/matrix_file /path/to/vector_file row_size col_size"<<std::endl;
    exit(-1);
  }
  std::string A_file = argv[1];
  std::string x_file = argv[2];
  int n_row = std::stoi(argv[3]);
  int n_col = std::stoi(argv[4]);
 
  //Load data 
  auto A = load_csv(A_file, n_row, n_col);
  auto x = load_csv(x_file, n_col, 1);
  std::vector<double> y(n_row, 0.0);
  // We parallelize across rows. Each thread handles a subset of n_row
  auto start = omp_get_wtime();
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < n_row; ++i) {
    double row_sum = 0.0;
    // Use SIMD to exploit CPU vector units (AVX/AVX2)
    #pragma omp simd reduction(+:row_sum)
    for (int j = 0; j < n_col; ++j) {
      row_sum += A[i * n_col + j] * x[j];
    }
    y[i] = row_sum;
  }
  auto end = omp_get_wtime();
  auto diff=end-start;
  std::cout << "OpenMP MatrixVector Time: " << diff*1000 << " ms" << std::endl;

  auto norm_result=dot_product(y, y); 
  printf ("Norm of result %.15e \n", sqrt(norm_result));
 
  return 0;
}
