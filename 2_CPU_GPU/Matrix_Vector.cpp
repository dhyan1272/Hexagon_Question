#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <cmath>

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

//Dot product for norm calcualtion
double dot_product(const std::vector<double>& a, const std::vector<double>& b) {
  double res = 0.0;
  for (size_t i = 0; i < a.size(); ++i) res += a[i] * b[i];
  return res;
}

std::vector<double> MatrixVector(const std::vector<double>& A, const std::vector<double>& x) {
  int rows = x.size();
  int cols = A.size()/rows;
  std::vector<double> result(rows);

  for (int i = 0; i < rows; ++i){
    double sum = 0.0;
    for (int j = 0; j < cols; ++j) {
      sum += A[i * cols + j] * x[j];
    }
    result[i] = sum;
  }
  //Debug
  printf("Number of rows %d columns %d \n", rows, cols);
  return result;
}


int main(int argc, char** argv) {
  
  if(argc < 4){
    std::cerr<<"Usage ./EXEC /PATH/TO/MATRIX_FILE  /PATH/TO/VECTOR_FILE MATRIX_SIZE"<<std::endl;
    exit(-1);
  }
  std::string A_file = argv[1];
  std::string x_file = argv[2];
  int n              = std::stoi(argv[3]);
  
  auto A = load_csv(A_file, n, n);
  auto x = load_csv(x_file, n, 1);
  auto result = MatrixVector(A, x);
  
  auto norm_result=dot_product(result, result);

  printf ("Norm of result %.15e \n", sqrt(norm_result));
  return 0;
}
