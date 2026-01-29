#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <fstream>
#include <string>

void spmv(const std::vector<int>& rowptr, const std::vector<int>& col, 
          const std::vector<double>& val, const std::vector<double>& x, 
          std::vector<double>& y) {
  int N = x.size();
  for (int i = 0; i < N; ++i) {
    double sum = 0.0;
    for (int k = rowptr[i]; k < rowptr[i+1]; ++k) {
      sum += val[k] * x[col[k]];
    }
    y[i] = sum;
  }
}

double dot_product(const std::vector<double>& a, const std::vector<double>& b) {
  double res = 0.0;
  for (size_t i = 0; i < a.size(); ++i) res += a[i] * b[i];
  return res;
}

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

int main() {

  std::string dir = "poisoon_64x64/";
  std::vector<int> rowptr = load_csv<int>(dir + "rowptr.csv");
  std::vector<int> col = load_csv<int>(dir + "col.csv");
  std::vector<double> val = load_csv<double>(dir + "val.csv");

  int N = rowptr.size() - 1;
  std::vector<double> x(N, 0.0);        // Initial guess (zeros)
  std::vector<double> b(N, 1.0);        // Right hand side (ones)

  std::vector<double> r = b;            // Residual r = b - Ax (since x=0, r=b)
  std::vector<double> d = r;            // Search direction is opposite to gradient
  std::vector<double> Ad(N);            // Multiplication of matrix and vector
    
  double r_old = dot_product(r, d);
  
  double tolerance  = 1e-10;
  double iterations = 1000;
  
  for (int i = 0; i < iterations; ++i) {
  
    //Matrix Vector Multiplication  
    spmv(rowptr, col, val, d, Ad);
    //Step size    
    double alpha = dot_product(r, d) / dot_product(d, Ad);
    //Take optimal step and update residual  
    for (int j = 0; j < N; ++j) {
      x[j] = x[j] + alpha * d[j];
      r[j] = r[j] - alpha * Ad[j];
    }
    //If resuidual is within tolerance, come out of the iteration loop
    double r_new = dot_product(r, r);
    if (std::sqrt(r_new) < tolerance) {
      std::cout << "Converged in " << i << " iterations.\n";
      break;
    }
    //Else find next A-orthogonal drection
    for (int j = 0; j < N; ++j) {
      d[j] = r[j] + (r_new / r_old) * d[j];
    }
    r_old = r_new;
  }
  return 0;
}
