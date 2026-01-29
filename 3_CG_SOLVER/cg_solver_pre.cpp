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

  double tolerance  = 1e-8;
  double iterations = 1000;
  
  std::string dir = "poisson_128x128/";
  std::vector<int> rowptr = load_csv<int>(dir + "rowptr.csv");
  std::vector<int> col = load_csv<int>(dir + "col.csv");
  std::vector<double> val = load_csv<double>(dir + "val.csv");

  int N = rowptr.size() - 1;
  std::vector<double> x(N, 0.0);        // Initial guess (zeros)
  std::vector<double> b(N, 1.0);        // Right hand side (ones)

  //Extract Diagonal for Jacobi Preconditioner
  std::vector<double> invM(N);
  for (int i = 0; i < N; ++i) {
    for (int j = rowptr[i]; j < rowptr[i + 1]; ++j) {
      if (col[j] == i) {
        invM[i] = 1.0 / val[j];
        break;
      }
    }
  }
  //Residuals
  std::vector<double> r = b;          // Residual r = b - Ax (since x=0, r=b)
  std::vector<double> residual_hist;
  residual_hist.reserve(iterations);
  residual_hist.push_back(std::sqrt(dot_product(r, r)));

  //Initial pre-conditioning step 
  std::vector<double> z(N);
  for (int j=0; j<N; j++) z[j] = invM[j] * r[j]; 

  std::vector<double> d = z;          // Search direction is pre-conditioned residual
  std::vector<double> Ad(N);          // Multiplication of matrix and vector
    
  double r_old = dot_product(r, z);
  double b_norm = sqrt(dot_product(b, b));
  std::cout<<"N and bnorm: "<<N<<" "<<b_norm<<std::endl;
 
  for (int i = 0; i < iterations; ++i) {
  
    //Matrix Vector Multiplication  
    spmv(rowptr, col, val, d, Ad);
    //Step size    
    double alpha = r_old / dot_product(d, Ad);
    //Take optimal step and update residual  
    for (int j = 0; j < N; ++j) {
      x[j] = x[j] + alpha * d[j];
      r[j] = r[j] - alpha * Ad[j];
    }
    
    double residue_norm = std::sqrt(dot_product(r, r));
    residual_hist.push_back(residue_norm);
    //If resuidual is within tolerance, come out of the iteration loop
    if (residue_norm/b_norm  < tolerance) {
      std::cout << "Converged in " << i << " iterations.\n";
      break;
    }
    //Apply pre-conditioner to the new residual
    for (int j=0; j<N; j++) 
      z[j] = invM[j] * r[j];
    double r_new = dot_product(r, z);
    //Find next A-orthogonal drection
    for (int j = 0; j < N; ++j) {
      d[j] = z[j] + (r_new / r_old) * d[j];
    }
    r_old = r_new;
  }

  //Writing to file
  std::ofstream outFile("residual_CG_64.csv");
  if (outFile.is_open()) {
    for (size_t i = 0; i < residual_hist.size(); ++i) {
      outFile << i << "," << residual_hist[i] << "\n";
    }
    outFile.close();
    std::cout << "Data saved to residual_CG.csv" << std::endl;
  }
  else {
    std::cerr << "Error: Could not open file for writing!" << std::endl;
  }
  return 0;
}
