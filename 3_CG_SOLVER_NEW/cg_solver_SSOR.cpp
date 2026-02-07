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



void apply_ssor(int n, const std::vector<int>& rowptr, const std::vector<int>& cols, 
                const std::vector<double>& vals, const std::vector<double>& r, 
                std::vector<double>& z, double omega) {
    
  std::vector<double> y(n, 0.0);
  std::vector<double> diag_vals(n);

  // 1. Forward Sweep: Solve (D + omega*L) Y = r
  for (int i = 0; i < n; ++i) {
    double sum_Ly = 0.0;
    double d_ii = 0.0;
    for (int k = rowptr[i]; k < rowptr[i + 1]; ++k) {
      if (cols[k] < i) {
        sum_Ly += vals[k] * y[cols[k]];
      } 
      else if (cols[k] == i) {
        d_ii = vals[k];
      }
    }
    diag_vals[i] = d_ii;
    y[i] = (r[i] - omega * sum_Ly) / d_ii;
  }
  // 2. & 3. Scaling and Backward Sweep
  // Solves (D + omega*U) Z = omega*(2-omega)* D \times Y
  double factor = omega * (2.0 - omega);
  for (int i = n - 1; i >= 0; --i) {
    double sum_Uz = 0.0;
    for (int k = rowptr[i]; k < rowptr[i + 1]; ++k) {
      if (cols[k] > i) {
        sum_Uz += vals[k] * z[cols[k]];
      }
    }                                                                                                                                 
    z[i] = (factor * diag_vals[i] * y[i] - omega * sum_Uz) / diag_vals[i];
  }
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
  std::vector<int> col = load_csv<int>(dir + "col.csv");
  std::vector<double> val = load_csv<double>(dir + "val.csv");
  int N = rowptr.size() - 1;

  //Initial guess and RHS
  std::vector<double> x(N, 0.0);        // Initial guess (zeros)
  std::vector<double> b(N, 1.0);        // Right hand side (ones)
  std::vector<double> r=b;              // Initial residual b-Ax =b, as x=0
  double b_norm = sqrt(dot_product(b, b));

  //SSOR parameters and get z from r using SSOR
  double omega = 1.7; // Relaxation factor
  std::vector<double> z(N, 0.0);
  apply_ssor(N, rowptr, col, val, r, z, omega);

  std::vector<double> d = z;          // Search direction is pre-conditioned residual
  std::vector<double> Ad(N);          // Multiplication of matrix and vector
    
  //Convergence history (optional)
  std::vector<double> residual_hist;
  residual_hist.reserve(iterations);
  residual_hist.push_back(std::sqrt(dot_product(r, r)));
    
  double r_old = dot_product(r, z);
 
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
    
    //Convergence criteria check
    double residue_norm = std::sqrt(dot_product(r, r));
    residual_hist.push_back(residue_norm);
    //If resuidual is within tolerance, come out of the iteration loop
    if (residue_norm/b_norm  < tolerance) {
      std::cout << "Converged in " << i << " iterations with normalized residual: " << residue_norm/b_norm << std::endl;
      break;
    }
    
    //Apply pre-conditioner to the new residual
    apply_ssor(N, rowptr, col, val, r, z, omega);

    double r_new = dot_product(r, z);
    //Find next A-orthogonal drection
    for (int j = 0; j < N; ++j) {
      d[j] = z[j] + (r_new / r_old) * d[j];
    }
    r_old = r_new;
  }

  //Writing to file
  std::ofstream outFile("residual_CG.csv");
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
