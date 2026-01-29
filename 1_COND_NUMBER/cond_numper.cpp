#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Eigenvalues>

Eigen::MatrixXd load_csv(const std::string& path, int rows, int cols) {
  
  Eigen::MatrixXd mat(rows, cols);
  std::ifstream file(path);
  std::string line, val;
  for (int i = 0; i < rows; i++) {
    getline(file, line);
    std::stringstream ss(line);
    for (int j = 0; j < cols; j++) {
      getline(ss, val, ',');
      mat(i, j) = stod(val);
    }
  }
  return mat;
}

void get_gershgorin_bounds(const Eigen::MatrixXd& A, double& low, double& high) {
  low = std::numeric_limits<double>::max();
  high = std::numeric_limits<double>::lowest();
  for (int i = 0; i < A.rows(); ++i) {
    double center = A(i, i);
    double radius = A.row(i).lpNorm<1>() - abs(center);
    low  = std::min(low, center - radius);
    high = std::max(high, center + radius);
  }
}

int main(int argc, char** argv) {
  
  if(argc < 2){
    std::cerr<<"Usage ./EXEC /PATH/TO/MATRIX_FILE MATRIX SIZE(DEFAULT 64)"<<std::endl;
    exit(-1);
  }
  std::string A_file=argv[1];

  int n = (argc > 2) ? std::stoi(argv[2]) : 64;
  double tol = 1e-10;
  int max_iter = 1000;

  // 1. Load Data
  Eigen::MatrixXd A = load_csv(A_file, n, n);
           
  // 2. Power Iteration (lambda_max)
  //Store rayleigh quotients to store
  std::vector<double>rayleigh_max;
  Eigen::VectorXd v = Eigen::VectorXd::Random(n);
  v.normalize();
  double l_max = 0;
  for (int i = 0; i < max_iter; ++i) {
    double l_old = l_max;
    Eigen::VectorXd w = A * v;
    v = w.normalized();
    l_max = v.dot(w); // Rayleigh Quotient
    rayleigh_max.push_back(l_max);
    if (abs(l_max - l_old) < tol) break;
  }

  //3 .Inverse Power Iteration via CG (lambda_min)
  //We solve Ax = v at each step to find the max eigenvalue of A^-1
  //x_k+1=A^(-1)x_k--------------------Difficult
  //Instead solve for x_k+1 using Ax_k+1=x_k (x_k is known)
  //Store rayleigh quotients to store
  std::vector<double>rayleigh_min;
  Eigen::VectorXd v_min = Eigen::VectorXd::Random(n);
  v_min.normalize();
  double l_min = 0;
  //Conjugate Gradient solver 
  Eigen::ConjugateGradient<Eigen::MatrixXd, Eigen::Lower|Eigen::Upper> cg;
  cg.setTolerance(tol);
  cg.compute(A);
  for (int i = 0; i < max_iter; ++i) {
    double l_old = l_min;
    Eigen::VectorXd w = cg.solve(v_min);
    v_min = w.normalized();
    l_min = v_min.dot(A * v_min); // Rayleigh Quotient
    rayleigh_min.push_back(l_min);
    if (abs(l_min - l_old) < tol) break;
  }

  double true_min, true_max;
  if(n==64){
    //Validation 1 using EigenSolve
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> ces(A);
    true_min = ces.eigenvalues().minCoeff();
    true_max = ces.eigenvalues().maxCoeff();
  }
  //Validation 2
  double g_low, g_high;
  get_gershgorin_bounds(A, g_low, g_high);
  
  std::cout << "Using power & inverse power iteration: " << l_min    << " and " << l_max    << std::endl;
  if(n==64)
    std::cout << "Eigen Solve Lambda Min and Max: "      << true_min << " and " << true_max << std::endl;
  std::cout << "Gershogin bounds : "                     << g_low    << " and " << g_high   << std::endl;


  //Writing to file
  std::ofstream outFile("rayleigh_max.csv");
  if (outFile.is_open()) {
    for (size_t i = 0; i < rayleigh_max.size(); ++i) {
      outFile << i << "," << rayleigh_max[i] << "\n";
    }
    outFile.close();
    std::cout << "Data saved to rayleigh_max.csv" << std::endl;
  } 
  else {
    std::cerr << "Error: Could not open file for writing!" << std::endl;
  }

  std::ofstream outFile2("rayleigh_min.csv");
  if (outFile2.is_open()) {
    for (size_t i = 0; i < rayleigh_min.size(); ++i) {
      outFile2 << i << "," << rayleigh_min[i] << "\n";
    }
    outFile2.close();
    std::cout << "Data saved to rayleigh_min.csv" << std::endl;
  } 
  else {
    std::cerr << "Error: Could not open file for writing!" << std::endl;
  }

  return 0;
}
