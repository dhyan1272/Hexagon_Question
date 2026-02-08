#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <fstream>
#include <string>
#include <cuda.h>

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

__global__ void ssor_forward_level_kernel(
    const int* __restrict__ flat_rows,
    int start, int end,
    const int* __restrict__ rowptr,
    const int* __restrict__ cols,
    const double* __restrict__ vals,
    double* __restrict__ y,
    const double* __restrict__ r,
    double omega){

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
  std::vector<double> val = load_csv<double>(dir + "val.csv");
  int N = rowptr.size() - 1;

  //Device Pointers
  int* device_rowptr, *device_col;
  double* device_val, *device_r, *device_y;
  
  //Memory Allocation
  cudaMalloc(&device_rowptr, (N + 1) * sizeof(int));
  cudaMalloc(&device_col, cols.size() * sizeof(int));
  cudaMalloc(&device_val, val.size() * sizeof(double));
  cudaMalloc(&device_r, N * sizeof(double));
  cudaMalloc(&device_y, N * sizeof(double));
  //Copying from host to device
  cudaMemcpy(device_rowptr, rowptr.data(), (N + 1) * sizeof(int), cudaMemcpyHostToDevice);    //matrix
  cudaMemcpy(device_col, cols.data(), cols.size() * sizeof(int), cudaMemcpyHostToDevice);      //matrix
  cudaMemcpy(device_val, val.data(), val.size() * sizeof(double), cudaMemcpyHostToDevice);    //matrix
  cudaMemset(device_r, 0, N * sizeof(double));                                                //x vector
  cudaMemset(device_y, 0, N * sizeof(double));                                                //x vector

  //Initial guess and RHS
  std::vector<double> x(N, 0.0);        // Initial guess (zeros)
  std::vector<double> b(N, 1.0);        // Right hand side (ones)
  std::vector<double> r=b;              // Initial residual b-Ax =b, as x=0

  //SSOR parameters and get z from r using SSOR
  double omega = 1.7; // Relaxation factor

  // Step 1: compute L-dependencies for each row
  std::vector<std::vector<int>> Ldeps(N);
  for (int i = 0; i < N; i++) {
    for (int idx = rowptr[i]; idx < rowptr[i+1]; idx++) {
      int col = cols[idx];
      if (col < i) Ldeps[i].push_back(col);
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
  std::vector<int> level_offsets(num_levels+1, 0);
  // Kind of like CSR i.e offset
  for (int l = 0; l < num_levels; l++)
    level_offsets[l+1] = level_offsets[l] + rows_in_level[l].size();
  int total = level_offsets[num_levels];
  //Actual data
  std::vector<int> flat_rows(total);
  for (int l = 0; l < num_levels; l++) {
    int start = level_offsets[l];
    for (int k = 0; k < rows_in_level[l].size(); k++) {
      flat_rows[start + k] = rows_in_level[l][k];
    }
  }
  
  //Copy to device
  int *device_flat_rows;
  int *device_level_offsets;

  cudaMalloc(&device_flat_rows, flat_rows.size()*sizeof(int));
  cudaMalloc(&device_level_offsets, level_offsets.size()*sizeof(int));
  cudaMemcpy(device_flat_rows, flat_rows.data(), flat_rows.size()*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(device_level_offsets, level_offsets.data(), level_offsets.size()*sizeof(int), cudaMemcpyHostToDevice);

  //Launch kernels sequentailly
  for (int lvl = 0; lvl < num_levels; lvl++) {
    std::cout<<"==== Launching kernel for all rows in level:"<<lvl<<"===="<<std::endl;
    int start = level_offsets[lvl];
    int end   = level_offsets[lvl+1];
    ssor_forward_level_kernel<<<1, 128>>>(device_flat_rows, start, end, device_rowptr, device_col, device_val, device_r, device_y, omega);
  }






  //Debug
  //for (int k=0; k<N; k++)
    //std::cout<<"Row "<<k<<" Label " << label[k]<< std::endl;
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
 
  return 0;
}
