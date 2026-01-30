near Algebra and HPC Kernels

This repository contains implementations of numerical linear algebra algorithms
and high-performance computing (HPC) kernels:

1. Condition Number Estimation for Symmetric Positive Definite (SPD) matrices  
2. Dense Matrix–Vector Multiplication  
   - CPU (serial)  
   - CPU (OpenMP)  
   - GPU (CUDA)  
3. Preconditioned Conjugate Gradient (PCG) solver for the 2-D Poisson equation
   using CSR (Compressed Sparse Row) storage

---

## 1. Condition Number Estimation (SPD Matrices)

This implementation uses the Eigen library.

**Dependency:**  
Eigen (header-only). Clone using  
`git clone https://gitlab.com/libeigen/eigen.git`

**Compile:**  
`g++ -O3 -I path/to/eigen cond_number.cpp -o cond_number`

**Run:**  
`./cond_number /path/to/matrix_file matrix_size`

---

## 2. Dense Matrix–Vector Multiplication

### CPU (Serial / OpenMP)

**Compile (serial):**  
`g++ -O3 Matrix_Vector.cpp -o matvec_cpu`

**Run:**  
`./matvec_cpu /path/to/A_file /path/to/x_file matrix_size`

**Compile (OpenMP):**  
`g++ -O3 -fopenmp Matrix_Vector_OMP.cpp -o Exec`

**Run:** 
`export OMP_NUM_THREADS=#THREADS` 
`./Exec /path/to/A_file /path/to/x_file rwo_size col_size`

---

### GPU (CUDA)

**Compile:**  
`nvcc -O3 Matrix_Vector.cu -o matvec_gpu`

**Run:**  
`./matvec_gpu /path/to/A_file /path/to/x_file row_size col_size threadsPerBlock option`

where  
- `row_size`, `col_size` are the matrix dimensions  
- `threadsPerBlock` specifies the CUDA block size  
- `option` selects the kernel implementation (e.g., 1 or 2)

---

## 3. Preconditioned Conjugate Gradient (PCG)

This implementation solves the 2-D Poisson equation using:
- CSR sparse matrix storage  
- Jacobi preconditioning  
- Residual-based convergence criteria

---
**Compile:**  g++ -O3 cg_solver_pre.cpp -o cg_solver

**Run:**  
`./cg_solver /path/to/poisson_dir`

