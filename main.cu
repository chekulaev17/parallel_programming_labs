#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <string>
#include <cstdlib>
#include <cuda_runtime.h>

using namespace std;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            cerr << "CUDA error: " << cudaGetErrorString(err) << endl; \
            exit(1); \
        } \
    } while(0)

vector<vector<double>> readMatrix(const string& filename, int& n) {
    ifstream fin(filename);
    if (!fin.is_open()) {
        cerr << "Ошибка: не удалось открыть файл '" << filename << "'" << endl;
        exit(1);
    }
    if (!(fin >> n) || n <= 0) {
        cerr << "Ошибка: неверный размер матрицы" << endl;
        exit(1);
    }
    vector<vector<double>> mat(n, vector<double>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (!(fin >> mat[i][j])) {
                cerr << "Ошибка: недостаточно данных" << endl;
                exit(1);
            }
    fin.close();
    return mat;
}

void writeMatrix(const string& filename, const vector<vector<double>>& mat, double elapsed_ms = -1.0) {
    ofstream fout(filename);
    if (!fout.is_open()) {
        cerr << "Ошибка: не удалось создать файл" << endl;
        exit(1);
    }
    int n = mat.size();
    fout << n << "\n";
    fout << fixed << setprecision(10);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            fout << mat[i][j] << (j == n - 1 ? "" : " ");
        fout << "\n";
    }
    fout.close();
}

__global__ void matMulKernel(const double* A, const double* B, double* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        double sum = 0.0;
        for (int k = 0; k < n; ++k)
            sum += A[row * n + k] * B[k * n + col];
        C[row * n + col] = sum;
    }
}

int main(int argc, char* argv[]) {
    int n = 0;
    int block_size = 16;
    
    if (argc >= 2) n = atoi(argv[1]);
    if (argc >= 3) block_size = atoi(argv[2]);
    
    if (n <= 0) {
        cerr << "Использование: " << argv[0] << " <размер> [block_size]" << endl;
        return 1;
    }

    string fileA = "A" + to_string(n) + ".txt";
    string fileB = "B" + to_string(n) + ".txt";
    
    int n1, n2;
    auto A_2d = readMatrix(fileA, n1);
    auto B_2d = readMatrix(fileB, n2);
    
    if (n1 != n2 || n1 != n) {
        cerr << "Ошибка: размеры матриц не совпадают" << endl;
        return 1;
    }

    vector<double> A_flat(n * n), B_flat(n * n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            A_flat[i * n + j] = A_2d[i][j];
            B_flat[i * n + j] = B_2d[i][j];
        }

    double *d_A, *d_B, *d_C;
    size_t bytes = n * n * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, A_flat.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B_flat.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(block_size, block_size);
    dim3 grid((n + block_size - 1) / block_size, (n + block_size - 1) / block_size);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    matMulKernel<<<grid, block>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernel_time_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_time_ms, start, stop));

    vector<double> C_flat(n * n);
    CUDA_CHECK(cudaMemcpy(C_flat.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    vector<vector<double>> C_2d(n, vector<double>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            C_2d[i][j] = C_flat[i * n + j];

    writeMatrix("result.txt", C_2d, kernel_time_ms);

    long long flops = 2LL * n * n * n;
    double gflops = (flops / 1e9) / (kernel_time_ms / 1000.0);

    cout << "Размер матрицы: " << n << "x" << n << endl;
    cout << "Блок: " << block_size << "x" << block_size << endl;
    cout << "Время: " << fixed << setprecision(4) << kernel_time_ms << " мс" << endl;
    cout << "FLOPs: " << flops << endl;
    cout << "Производительность: " << fixed << setprecision(2) << gflops << " GFLOPS" << endl;

    return 0;
}

