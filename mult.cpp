#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>
#include <windows.h>

using namespace std;

struct Metrics {
    int dimension;
    double duration_ms;
    long long total_ops;
    long long memory_usage;
};

bool loadMatrix(const string& path, vector<vector<double>>& mat, int& dim) {
    ifstream file(path);
    if (!file.is_open()) {
        cerr << "Не удалось открыть " << path << endl;
        return false;
    }

    if (!(file >> dim)) {
        cerr << "Ошибка чтения размера из " << path << endl;
        return false;
    }

    mat.resize(dim, vector<double>(dim));
    for (int i = 0; i < dim; ++i)
        for (int j = 0; j < dim; ++j)
            if (!(file >> mat[i][j])) {
                cerr << "Ошибка чтения элемента [" << i << "][" << j << "]" << endl;
                return false;
            }
    file.close();
    return true;
}

bool saveResult(const string& path, const vector<vector<double>>& mat, const Metrics& m) {
    ofstream file(path);
    if (!file.is_open()) {
        cerr << "Не удалось создать " << path << endl;
        return false;
    }

    int dim = mat.size();
    file << dim << endl;
    file << fixed << setprecision(10);

    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j)
            file << mat[i][j] << " ";
        file << endl;
    }

    file << "# METADATA_START" << endl;
    file << "SIZE: " << m.dimension << endl;
    file << "TIME_MS: " << m.duration_ms << endl;
    file << "FLOPS: " << m.total_ops << endl;
    file << "MEMORY_BYTES: " << m.memory_usage << endl;
    file << "# METADATA_END" << endl;

    file.close();
    return true;
}

void multiplyMatrices(const vector<vector<double>>& A, const vector<vector<double>>& B, vector<vector<double>>& C, int dim) {
    C.assign(dim, vector<double>(dim, 0.0));
    for (int i = 0; i < dim; ++i)
        for (int j = 0; j < dim; ++j)
            for (int k = 0; k < dim; ++k)
                C[i][j] += A[i][k] * B[k][j];
}

int main(int argc, char* argv[]) {
    SetConsoleOutputCP(65001);

    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " <matrix_A> <matrix_B> <result_file>" << endl;
        return 1;
    }

    string fileA = argv[1], fileB = argv[2], fileRes = argv[3];
    vector<vector<double>> A, B, C;
    int dimA, dimB;

    cout << "Загрузка матриц..." << endl;
    if (!loadMatrix(fileA, A, dimA)) return 1;
    if (!loadMatrix(fileB, B, dimB)) return 1;

    if (dimA != dimB) {
        cerr << "Размеры матриц не совпадают!" << endl;
        return 1;
    }
    int dim = dimA;

    cout << "Размер: " << dim << "x" << dim << endl;
    cout << "Умножение..." << endl;

    auto start = chrono::high_resolution_clock::now();
    multiplyMatrices(A, B, C, dim);
    auto end = chrono::high_resolution_clock::now();

    double duration_ms = chrono::duration<double, milli>(end - start).count();
    long long total_ops = 2LL * dim * dim * dim;
    long long memory_usage = 3LL * dim * dim * sizeof(double);

    Metrics m = {dim, duration_ms, total_ops, memory_usage};

    if (!saveResult(fileRes, C, m)) return 1;

    cout << "Результат: " << fileRes << endl;
    cout << "Время: " << duration_ms << " мс" << endl;
    cout << "FLOPs: " << total_ops << endl;
    cout << "Память: " << memory_usage << " байт" << endl;

    return 0;
}

