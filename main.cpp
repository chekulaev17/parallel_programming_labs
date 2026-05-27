#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>
#include <cstdlib>
#include <omp.h>

using namespace std;

vector<vector<double>> readMatrix(const string& filename, int& n) {
    ifstream fin(filename);
    if (!fin.is_open()) {
        cerr << "Ошибка: не удалось открыть файл '" << filename << "'" << endl;
        exit(1);
    }

    if (!(fin >> n) || n <= 0) {
        cerr << "Ошибка: неверный размер матрицы в '" << filename << "'" << endl;
        exit(1);
    }

    vector<vector<double>> mat(n, vector<double>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (!(fin >> mat[i][j])) {
                cerr << "Ошибка: недостаточно данных в '" << filename << "'" << endl;
                exit(1);
            }
        }
    }
    fin.close();
    return mat;
}

void writeMatrix(const string& filename, const vector<vector<double>>& mat, double elapsed_ms = -1.0) {
    ofstream fout(filename);
    if (!fout.is_open()) {
        cerr << "Ошибка: не удалось создать файл '" << filename << "'" << endl;
        exit(1);
    }

    int n = mat.size();
    fout << n << "\n";
    fout << fixed << setprecision(10);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            fout << mat[i][j] << (j == n - 1 ? "" : " ");
        }
        fout << "\n";
    }

    fout << "# METADATA_START" << endl;
    fout << "TIME_MS: " << fixed << setprecision(6) << elapsed_ms << endl;
    fout << "# METADATA_END" << endl;

    fout.close();
}

int main(int argc, char* argv[]) {
    int n = 0;
    int num_threads = 1;

    if (argc >= 2) n = atoi(argv[1]);
    if (argc >= 3) num_threads = atoi(argv[2]);

    if (n <= 0) {
        cerr << "Использование: " << argv[0] << " <размер_матрицы> [число_потоков]" << endl;
        return 1;
    }

    string fileA = "A" + to_string(n) + ".txt";
    string fileB = "B" + to_string(n) + ".txt";

    int n1, n2;
    vector<vector<double>> A = readMatrix(fileA, n1);
    vector<vector<double>> B = readMatrix(fileB, n2);

    if (n1 != n2) {
        cerr << "Ошибка: размеры матриц не совпадают" << endl;
        return 1;
    }
    n = n1;

    vector<vector<double>> BT(n, vector<double>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            BT[j][i] = B[i][j];

    vector<vector<double>> C(n, vector<double>(n, 0.0));

    omp_set_num_threads(num_threads);
    double start = omp_get_wtime();

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += A[i][k] * BT[j][k];
            }
            C[i][j] = sum;
        }
    }

    double elapsed_ms = (omp_get_wtime() - start) * 1000.0;
    long long flops = 2LL * n * n * n;

    writeMatrix("result.txt", C, elapsed_ms);

    cout << "Размер матрицы: " << n << "x" << n << endl;
    cout << "Потоков: " << num_threads << endl;
    cout << "Время: " << fixed << setprecision(4) << elapsed_ms << " мс" << endl;
    cout << "FLOPs: " << flops << endl;
    cout << "Производительность: " << fixed << setprecision(2) << (flops / 1e9) / (elapsed_ms / 1000.0) << " GFLOPS" << endl;

    return 0;
}

