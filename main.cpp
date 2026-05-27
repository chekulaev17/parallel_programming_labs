#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <string>
#include <cstdlib>
#include <mpi.h>

using namespace std;

vector<vector<double> > readMatrix(const string& filename, int& n) {
    ifstream fin(filename.c_str());
    if (!fin.is_open()) {
        cerr << "Ошибка: не удалось открыть файл '" << filename << "'" << endl;
        exit(1);
    }
    if (!(fin >> n) || n <= 0) {
        cerr << "Ошибка: неверный размер матрицы" << endl;
        exit(1);
    }
    vector<vector<double> > mat(n, vector<double>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (!(fin >> mat[i][j])) {
                cerr << "Ошибка: недостаточно данных" << endl;
                exit(1);
            }
    fin.close();
    return mat;
}

void writeMatrix(const string& filename, const vector<vector<double> >& mat) {
    ofstream fout(filename.c_str());
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

pair<int, int> getRowRange(int rank, int size, int n) {
    int rows_per_proc = n / size;
    int remainder = n % size;
    int start_row = rank * rows_per_proc + min(rank, remainder);
    int local_rows = rows_per_proc + (rank < remainder ? 1 : 0);
    return make_pair(start_row, local_rows);
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 0;
    if (argc >= 2) n = atoi(argv[1]);

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (n <= 0) {
        if (rank == 0) {
            cerr << "Использование: mpirun -np <N> " << argv[0] << " <размер>" << endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    vector<double> A_flat, BT_flat;

    if (rank == 0) {
        string fileA = "A" + to_string(n) + ".txt";
        string fileB = "B" + to_string(n) + ".txt";

        int n1, n2;
        vector<vector<double> > A = readMatrix(fileA, n1);
        vector<vector<double> > B = readMatrix(fileB, n2);

        if (n1 != n2 || n1 != n) {
            cerr << "Ошибка: размеры матриц не совпадают" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        A_flat.resize(n * n);
        BT_flat.resize(n * n);

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                A_flat[i * n + j] = A[i][j];
                BT_flat[i * n + j] = B[j][i];
            }
        }
    } else {
        A_flat.resize(n * n);
        BT_flat.resize(n * n);
    }

    MPI_Bcast(&A_flat[0], n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&BT_flat[0], n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    pair<int, int> range = getRowRange(rank, size, n);
    int start_row = range.first;
    int local_rows = range.second;

    vector<double> C_local(local_rows * n, 0.0);

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    for (int i = 0; i < local_rows; ++i) {
        int ig = start_row + i;
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += A_flat[ig * n + k] * BT_flat[j * n + k];
            }
            C_local[i * n + j] = sum;
        }
    }

    double elapsed_ms = (MPI_Wtime() - start) * 1000.0;

    vector<int> recv_counts(size), displs(size);
    for (int p = 0; p < size; ++p) {
        pair<int, int> pr = getRowRange(p, size, n);
        recv_counts[p] = pr.second * n;
        displs[p] = pr.first * n;
    }

    vector<double> C_flat(n * n);
    MPI_Gatherv(&C_local[0], local_rows * n, MPI_DOUBLE,
                &C_flat[0], &recv_counts[0], &displs[0], MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        vector<vector<double> > C(n, vector<double>(n));
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                C[i][j] = C_flat[i * n + j];

        writeMatrix("result.txt", C);

        long long flops = 2LL * n * n * n;
        double gflops = (flops / 1e9) / (elapsed_ms / 1000.0);

        cout << "Размер матрицы: " << n << "x" << n << endl;
        cout << "Процессов MPI: " << size << endl;
        cout << "Время: " << fixed << setprecision(4) << elapsed_ms << " мс" << endl;
        cout << "FLOPs: " << flops << endl;
        cout << "Производительность: " << fixed << setprecision(2) << gflops << " GFLOPS" << endl;
    }

    MPI_Finalize();
    return 0;
}

