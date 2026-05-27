import subprocess
import numpy as np
import sys
import os
import csv
import re

EXEC = "matmul_mpi"
MATRIX_SIZES = [200, 400, 800, 1200, 1600, 2000]
PROCESS_COUNTS = [1, 2, 4, 8]
RESULTS_FILE = "results_table.csv"
VERIFY_UP_TO = 1200

def run_program(n, num_procs):
    cmd = ["mpirun", "-np", str(num_procs), EXEC, str(n)]
    print(f"  Запуск: {' '.join(cmd)}", flush=True)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        print("Таймаут")
        return None

    if result.returncode != 0:
        print(f"Ошибка: {result.stderr}")
        return None

    metrics = {}
    for line in result.stdout.split('\n'):
        if 'Время:' in line:
            match = re.search(r'([\d.]+)\s*мс', line)
            if match:
                metrics['time_ms'] = float(match.group(1))
        elif 'Производительность:' in line:
            match = re.search(r'([\d.]+)\s*GFLOPS', line)
            if match:
                metrics['gflops'] = float(match.group(1))

    if 'time_ms' not in metrics or 'gflops' not in metrics:
        print("Ошибка парсинга")
        return None
    return metrics

def read_matrix(filepath):
    with open(filepath, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip() and not line.strip().startswith('#')]
    data = []
    for line in lines:
        data.extend(line.split())
    n = int(data[0])
    values = np.array(data[1:], dtype=np.float64)
    return values.reshape(n, n)

def verify_results(n):
    try:
        A = read_matrix(f"A{n}.txt")
        B = read_matrix(f"B{n}.txt")
        C_cpp = read_matrix("result.txt")
        C_ref = np.dot(A, B)
        is_match = np.allclose(C_cpp, C_ref, atol=1e-5, rtol=1e-5)
        max_err = np.max(np.abs(C_cpp - C_ref))
        print(f"Верификация: {'OK' if is_match else 'FAIL'}, погрешность: {max_err:.2e}")
        return is_match
    except Exception as e:
        print(f"Ошибка: {e}")
        return False

def init_csv():
    with open(RESULTS_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Matrix_Size', 'Processes', 'Time_ms', 'GFLOPS', 'Verified'])

def save_result(n, procs, metrics, verified):
    with open(RESULTS_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([n, procs, f"{metrics['time_ms']:.4f}", f"{metrics['gflops']:.2f}", 'YES' if verified else 'NO'])

def main():
    print("=== Тестирование MPI ===\n")

    if not os.path.isfile(EXEC):
        print(f"Ошибка: {EXEC} не найден")
        sys.exit(1)

    init_csv()
    success = 0

    for n in MATRIX_SIZES:
        print(f"\nРазмер: {n}x{n}")

        if not os.path.isfile(f"A{n}.txt") or not os.path.isfile(f"B{n}.txt"):
            print(f"Файлы A{n}.txt или B{n}.txt не найдены")
            continue

        for procs in PROCESS_COUNTS:
            print(f"  Процессов: {procs}...", end=" ", flush=True)

            metrics = run_program(n, procs)
            if not metrics:
                print("- ошибка")
                continue

            verified = verify_results(n) if n <= VERIFY_UP_TO else True
            save_result(n, procs, metrics, verified)
            success += 1

            print(f"{metrics['time_ms']:.2f} мс, {metrics['gflops']:.2f} GFLOPS")

    print(f"\nГотово. Успешно: {success}. Результаты в {RESULTS_FILE}")

if __name__ == "__main__":
    main()

