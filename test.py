import os
import subprocess
import csv
import re
import sys
import time


def execute_test(dimension):
    print(f"Тест для {dimension}x{dimension}...")

    for f in ["mat_a.txt", "mat_b.txt", "res.txt"]:
        if os.path.exists(f):
            os.remove(f)

    print(f"  Генерация...")
    gen_cmd = [sys.executable, "generate.py", str(dimension), "mat_a.txt", "mat_b.txt"]
    proc = subprocess.run(gen_cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  Ошибка: {proc.stderr}")
        return None

    time.sleep(0.1)

    print(f"  Вычисление...")
    exe = "matrix_mult.exe" if os.name == 'nt' else "./matrix_mult"
    proc = subprocess.run([exe, "mat_a.txt", "mat_b.txt", "res.txt"], capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  Ошибка: {proc.stderr}")
        return None

    time.sleep(0.1)

    try:
        with open("res.txt", "r", encoding="utf-8") as f:
            content = f.read()

        time_match = re.search(r"TIME_MS:\s*([\d.]+)", content)
        flops_match = re.search(r"FLOPS:\s*(\d+)", content)
        mem_match = re.search(r"MEMORY_BYTES:\s*(\d+)", content)

        if time_match and flops_match and mem_match:
            time_sec = float(time_match.group(1)) / 1000.0
            return {
                'dim': dimension,
                'time': time_sec,
                'ops': int(flops_match.group(1)),
                'mem': int(mem_match.group(1))
            }
        else:
            print(f"  Метаданные не найдены")
            return None

    except Exception as e:
        print(f"Ошибка: {e}")
        return None


def main():
    dimensions = [200, 400, 800, 1200, 1600, 2000]
    results = []

    print("Запуск тестов\n")

    for d in dimensions:
        data = execute_test(d)
        if data:
            results.append(data)
            print(f"✓ {d}: {data['time']:.6f} сек, {data['ops']} ops, {data['mem']} байт\n")

    if results:
        csv_path = "matrix_stats.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(["Размер матрицы", "Время выполнения(сек.)", "Количество операций", "Объём данных"])
            for r in results:
                writer.writerow([r['dim'], f"{r['time']:.7f}", r['ops'], r['mem']])
        print(f"Сохранено в {csv_path}")

    for f in ["mat_a.txt", "mat_b.txt", "res.txt"]:
        if os.path.exists(f):
            os.remove(f)


if __name__ == "__main__":
    main()


