import sys
import numpy as np


def write_random_matrix(filepath, size):
    data = np.random.rand(size, size)
    with open(filepath, 'w') as f:
        f.write(f"{size}\n")
        for row in data:
            f.write(" ".join(map(str, row)) + "\n")
    print(f"[OK] Матрица {size}x{size} сохранена в {filepath}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Запуск: python generate.py <N> <файл_A> <файл_B>")
        sys.exit(1)

    N = int(sys.argv[1])
    write_random_matrix(sys.argv[2], N)
    write_random_matrix(sys.argv[3], N)
