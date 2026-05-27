import sys
import numpy as np


def load_cpp_matrix(filepath):
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        n = int(lines[0].strip())
        mat = []
        for i in range(1, n + 1):
            mat.append([float(x) for x in lines[i].split()])
        return np.array(mat), n
    except Exception as e:
        print(f"Ошибка: {e}")
        return None, None


def load_input_matrix(filepath):
    try:
        with open(filepath, 'r') as f:
            data = f.read().split()
        it = iter(data)
        n = int(next(it))
        mat = [[float(next(it)) for _ in range(n)] for _ in range(n)]
        return np.array(mat)
    except Exception as e:
        print(f"Ошибка: {e}")
        return None


def main():
    if len(sys.argv) != 4:
        print("Использование: python verify.py <A> <B> <result>")
        sys.exit(1)

    path_a, path_b, path_res = sys.argv[1], sys.argv[2], sys.argv[3]

    print("--- Верификация ---")

    A = load_input_matrix(path_a)
    B = load_input_matrix(path_b)
    C_cpp, n = load_cpp_matrix(path_res)

    if A is None or B is None or C_cpp is None:
        print("ВЕРИФИКАЦИЯ: НЕ УДАЛАСЬ (ошибка чтения)")
        sys.exit(1)

    C_np = np.dot(A, B)

    if np.allclose(C_cpp, C_np, rtol=1e-5, atol=1e-8):
        print("ВЕРИФИКАЦИЯ: УСПЕШНО")
        print(f"Размер: {n}x{n}")
        print(f"Макс. отклонение: {np.max(np.abs(C_cpp - C_np))}")
        sys.exit(0)
    else:
        print("ВЕРИФИКАЦИЯ: НЕ УДАЛАСЬ")
        print(f"Макс. отклонение: {np.max(np.abs(C_cpp - C_np))}")
        sys.exit(1)


if __name__ == "__main__":
    main()

