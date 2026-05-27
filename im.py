#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import os

CSV_FILE = "results_table.csv"
OUTPUT_DIR = "image"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_FILE)

df['Volume_GFLOPs'] = (2 * df['Matrix_Size']**3) / 1e9

PROC_STYLES = {
    1: {'color': 'red', 'marker': 'o', 'label': '1 proc'},
    2: {'color': 'blue', 'marker': 's', 'label': '2 procs'},
    4: {'color': 'green', 'marker': '^', 'label': '4 procs'},
    8: {'color': 'orange', 'marker': 'v', 'label': '8 procs'},
}

plt.figure(figsize=(10, 6))
for procs in sorted(df['Processes'].unique()):
    subset = df[df['Processes'] == procs]
    style = PROC_STYLES.get(procs, {'color': 'gray', 'marker': 'x', 'label': f'{procs} procs'})
    plt.plot(subset['Matrix_Size'], subset['Time_ms'], marker=style['marker'],
             color=style['color'], label=style['label'], linewidth=2, markersize=6)

plt.xlabel('Размер матрицы (N × N)', fontsize=12)
plt.ylabel('Время выполнения (мс)', fontsize=12)
plt.title('Зависимость времени выполнения от размера матрицы (MPI)', fontsize=14)
plt.legend(title='Число процессов', fontsize=10)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "time_vs_size.png"), dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 6))
for procs in sorted(df['Processes'].unique()):
    subset = df[df['Processes'] == procs]
    style = PROC_STYLES.get(procs, {'color': 'gray', 'marker': 'x', 'label': f'{procs} procs'})
    plt.plot(subset['Volume_GFLOPs'], subset['Time_ms'], marker=style['marker'],
             color=style['color'], label=style['label'], linewidth=2, markersize=6)

plt.xlabel('Вычислительный объём (GFLOPs)', fontsize=12)
plt.ylabel('Время выполнения (мс)', fontsize=12)
plt.title('Зависимость времени выполнения от вычислительного объёма (MPI)', fontsize=14)
plt.legend(title='Число процессов', fontsize=10)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "time_vs_volume.png"), dpi=300, bbox_inches='tight')
plt.close()

print(f"Графики сохранены в {OUTPUT_DIR}/")

