import csv
import numpy as np

# Matriz de confusão
confusion_matrix = np.array([
    [10, 4, 2, 1],
    [1, 15, 2, 0],
    [2, 3, 20, 5],
    [4, 1, 2, 50]
])

# Nomes das classes
class_names = ['A', 'B', 'C', 'D']
# Arquivo CSV de saída
output_csv = 'classification_metrics.csv'

# Calcular as métricas para cada classe
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Escrever o cabeçalho do CSV
    writer.writerow(['Class', 'Precision', 'Recall', 'F1Score', 'TVP', 'TFN', 'TFP', 'TVN'])

    # Iterar sobre cada classe para calcular as métricas
    for i, class_name in enumerate(class_names):
        TP = confusion_matrix[i, i]
        FP = sum(confusion_matrix[:, i]) - TP
        FN = sum(confusion_matrix[i, :]) - TP
        TN = confusion_matrix.sum() - TP - FP - FN
        
        Precision = TP / (TP + FP) if TP + FP > 0 else 0
        Recall = TP / (TP + FN) if TP + FN > 0 else 0
        F1Score = 2 * Precision * Recall / (Precision + Recall) if Precision + Recall > 0 else 0
        TVP = Recall
        TFN = TN / (TN + FP) if TN + FP > 0 else 0
        TFP = FP / (FP + TN) if FP + TN > 0 else 0
        TVN = FN / (FN + TP) if FN + TP > 0 else 0
        
        # Escrever as métricas da classe atual no arquivo CSV
        writer.writerow([class_name, Precision, Recall, F1Score, TVP, TFN, TFP, TVN])