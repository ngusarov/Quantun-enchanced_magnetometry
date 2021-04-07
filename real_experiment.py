detuning_MHz = 2  # соответствует полю
...
ramsey_values = []
for i in range(len(times_us)):
    ramsey_values.append(ramsey_results.get_memory(i)[qubit] * scale_factor)
# Берем время взаимодействия. Берем ramsey_results.get_memory(i)[qubit] * scale_factor и смотрим
# по фазовой плоскости с прямой -- это 1 или 0.