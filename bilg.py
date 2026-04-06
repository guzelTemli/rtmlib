import platform
import psutil

print("CPU:", platform.processor())
print("Cores:", psutil.cpu_count())
print("RAM (GB):", round(psutil.virtual_memory().total / (1024**3), 2))