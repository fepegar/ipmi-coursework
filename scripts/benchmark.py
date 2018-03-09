import time
from pathlib import Path
import multiprocessing as mp

from ipmi.registration import register

images_dir = Path('/', 'tmp', 'parallel')
floating_paths = list(images_dir.glob('*.nii.gz'))
ref_path = floating_paths.pop()


## Multiprocessing ##
start = time.time()
processes = []

for flo_path in floating_paths:
    processes.append(mp.Process(target=register, args=(ref_path, flo_path)))

# Run processes
for p in processes:
    p.start()

# Exit the completed processes
for p in processes:
    p.join()

stop = time.time()
mp_time = int(stop - start)



start = time.time()
pool = mp.Pool(processes=4)
for flo_path in floating_paths:
    pool.apply(register, args=(ref_path, flo_path))
stop = time.time()
mp_pool_apply_time = int(stop - start)


start = time.time()
pool = mp.Pool(processes=4)
for flo_path in floating_paths:
    pool.apply_async(register, args=(ref_path, flo_path))
stop = time.time()
mp_pool_apply_async_time = int(stop - start)



## Single process ##
start = time.time()
for flo_path in floating_paths:
    register(ref_path, flo_path)
stop = time.time()
sp_time = int(stop - start)


print('Multiprocessing time (using Process):', mp_time)
print('Multiprocessing time (using Pool.apply):', mp_pool_apply_time)
print('Multiprocessing time (using Pool.apply_async):', mp_pool_apply_async_time)
print('Single process time:', sp_time)

## Output:
# Multiprocessing time (using Process): 871
# Multiprocessing time (using Pool.apply): 1190
# Multiprocessing time (using Pool.apply_async): 0
# Single process time: 1290
