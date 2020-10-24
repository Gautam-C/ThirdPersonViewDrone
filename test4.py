import time
import threading
import concurrent.futures

start = time.perf_counter()

def do_something(seconds):
    print(f"Sleeping for {seconds} second(s)")
    time.sleep(seconds)
    return f"Done sleeping ({seconds})"

# prevents script from running until the threads have finished
# t1.join()
# t2.join()

with concurrent.futures.ThreadPoolExecutor() as executor:
    secs = [5, 4, 3, 2, 1]

    # runs the function for every value in the list in a different thread
    results = executor.map(do_something, secs)

    for result in results:
        print(result)

'''
with concurrent.futures.ThreadPoolExecutor() as executor:

    secs = [5, 4, 3, 2, 1]
    
    results = [executor.submit(do_something, sec) for sec in secs]

    for f in concurrent.futures.as_completed(results):
        print(f.result())

'''
'''
threadList = []

for _ in range(10):
    t = threading.Thread(target=do_something, args=[1.5])
    t.start()
    threadList.append(t)

for thread in threadList:
    thread.join()
'''

finish = time.perf_counter()

print(f"Finished in {round(finish-start, 2)} seconds")
