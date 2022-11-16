# Python program to illustrate the concept
# of threading
# importing the threading module
import threading
import queue
import time
import random

global_queue = queue.Queue()

def worker(num, queue):
	for i in range(1, 10):
		time.sleep(random.randrange(1, 3))
		queue.put(num, block=False, timeout=None)

if __name__ =="__main__":
	# creating thread
	threads = []

	for i in range(1, 10):
		thread = threading.Thread(target=worker, args=(i, global_queue))
		threads.append(thread)

	for thread in threads:
		thread.start()

	while True:
		print(global_queue.get())

	for thread in threads:
		thread.join()



	print("Done!")
