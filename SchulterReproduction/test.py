import sys

num_val_steps=0

for i in range(int(sys.argv[1])):
	num_val_steps = num_val_steps + i*i

print(num_val_steps)