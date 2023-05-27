import matplotlib.pyplot as plt
import numpy as np

###### load

quantum = np.fromfile(open("/mnt/c/Users/blash/OneDrive - Deakin University/Honours/Project/gnu-radio/quantum.bin"), dtype=np.float32)
raw = np.fromfile(open("/mnt/c/Users/blash/OneDrive - Deakin University/Honours/Project/gnu-radio/raw.bin"), dtype=np.float32)

# Stats

print(len(quantum))
print(len(raw))

###### store

min_length = min(len(quantum), len(raw))
data = np.column_stack((raw[:min_length], quantum[:min_length]))
np.savetxt("data.csv", data, delimiter=",", fmt="%.8f")
###### Display

# Load the data from the CSV file
data = np.genfromtxt('data.csv', delimiter=',')

# Extract the columns
raw = data[:, 0]
quantum = data[:, 1]

# Create the x-axis values
x = np.arange(len(raw))

# Plot the data
plt.plot(x, raw, label='Raw')
plt.plot(x, quantum, label='Quantum')

# Set the chart title and labels
plt.title('Raw and Quantum Data')
plt.xlabel('Sample Index')
plt.ylabel('Value')

# Show a legend
plt.legend()

# Display the chart
plt.show()
import matplotlib.pyplot as plt
import numpy as np

###### load

quantum = np.fromfile(open("/mnt/c/Users/blash/OneDrive - Deakin University/Honours/Project/gnu-radio/quantum.bin"), dtype=np.float32)
raw = np.fromfile(open("/mnt/c/Users/blash/OneDrive - Deakin University/Honours/Project/gnu-radio/raw.bin"), dtype=np.float32)

# Stats

print(len(quantum))
print(len(raw))

###### store

min_length = min(len(quantum), len(raw))
data = np.column_stack((raw[:min_length], quantum[:min_length]))
np.savetxt("/mnt/c/Users/blash/OneDrive - Deakin University/Honours/Project/gnu-radio/data.csv", data, delimiter=",", fmt="%.8f")
###### Display

# Load the data from the CSV file
data = np.genfromtxt('/mnt/c/Users/blash/OneDrive - Deakin University/Honours/Project/gnu-radio/data.csv', delimiter=',')

# Extract the columns
raw = data[:, 0]
quantum = data[:, 1]

# Create the x-axis values
x = np.arange(len(raw))

# Plot the data
plt.plot(x, raw, label='Raw')
plt.plot(x, quantum, label='Quantum')

# Set the chart title and labels
plt.title('Raw and Quantum Data')
plt.xlabel('Index')
plt.ylabel('Value')

# Show a legend
plt.legend()

# Display the chart
plt.show()
