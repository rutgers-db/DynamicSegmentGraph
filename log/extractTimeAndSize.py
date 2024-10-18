import sys

def extract_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    batch_times = []
    sum_forward_nn = []
    
    for line in lines:
        if 'batch need' in line:
            # Extract time before 'batch need'
            time_part = line.split('batch need')[0].split()[-1]
            batch_times.append(float(time_part))
        if 'Sum of forward nn #' in line:
            # Extract the number after 'Sum of forward nn #'
            sum_part = line.split('Sum of forward nn #: ')[1]
            sum_forward_nn.append(int(sum_part))
    
    return batch_times, sum_forward_nn

def calculate_difference(sum_forward_nn):
    differences = []
    differences.append(sum_forward_nn[0])
    for i in range(1, len(sum_forward_nn)):
        differences.append(sum_forward_nn[i] - sum_forward_nn[i - 1])
    return differences


if len(sys.argv) != 2:
    print("Usage: python script.py <file_path>")

file_path = sys.argv[1]
batch_times, sum_forward_nn = extract_data(file_path)

print(batch_times)
print(calculate_difference(sum_forward_nn))

