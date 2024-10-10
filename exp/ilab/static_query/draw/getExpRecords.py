import re
import matplotlib.pyplot as plt

def extract_metrics(data):
    # Define the regex patterns
    search_ef_pattern = r"Search ef: (\d+)"
    range_pattern = r"range: (\d+)\s+recall: ([\d\.]+)\s+QPS: (\d+)"

    # Find all 'Search ef' sections
    search_efs = re.findall(search_ef_pattern, data)

    # Split data by 'Search ef' sections
    sections = re.split(search_ef_pattern, data)[1:]

    # Extract information for each 'Search ef'
    results = []
    for i in range(0, len(sections), 2):
        search_ef = sections[i].strip()
        content = sections[i + 1]
        ranges = re.findall(range_pattern, content)
        
        search_ef_data = {
            "Search ef": search_ef,
            "Ranges": []
        }
        for r in ranges:
            search_ef_data["Ranges"].append({
                "Range": r[0],
                "Recall": r[1],
                "QPS": r[2]
            })
        results.append(search_ef_data)
    
    return results

# Read data from a txt file
def read_data_from_file(filename):
    with open(filename, 'r') as file:
        return file.read()

# Example usage
# filename1 = '../yt8m-audio/1m_16_750_100/seg.txt'
# filename2 = '../yt8m-audio/1m_16_750_100/compact.txt'
filename1 = '../wiki/1m/seg.txt'
filename2 = '../wiki/1m/compact.txt'

data1 = read_data_from_file(filename1)
data2 = read_data_from_file(filename2)

results1 = extract_metrics(data1)
results2 = extract_metrics(data2)

# Prepare data for plotting
graph_data = {}
for results, label in [(results1, 'seg'), (results2, 'compact')]:
    for result in results:
        for r in result['Ranges']:
            range_value = r['Range']
            if range_value not in graph_data:
                graph_data[range_value] = {'seg': {'Recall': [], 'QPS': []}, 'compact': {'Recall': [], 'QPS': []}}
            graph_data[range_value][label]['Recall'].append(float(r['Recall']))
            graph_data[range_value][label]['QPS'].append(float(r['QPS']))

# Plotting the graphs
for range_value, data in graph_data.items():
    plt.figure(figsize=(10, 6))
    
    # Plotting both lines for seg and compact
    plt.plot(data['seg']['Recall'], data['seg']['QPS'], marker='o', linestyle='-', color='b', label='seg')
    plt.plot(data['compact']['Recall'], data['compact']['QPS'], marker='x', linestyle='--', color='r', label='compact')

    # Setting y-axis to logarithmic scale
    plt.yscale('log')
    # plt.gca().invert_yaxis()  # Ensure the y-axis is not auto-reversed

    # Adding labels, title, and legend
    plt.xlabel('Recall')
    plt.ylabel('QPS (Log Scale)')
    plt.title(f'Recall vs QPS (Logarithmic Scale) for Range {range_value}')
    plt.legend()
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    
    # Save the plot
    plt.savefig(f'query_range{range_value}.png')
    plt.close()

print("Graphs have been saved as PNG files.")