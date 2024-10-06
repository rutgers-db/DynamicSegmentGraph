import pickle
from collections import defaultdict

def process_lines(file_path, load_dominance=True):
    """
    处理文件中的每一行数据，解析并存储邻居信息及支配关系。
    
    :param file_path: 文件路径 ('/Users/zhencan/WorkPlace/Serf_V2/simulation/sample_data/sampled_neighbors_domination.txt')
    :param load_dominance: 是否加载支配关系，默认为True
    :return: (neighbors, dominance_relations)，其中：
             neighbors 是一个元组列表，每个元组包含 (neighbor_id, distance)；
             dominance_relations 是一个列表，每个元素对应一个点，
             表示哪些点支配当前点。
             
    """
    neighbors = []
    dominance_relations = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        parts = line.strip().split()
        neighbor_id = int(parts[0])
        distance = float(parts[1])
        neighbors.append((neighbor_id, distance))

        # 跳过距离字段，直接处理支配关系
        dominating_points = list(map(int, parts[2:]))
        if load_dominance:
            dominance_relations.append(dominating_points)
        else:
            dominance_relations.append([])

    return neighbors, dominance_relations

def extract_query_ranges_from_gt(file_path):
    query_ranges = []
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line by commas
            parts = line.split(',')
            
            # Extract the second, third, and fourth numbers
            if len(parts) >= 4:
                second = int(parts[1].strip())
                third = int(parts[2].strip())
                fourth = int(parts[3].strip())
                query_ranges.append((second, third, fourth))
    return query_ranges

# Function to save the defaultdict to a file
def save_relaxed_points(file_path, relaxed_points):
    with open(file_path, 'wb') as file:
        pickle.dump(dict(relaxed_points), file)

# Function to load the defaultdict from a file
def load_relaxed_points(file_path):
    def default_4_tuple():
        return (0, 0, 0, 0)

    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)
    return defaultdict(default_4_tuple, loaded_data)