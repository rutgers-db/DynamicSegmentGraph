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