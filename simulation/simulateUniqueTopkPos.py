import numpy as np

def simulate_expectation(N, k, simulations=50):
    count_valid_i = 0

    for _ in range(simulations):
        # 生成随机数组
        array = np.random.rand(N)
        for i in range(N - 1):
            # 取 i 右边的数字
            right_numbers = array[i + 1:]

            # 计算比a[i]小的数量
            count_smaller = np.sum(right_numbers < array[i])

            # 如果小于k-1, 则count_valid_i加1
            if count_smaller <= k - 1:
                count_valid_i += 1

    # 计算期望
    return count_valid_i / (simulations)

# 设置参数
N = 1 << 11    # 数组长度
k = 8       # k值
expectation = simulate_expectation(N, k)
print(f'The expected number of indices i in {N} length array with at most {k-1} numbers smaller on the right is: {expectation}')
