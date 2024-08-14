import heapq
import random
import timeit
from sortedcontainers import SortedList
# 发现是sortedcontainers比heapq慢，可能是sortedcontainers的插入和删除操作比heapq慢吧 维护sortedcontainers的代价还高一些的
# 一个0.000330 一个0.00021
# class KthLargest:
#     def __init__(self, k):
#         """
#         初始化KthLargest对象。
        
#         @param k: int 类型，表示要维护的最大元素数量。
#         """
#         self.k = k
#         self.sortedList = SortedList()  # SortedList

#     def add(self, i):
#         """
#         添加一个新的元素到SortedSet中并更新相关数据结构。
        
#         @param i: 要添加的新元素。
#         """
#         if len(self.sortedList) < self.k:
#             self.sortedList.add(i)
#         else:
#             smallest = self.sortedList[0]
#             if i > smallest:
#                 self.sortedList.pop(0)
#                 self.sortedList.add(i)
#             else:
#                 return -1
        
#         return self.sortedList.bisect_left(i)

#     def show(self):
#         print(self.sortedList)

class KthLargest:
    def __init__(self, k):
        self.k = k
        self.heap = []  # 最小堆
    def add(self, i):
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, i)
        else:
            if i > self.heap[0]:
                heapq.heappop(self.heap)
                heapq.heappush(self.heap, i)
                sorted_list = sorted(self.heap)
                # 查找元素i在有序列表中的位置索引
                return sorted_list.index(i)
        return -1
    def show(self):
            print(sorted(self.heap))
    
def test_kth_largest():
    k = 32
    kl = KthLargest(k)
    elements_num = 100
    # 生成1000个随机整数
    random.seed(42)
    numbers = [random.randint(1, elements_num) for _ in range(elements_num)]
    
    start_time = timeit.default_timer()
    
    for num in numbers:
        kl.add(num)
    
    end_time = timeit.default_timer()
    
    elapsed_time = end_time - start_time
    
    print(f"Total time taken to process {len(numbers)} numbers: {elapsed_time:.6f} seconds")
    kl.show()

if __name__ == "__main__":
    test_kth_largest()
