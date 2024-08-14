#include "../include/common/sorted_array.h"
#include <iostream>

// 测试函数
void test_sorted_array()
{
    const unsigned max_size = 5;

    // 测试 MinSortedArray
    MinSortedArray min_sa(max_size);

    // 添加一系列点
    unsigned min_points[] = {3, 1, 5, 2, 4, 6, 9, 12, 11, 0};
    for (unsigned point : min_points)
    {
        unsigned rank;
        bool added = min_sa.addPoint(point, rank);
        std::cout << "Adding point " << point << " to MinSortedArray: Rank " << rank << ", Added? " << added << std::endl;
    }

    // 打印最终数组
    std::cout << "\nFinal MinSortedArray: ";
    for (unsigned p : min_sa.sorted_arr)
    {
        std::cout << p << " ";
    }
    std::cout << std::endl;

    // 清空数组并再次测试 MaxSortedArray
    min_sa.clear();

    // 测试 MaxSortedArray
    MaxSortedArray max_sa(max_size);

    // 添加一系列点
    unsigned max_points[] = {3, 1, 11, 6, 4, 8, 7, 5, 3, 12};
    for (unsigned point : max_points)
    {
        unsigned rank;
        bool added = max_sa.addPoint(point, rank);
        std::cout << "\nAdding point " << point << " to MaxSortedArray: Rank " << rank << ", Added? " << added << std::endl;
    }

    // 打印最终数组
    std::cout << "\nFinal MaxSortedArray: ";
    for (unsigned p : max_sa.sorted_arr)
    {
        std::cout << p << " ";
    }
    std::cout << std::endl;
}

int main()
{
    test_sorted_array();
    return 0;
}
