/**
 * @file utils.h
 * @brief 提供了一系列实用工具函数。
 */

 #pragma once

 #include <assert.h>
 #include <ctime>
 #include <fstream>
 #include <functional>
 #include <iostream>
 #include <numeric>
 #include <queue>
 #include <random>
 #include <sstream>
 #include <unordered_map>
 #include <unordered_set>
 #include <vector>
 #include <algorithm>
 #include <sys/time.h>
 
 #ifdef __linux__
 #include "sys/sysinfo.h"
 #include "sys/types.h"
 #elif __APPLE__
 #include <mach/mach_host.h>
 #include <mach/mach_init.h>
 #include <mach/mach_types.h>
 #include <mach/vm_statistics.h>
 #endif
 
 // 使用标准库中的命名空间元素
 using std::cout;
 using std::endl;
 using std::getline;
 using std::ifstream;
 using std::ios;
 using std::make_pair;
 using std::pair;
 using std::string;
 using std::vector;
 
 

 /**
  * 积累时间差值。
  *
  * @param t2 结束时间。
  * @param t1 开始时间。
  * @param val_time 时间差值。
  */
 void AccumulateTime(timeval &t2, timeval &t1, double &val_time)
 {
    val_time += (t2.tv_sec - t1.tv_sec +
                 (t2.tv_usec - t1.tv_usec) * 1.0 / CLOCKS_PER_SEC);
 }

 /**
  * 计算并记录时间差值。
  *
  * @param t1 开始时间。
  * @param t2 结束时间。
  * @param val_time 时间差值。
  */
 void CountTime(timeval &t1, timeval &t2, double &val_time)
 {
    val_time = 0;
    val_time += (t2.tv_sec - t1.tv_sec +
                 (t2.tv_usec - t1.tv_usec) * 1.0 / CLOCKS_PER_SEC);
 }

 /**
  * 返回两次时间测量的时间差值。
  *
  * @param t1 开始时间。
  * @param t2 结束时间。
  * @return 时间差值。
  */
 double CountTime(timeval &t1, timeval &t2)
 {
    double val_time = 0.0;
    val_time += (t2.tv_sec - t1.tv_sec +
                 (t2.tv_usec - t1.tv_usec) * 1.0 / CLOCKS_PER_SEC);
    return val_time;
 }
 
 void logTime(timeval &begin, timeval &end, const string &log){
    gettimeofday(&end, NULL);
    fprintf(stdout, ("# " + log + ": %.7fs\n").c_str(),
            end.tv_sec - begin.tv_sec +
                (end.tv_usec - begin.tv_usec) * 1.0 / CLOCKS_PER_SEC);
 };
 /**
  * 计算精度。
  *
  * @param truth 真实结果。
  * @param pred 预测结果。
  * @return 精度值。
  */
 double countPrecision(const vector<int> &truth, const vector<int> &pred);
 
 /**
  * 计算近似比率。
  *
  * @param raw_data 原始数据集。
  * @param truth 真实结果。
  * @param pred 预测结果。
  * @param query 查询点。
  * @return 近似比率。
  */
 double countApproximationRatio(const vector<vector<float>> &raw_data,
                                const vector<int> &truth,
                                const vector<int> &pred,
                                const vector<float> &query);
 
 /**
  * 打印内存使用情况。
  */
 void print_memory();
 
 /**
  * 记录当前内存使用情况。
  *
  * @param memory 当前内存使用量引用。
  */
 void record_memory(long long &memory);
 