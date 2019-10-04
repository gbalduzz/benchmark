#pragma once

#include <chrono>
#include <cmath>
#include <vector>
#include <tuple>

#include "cuda_event.hpp"
#include "cuda_stream.hpp"

template <typename T> auto meanAndStdErr(const std::vector<T> &v) {
  double sum = 0, sum2 = 0;
  for (auto x : v) {
    sum += x;
    sum2 += x * x;
  }

  const double mean = sum / v.size();
  const double var2 = sum2 / v.size() - mean * mean;
  return std::make_tuple(mean, std::sqrt(var2 / v.size()));
}

template <class F> auto benchmarkCuda(F &&f, CudaStream &stream, int n_times) {
  CudaEvent start, stop;
  std::vector<double> times(n_times);
  for (auto &time : times) {
    start.record(stream);
    f();
    stop.record(stream);
    time = elapsedTime(stop, start);
  }

  return times;
};

template <class F> auto benchmark(F &&f, int n_times) {
    std::vector<double> times(n_times);
    for (auto &time : times) {
        const auto start = std::chrono::high_resolution_clock::now();
        f();
        const auto stop = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start);
        time = duration.count();
    }

    return times;
};
