#include "benchmark.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>

int main() {
  CudaStream stream;
  const int n_pings = 10;
  std::ofstream out("host_device_result.txt");
  out << "#size\ttime\terr\tbandwidth [GB/s]\terr\n\n";

  for (auto size :
       std::vector<unsigned>{1, 100, 100000, 1000000, 100000000, 1000000000}) {
    char *v1 = (char*)malloc(size);
    char *v2;
    cudaMalloc((void **)&v2, size);
    std::fill_n(v1, size, 0);

    const int n_meas = size <= 100 ? 1000 : 10;

    auto times = benchmarkCuda(
        [&] {
          for (int i = 0; i < n_pings; ++i) {
            cudaMemcpyAsync(v2, v1, size, cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(v1, v2, size, cudaMemcpyDeviceToHost, stream);
          }
        },
        stream, n_meas);

    delete[] v1;
    cudaFree(v2);

    auto [mean, err] = meanAndStdErr(times);
    std::cout << "Size " << size << "\nTime " << mean << " +- " << err;

    std::vector<double> perfs(times.size());
    std::transform(times.begin(), times.end(), perfs.begin(),
                   [=](auto t) { return 2. * size / t * 1e-9; });
    auto [pmean, perr] = meanAndStdErr(perfs);

    std::cout << "\nPerf " << pmean << " +- " << perr << std::endl;

    out << mean << "\t" << err << "\t" << pmean << "\t" << perr << "\n";
  }
}
