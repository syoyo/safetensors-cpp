#include <iostream>
#include <sstream>
#include <random>
#include <cassert>
#include <cstring>

#if !defined(SAFETENSORS_CPP_NO_IMPLEMENTATION)
#define SAFETENSORS_CPP_IMPLEMENTATION
#endif
#include "safetensors.hh"

// generate n x m 2D array filled with random numbers.
std::vector<float> gen_random(size_t n, size_t m) {

  std::random_device seed_gen;
  std::mt19937 engine(seed_gen());

  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  std::vector<float> result;

  for (size_t i = 0; i < n * m; i++) {
    result.push_back(dist(engine)); 
  }

  return result;
}


static void swap2(unsigned short *val) {
    unsigned short tmp = *val;
    unsigned char *dst = reinterpret_cast<unsigned char *>(val);
    unsigned char *src = reinterpret_cast<unsigned char *>(&tmp);

    dst[0] = src[1];
    dst[1] = src[0];
}


int main(int argc, char **argv) {

  // Directly construct tensor and safetensors.
 
  safetensors::safetensors_t st;

  size_t data_offset_base = 0;

  {
    std::vector<float> weight = gen_random(8, 8);
    size_t dst_offset = st.storage.size();
    size_t sz = sizeof(float) * 8 * 8;
    assert(sz == weight.size() * sizeof(float));

    // expand
    st.storage.resize(dst_offset + sz);
    memcpy(st.storage.data() + dst_offset, weight.data(), sz);

    safetensors::tensor_t tensor;
    tensor.dtype = safetensors::dtype::kFLOAT32;
    tensor.data_offsets[0] = dst_offset;
    tensor.data_offsets[1] = dst_offset + sz;
    tensor.shape.resize(2);
    tensor.shape[0] = 8;
    tensor.shape[1] = 8;

    st.tensors.emplace("weight0", tensor);
  }
  
  {
    // fp16 tensor
    std::vector<float> _weight = gen_random(16, 16);
    size_t dst_offset = st.storage.size();
    size_t sz = sizeof(uint16_t) * 16 * 16;

    std::vector<uint16_t> half_weight;
    half_weight.resize(sz);

    for (size_t i = 0; i < sz; i++) {
      uint16_t val = safetensors::float_to_fp16(_weight[i]);
  
      // To avoid annoying endianness issue, use mempcy()
      memcpy(&half_weight[i], &val, 2);
    }
    
    assert(sz == half_weight.size() * sizeof(uint16_t));

    // expand
    st.storage.resize(dst_offset + sz);
    memcpy(st.storage.data() + dst_offset, half_weight.data(), sz);

    safetensors::tensor_t tensor;
    tensor.dtype = safetensors::dtype::kFLOAT16;
    tensor.data_offsets[0] = dst_offset;
    tensor.data_offsets[1] = dst_offset + sz;
    tensor.shape.resize(2);
    tensor.shape[0] = 16;
    tensor.shape[1] = 16;

    st.tensors.emplace("weight1", tensor);
  }

  // __metadata__
  {
    st.metadata.emplace("creater", "safetensors-cpp");
  }

  std::string filename = "example.safetensors";
  std::string warn;
  std::string err;
  bool ret = safetensors::save_to_file(st, filename, &warn, &err);

  if (warn.size()) {
    std::cout << "WARN: " << warn << "\n";
  }
  
  if (!ret) {
    std::cerr << "Failed to write safetensor data to " << filename << "\n";
    if (err.size()) {
      std::cout << "ERR: " << err << "\n";
    }
    return EXIT_FAILURE;
  }
  
  return EXIT_SUCCESS;
}
