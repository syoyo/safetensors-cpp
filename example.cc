#include <iostream>

#if !defined(SAFETENSORS_CPP_NO_IMPLEMENTATION)
#define SAFETENSORS_CPP_IMPLEMENTATION
#endif
#include "safetensors.hh"

// TODO: provide printer for each dtype for efficiency.
std::string to_string(safetensors::dtype dtype, const uint8_t *data) {
  switch (dtype) {
    case safetensors::dtype::kBOOL: {
      return std::to_string(data[0] ? 1 : 0);
    }
    case safetensors::dtype::kUINT8: {
      return std::to_string(data[0]);
    }
    case safetensors::dtype::kINT8: {
      return std::to_string(*reinterpret_cast<const int8_t *>(data));
    }
    case safetensors::dtype::kUINT16: {
      return std::to_string(*reinterpret_cast<const uint16_t *>(data));
    }
    case safetensors::dtype::kINT16: {
      return std::to_string(*reinterpret_cast<const int16_t *>(data));
    }
    case safetensors::dtype::kUINT32: {
      return std::to_string(*reinterpret_cast<const uint32_t *>(data));
    }
    case safetensors::dtype::kINT32: {
      return std::to_string(*reinterpret_cast<const int32_t *>(data));
    }
    case safetensors::dtype::kUINT64: {
      return std::to_string(*reinterpret_cast<const uint64_t *>(data));
    }
    case safetensors::dtype::kINT64: {
      return std::to_string(*reinterpret_cast<const int64_t *>(data));
    }
    case safetensors::dtype::kFLOAT16: {
      return std::to_string(safetensors::fp16_to_float(
          *reinterpret_cast<const uint16_t *>(data)));
    }
    case safetensors::dtype::kBFLOAT16: {
      return std::to_string(safetensors::bfloat16_to_float(
          *reinterpret_cast<const int64_t *>(data)));
    }
    case safetensors::dtype::kFLOAT32: {
      return std::to_string(*reinterpret_cast<const float *>(data));
    }
    case safetensors::dtype::kFLOAT64: {
      return std::to_string(*reinterpret_cast<const double *>(data));
    }
  }

  return std::string("???");
}

//
// print tensor in linearized 1D array
// In safetensors, data is not strided(tightly packed)
//
std::string to_string_snipped(const safetensors::tensor_t &t,
                              const uint8_t *databuffer, size_t N = 8) {
  std::stringstream ss;
  size_t nitems = safetensors::get_shape_size(t);
  size_t itembytes = safetensors::get_dtype_bytes(t.dtype);

  if ((N == 0) || ((N * 2) >= nitems)) {
    ss << "[";
    for (size_t i = 0; i < nitems; i++) {
      if (i > 0) {
        ss << ", ";
      }
      ss << to_string(t.dtype, databuffer + t.data_offsets[0] + i * itembytes);
    }
    ss << "]";
  } else {
    size_t head_end = (std::min)(N, nitems);
    size_t tail_start = (std::max)(nitems - N, head_end);

    for (size_t i = 0; i < head_end; i++) {
      if (i > 0) {
        ss << ", ";
      }
      ss << to_string(t.dtype, databuffer + t.data_offsets[0] + i * itembytes);
    }

    ss << ", ..., ";

    for (size_t i = tail_start; i < nitems; i++) {
      if (i > tail_start) {
        ss << ", ";
      }
      ss << to_string(t.dtype, databuffer + t.data_offsets[0] + i * itembytes);
    }

    ss << "]";
  }

  return ss.str();
}

int main(int argc, char **argv) {
  std::string filename = "gen/model.safetensors";

  safetensors::safetensors_t st;

  if (argc > 1) {
    filename = argv[1];
  }

  std::string warn, err;
  bool ret = safetensors::load_from_file(filename, &st, &warn, &err);

  if (warn.size()) {
    std::cout << "WARN: " << warn << "\n";
  }

  if (!ret) {
    std::cerr << "Failed to load: " << filename << "\n";
    std::cerr << "  ERR: " << err << "\n";

    return EXIT_FAILURE;
  }

  // Check if data_offsets are valid.
  if (!safetensors::validate_data_offsets(st, err)) {
    std::cerr << "Invalid data_offsets\n";
    std::cerr << err << "\n";

    return EXIT_FAILURE;
  }

  const uint8_t *databuffer{nullptr};
  if (st.mmaped) {
    databuffer = st.mmap_addr;
  } else {
    databuffer = st.storage.data();
  }

  // Print Tensor info & value.
  for (const auto &item : st.tensors) {
    std::cout << item.first << ": "
              << safetensors::get_dtype_str(item.second.dtype) << " ";
    std::cout << "[";
    for (size_t i = 0; i < item.second.shape.size(); i++) {
      if (i > 0) {
        std::cout << ", ";
      }
      std::cout << std::to_string(item.second.shape[i]);
    }
    std::cout << "]\n";

    std::cout << "  data_offsets["
              << std::to_string(item.second.data_offsets[0]) << ", "
              << std::to_string(item.second.data_offsets[1]) << "\n";
    std::cout << "  " << to_string_snipped(item.second, databuffer) << "\n";
  }

  // Print metadata
  if (st.metadata.size()) {
    std::cout << "\n";
    std::cout << "__metadata__\n";
    for (const auto &item : st.metadata) {
      std::cout << "  " << item.first << ":" << item.second << "\n";
    }
  }

  return EXIT_SUCCESS;
}
