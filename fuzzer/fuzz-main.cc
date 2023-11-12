#define SAFETENSORS_CPP_IMPLEMENTATION
#include "safetensors.hh"

static void parse_safetensors(const uint8_t *data, size_t size)
{
  safetensors::safetensors_t st;
  std::string warn, err;

  bool ret = safetensors::load_from_memory(data, size, /* filename */"", &st, &warn, &err);
  (void)ret;

  return;
}

extern "C"
int LLVMFuzzerTestOneInput(std::uint8_t const* data, std::size_t size)
{
    parse_safetensors(data, size);
    return 0;
}
