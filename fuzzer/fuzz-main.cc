#include "safetensors.hh"

static void parse_safetensors(const uint8_t *data, size_t size)
{
  return;
}

extern "C"
int LLVMFuzzerTestOneInput(std::uint8_t const* data, std::size_t size)
{
    parse_safetensors(data, size);
    return 0;
}
