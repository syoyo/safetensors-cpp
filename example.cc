#include <iostream>

#define SAFETENSORS_CPP_IMPLEMENTATION
#include "safetensors.hh"

int main(int argc, char **argv)
{
  std::string filename = "test/safetensors_abuse_attempt_1.safetensors";

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

  return EXIT_SUCCESS;

}
