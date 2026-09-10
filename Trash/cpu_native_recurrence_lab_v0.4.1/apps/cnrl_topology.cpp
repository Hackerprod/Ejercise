#include <iostream>
#include <string_view>
#include "cnrl/platform.hpp"
int main(int argc, char** argv) {
  try {
    bool csv = false;
    if (argc > 1) {
      const std::string_view arg(argv[1]);
      if (arg == "--csv") csv = true;
      else if (arg != "--json") throw std::runtime_error("usage: cnrl_topology [--json|--csv]");
    }
    const auto topology = cnrl::discover_cpu_topology();
    std::cout << (csv ? cnrl::topology_as_csv(topology) : cnrl::topology_as_json(topology));
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 2;
  }
}
