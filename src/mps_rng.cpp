#include <ATen/mps/MPSGeneratorImpl.h>
#include <c10/util/Exception.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <tuple>


static int64_t signed_bit_pattern(uint64_t value) {
    static_assert(sizeof(int64_t) == sizeof(uint64_t));
    int64_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}


static std::tuple<int64_t, int64_t> reserve_default_mps_philox(
        uint64_t rounds) {
    TORCH_CHECK(rounds > 0, "Philox reservation must be positive");

    const at::Generator& generator =
        at::mps::detail::getDefaultMPSGenerator();
    auto* implementation = generator.get<at::MPSGeneratorImpl>();

    // Python exposes get_offset and set_offset separately. Only this combined
    // critical section prevents another PyTorch RNG operation from reserving
    // the same counter range between those calls.
    std::lock_guard<std::mutex> lock(implementation->mutex_);
    const uint64_t seed = implementation->current_seed();
    const uint64_t offset = implementation->get_offset();
    TORCH_CHECK(
        rounds <= std::numeric_limits<uint64_t>::max() - offset,
        "MPS Philox offset overflow");
    implementation->set_offset(offset + rounds);

    // compile_shader binds Python integers as signed int64. Preserve the bit
    // pattern exactly, matching PyTorch's own MPS distribution dispatcher.
    return {
        signed_bit_pattern(seed),
        signed_bit_pattern(offset),
    };
}


PYBIND11_MODULE(_mps_rng, module) {
    module.def(
        "reserve_default_mps_philox",
        &reserve_default_mps_philox,
        "Atomically reserve rounds from PyTorch's default MPS generator",
        pybind11::arg("rounds"));
}
