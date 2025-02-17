#pragma once
#include "barretenberg/flavor/flavor.hpp"
#include "barretenberg/relations/relation_parameters.hpp"
#include "barretenberg/stdlib/protogalaxy_verifier/recursive_decider_verification_key.hpp"

namespace bb::stdlib::recursion::honk {
template <IsRecursiveFlavor Flavor_, size_t NUM_> struct RecursiveDeciderVerificationKeys_ {
    using Flavor = Flavor_;
    using Builder = typename Flavor::CircuitBuilder;
    using VerificationKey = typename Flavor::VerificationKey;
    using DeciderVK = RecursiveDeciderVerificationKey_<Flavor>;
    using ArrayType = std::array<std::shared_ptr<DeciderVK>, NUM_>;

  public:
    static constexpr size_t NUM = NUM_;
    static constexpr size_t BATCHED_EXTENDED_LENGTH = (Flavor::MAX_TOTAL_RELATION_LENGTH - 1 + NUM - 1) * (NUM - 1) + 1;
    ArrayType _data;
    std::shared_ptr<DeciderVK> const& operator[](size_t idx) const { return _data[idx]; }
    typename ArrayType::iterator begin() { return _data.begin(); };
    typename ArrayType::iterator end() { return _data.end(); };
    Builder* builder;

    RecursiveDeciderVerificationKeys_(Builder* builder,
                                      const std::shared_ptr<DeciderVK>& accumulator,
                                      const std::vector<std::shared_ptr<VerificationKey>>& vks)
        : builder(builder)
    {
        ASSERT(vks.size() == NUM - 1);

        _data[0] = accumulator;

        size_t idx = 1;
        for (auto& vk : vks) {
            _data[idx] = std::make_shared<DeciderVK>(builder, vk);
            idx++;
        }
    }
};
} // namespace bb::stdlib::recursion::honk
