#pragma once

#include "barretenberg/goblin/goblin.hpp"
#include "barretenberg/goblin/mock_circuits.hpp"
#include "barretenberg/plonk_honk_shared/arithmetization/max_block_size_tracker.hpp"
#include "barretenberg/protogalaxy/protogalaxy_prover.hpp"
#include "barretenberg/protogalaxy/protogalaxy_verifier.hpp"
#include "barretenberg/ultra_honk/decider_keys.hpp"
#include "barretenberg/ultra_honk/decider_prover.hpp"
#include "barretenberg/ultra_honk/decider_verifier.hpp"
#include <algorithm>

namespace bb {

/**
 * @brief The IVC interface to be used by the aztec client for private function execution
 * @details Combines Protogalaxy with Goblin to accumulate one circuit at a time with efficient EC group operations
 */
class ClientIVC {

  public:
    using Flavor = MegaFlavor;
    using VerificationKey = Flavor::VerificationKey;
    using FF = Flavor::FF;
    using FoldProof = std::vector<FF>;
    using DeciderProvingKey = DeciderProvingKey_<Flavor>;
    using DeciderVerificationKey = DeciderVerificationKey_<Flavor>;
    using ClientCircuit = MegaCircuitBuilder; // can only be Mega
    using DeciderProver = DeciderProver_<Flavor>;
    using DeciderVerifier = DeciderVerifier_<Flavor>;
    using DeciderProvingKeys = DeciderProvingKeys_<Flavor>;
    using FoldingProver = ProtogalaxyProver_<DeciderProvingKeys>;
    using DeciderVerificationKeys = DeciderVerificationKeys_<Flavor>;
    using FoldingVerifier = ProtogalaxyVerifier_<DeciderVerificationKeys>;
    using ECCVMVerificationKey = bb::ECCVMFlavor::VerificationKey;
    using TranslatorVerificationKey = bb::TranslatorFlavor::VerificationKey;

    using GURecursiveFlavor = MegaRecursiveFlavor_<bb::MegaCircuitBuilder>;
    using RecursiveDeciderVerificationKeys =
        bb::stdlib::recursion::honk::RecursiveDeciderVerificationKeys_<GURecursiveFlavor, 2>;
    using RecursiveDeciderVerificationKey = RecursiveDeciderVerificationKeys::DeciderVK;
    using RecursiveVerificationKey = RecursiveDeciderVerificationKeys::VerificationKey;
    using FoldingRecursiveVerifier =
        bb::stdlib::recursion::honk::ProtogalaxyRecursiveVerifier_<RecursiveDeciderVerificationKeys>;

    // A full proof for the IVC scheme
    struct Proof {
        FoldProof folding_proof; // final fold proof
        HonkProof decider_proof;
        GoblinProof goblin_proof;

        size_t size() const { return folding_proof.size() + decider_proof.size() + goblin_proof.size(); }

        MSGPACK_FIELDS(folding_proof, decider_proof, goblin_proof);
    };

    // Utility for tracking the max size of each block across the full IVC
    MaxBlockSizeTracker max_block_size_tracker;

  private:
    using ProverFoldOutput = FoldingResult<Flavor>;
    // Note: We need to save the last proving key that was folded in order to compute its verification key, this will
    // not be needed in the real IVC as they are provided as inputs

  public:
    GoblinProver goblin;
    ProverFoldOutput fold_output;
    std::shared_ptr<DeciderVerificationKey> verifier_accumulator;
    std::shared_ptr<VerificationKey> decider_vk;

    // A flag indicating whether or not to construct a structured trace in the DeciderProvingKey
    TraceStructure trace_structure = TraceStructure::NONE;

    // A flag indicating whether the IVC has been initialized with an initial decider proving key
    bool initialized = false;

    void accumulate(ClientCircuit& circuit, const std::shared_ptr<VerificationKey>& precomputed_vk = nullptr);

    Proof prove();

    static bool verify(const Proof& proof,
                       const std::shared_ptr<DeciderVerificationKey>& accumulator,
                       const std::shared_ptr<DeciderVerificationKey>& final_stack_vk,
                       const std::shared_ptr<ClientIVC::ECCVMVerificationKey>& eccvm_vk,
                       const std::shared_ptr<ClientIVC::TranslatorVerificationKey>& translator_vk);

    bool verify(Proof& proof, const std::vector<std::shared_ptr<DeciderVerificationKey>>& vk_stack) const;

    bool prove_and_verify();

    HonkProof decider_prove() const;

    std::vector<std::shared_ptr<VerificationKey>> precompute_folding_verification_keys(std::vector<ClientCircuit>);
};
} // namespace bb