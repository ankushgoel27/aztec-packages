#include "barretenberg/aztec_ivc/aztec_ivc.hpp"
#include "barretenberg/ultra_honk/oink_prover.hpp"

namespace bb {

/**
 * @brief Instantiate a stdlib verification queue for use in the kernel completion logic
 * @details Construct a stdlib proof/verification_key for each entry in the native verification queue. By default, both
 * are constructed from their counterpart in the native queue. Alternatively, Stdlib verification keys can be provided
 * directly as input to this method. (The later option is used, for example, when constructing recursive verifiers based
 * on the verification key witnesses from an acir recursion constraint. This option is not provided for proofs since
 * valid proof witnesses are in general not known at the time of acir constraint generation).
 *
 * @param circuit
 */
void AztecIVC::instantiate_stdlib_verification_queue(
    ClientCircuit& circuit, const std::vector<std::shared_ptr<RecursiveVerificationKey>>& input_keys)
{
    bool vkeys_provided = !input_keys.empty();
    if (vkeys_provided && verification_queue.size() != input_keys.size()) {
        info("Warning: Incorrect number of verification keys provided in stdlib verification queue instantiation.");
        ASSERT(false);
    }

    size_t key_idx = 0;
    for (auto& [proof, vkey, type] : verification_queue) {
        // Construct stdlib proof directly from the internal native queue data
        auto stdlib_proof = bb::convert_proof_to_witness(&circuit, proof);

        // Use the provided stdlib vkey if present, otherwise construct one from the internal native queue
        auto stdlib_vkey =
            vkeys_provided ? input_keys[key_idx++] : std::make_shared<RecursiveVerificationKey>(&circuit, vkey);

        stdlib_verification_queue.push_back({ stdlib_proof, stdlib_vkey, type });
    }
    verification_queue.clear(); // the native data is not needed beyond this point
}

/**
 * @brief Populate the provided circuit with constraints for (1) recursive verification of the provided accumulation
 * proof and (2) the associated databus commitment consistency checks.
 * @details The recursive verifier will be either Oink or Protogalaxy depending on the specified proof type. In either
 * case, the verifier accumulator is updated in place via the verification algorithm. Databus commitment consistency
 * checks are performed on the witness commitments and public inputs extracted from the proof by the verifier.
 *
 * @param circuit The circuit to which the constraints are appended
 * @param proof A stdlib proof to be recursively verified (either oink or PG)
 * @param vkey The stdlib verfication key associated with the proof
 * @param type The type of the proof (equivalently, the type of the verifier)
 */
void AztecIVC::perform_recursive_verification_and_databus_consistency_checks(
    ClientCircuit& circuit,
    const StdlibProof<ClientCircuit>& proof,
    const std::shared_ptr<RecursiveVerificationKey>& vkey,
    const QUEUE_TYPE type)
{
    switch (type) {
    case QUEUE_TYPE::PG: {
        // Construct stdlib verifier accumulator from the native counterpart computed on a previous round
        auto stdlib_verifier_accum = std::make_shared<RecursiveDeciderVerificationKey>(&circuit, verifier_accumulator);

        // Perform folding recursive verification to update the verifier accumulator
        FoldingRecursiveVerifier verifier{ &circuit, stdlib_verifier_accum, { vkey } };
        auto verifier_accum = verifier.verify_folding_proof(proof);

        // Extract native verifier accumulator from the stdlib accum for use on the next round
        verifier_accumulator = std::make_shared<DeciderVerificationKey>(verifier_accum->get_value());

        // Perform databus commitment consistency checks and propagate return data commitments via public inputs
        bus_depot.execute(verifier.keys_to_fold[1]->witness_commitments,
                          verifier.keys_to_fold[1]->public_inputs,
                          verifier.keys_to_fold[1]->verification_key->databus_propagation_data);
        break;
    }
    case QUEUE_TYPE::OINK: {
        // Construct an incomplete stdlib verifier accumulator from the corresponding stdlib verification key
        auto verifier_accum = std::make_shared<RecursiveDeciderVerificationKey>(&circuit, vkey);

        // Perform oink recursive verification to complete the initial verifier accumulator
        OinkRecursiveVerifier oink{ &circuit, verifier_accum };
        oink.verify_proof(proof);
        verifier_accum->is_accumulator = true; // indicate to PG that it should not run oink

        // Extract native verifier accumulator from the stdlib accum for use on the next round
        verifier_accumulator = std::make_shared<DeciderVerificationKey>(verifier_accum->get_value());
        // Initialize the gate challenges to zero for use in first round of folding
        auto log_circuit_size = static_cast<size_t>(verifier_accum->verification_key->log_circuit_size);
        verifier_accumulator->gate_challenges = std::vector<FF>(log_circuit_size, 0);

        // Perform databus commitment consistency checks and propagate return data commitments via public inputs
        bus_depot.execute(verifier_accum->witness_commitments,
                          verifier_accum->public_inputs,
                          verifier_accum->verification_key->databus_propagation_data);

        break;
    }
    }
}

/**
 * @brief Perform recursive merge verification for each merge proof in the queue
 *
 * @param circuit
 */
void AztecIVC::process_recursive_merge_verification_queue(ClientCircuit& circuit)
{
    // Recusively verify all merge proofs in queue
    for (auto& proof : merge_verification_queue) {
        goblin.verify_merge(circuit, proof);
    }
    merge_verification_queue.clear();
}

/**
 * @brief Append logic to complete a kernel circuit
 * @details A kernel circuit may contain some combination of PG recursive verification, merge recursive
 * verification, and databus commitment consistency checks. This method appends this logic to a provided kernel
 * circuit.
 *
 * @param circuit
 */
void AztecIVC::complete_kernel_circuit_logic(ClientCircuit& circuit)
{
    circuit.databus_propagation_data.is_kernel = true;

    // Instantiate stdlib verifier inputs from their native counterparts
    if (stdlib_verification_queue.empty()) {
        instantiate_stdlib_verification_queue(circuit);
    }

    // Peform recursive verification and databus consistency checks for each entry in the verification queue
    for (auto& [proof, vkey, type] : stdlib_verification_queue) {
        perform_recursive_verification_and_databus_consistency_checks(circuit, proof, vkey, type);
    }
    stdlib_verification_queue.clear();

    // Perform recursive merge verification for every merge proof in the queue
    process_recursive_merge_verification_queue(circuit);
}

/**
 * @brief Execute prover work for accumulation
 * @details Construct an proving key for the provided circuit. If this is the first step in the IVC, simply initialize
 * the folding accumulator. Otherwise, execute the PG prover to fold the proving key into the accumulator and produce a
 * folding proof. Also execute the merge protocol to produce a merge proof.
 *
 * @param circuit
 * @param precomputed_vk
 */
void AztecIVC::accumulate(ClientCircuit& circuit, const std::shared_ptr<VerificationKey>& precomputed_vk)
{
    // Construct merge proof for the present circuit and add to merge verification queue
    MergeProof merge_proof = goblin.prove_merge(circuit);
    merge_verification_queue.emplace_back(merge_proof);

    // TODO(https://github.com/AztecProtocol/barretenberg/issues/1069): Do proper aggregation with merge recursive
    // verifier.
    circuit.add_recursive_proof(stdlib::recursion::init_default_agg_obj_indices<ClientCircuit>(circuit));

    // Construct the proving key for circuit
    std::shared_ptr<DeciderProvingKey> proving_key;
    if (!initialized) {
        proving_key = std::make_shared<DeciderProvingKey>(circuit, trace_structure);
    } else {
        proving_key = std::make_shared<DeciderProvingKey>(
            circuit, trace_structure, fold_output.accumulator->proving_key.commitment_key);
    }

    // Set the verification key from precomputed if available, else compute it
    honk_vk = precomputed_vk ? precomputed_vk : std::make_shared<VerificationKey>(proving_key->proving_key);

    // If this is the first circuit in the IVC, use oink to complete the decider proving key and generate an oink proof
    if (!initialized) {
        OinkProver<Flavor> oink_prover{ proving_key };
        oink_prover.prove();
        proving_key->is_accumulator = true; // indicate to PG that it should not run oink on this key
        // Initialize the gate challenges to zero for use in first round of folding
        proving_key->gate_challenges = std::vector<FF>(proving_key->proving_key.log_circuit_size, 0);

        fold_output.accumulator = proving_key; // initialize the prover accum with the completed key

        // Add oink proof and corresponding verification key to the verification queue
        verification_queue.push_back(
            bb::AztecIVC::VerifierInputs{ oink_prover.transcript->proof_data, honk_vk, QUEUE_TYPE::OINK });

        initialized = true;
    } else { // Otherwise, fold the new key into the accumulator
        FoldingProver folding_prover({ fold_output.accumulator, proving_key });
        fold_output = folding_prover.prove();

        // Add fold proof and corresponding verification key to the verification queue
        verification_queue.push_back(bb::AztecIVC::VerifierInputs{ fold_output.proof, honk_vk, QUEUE_TYPE::PG });
    }

    // Track the maximum size of each block for all circuits porcessed (for debugging purposes only)
    max_block_size_tracker.update(circuit);
}

/**
 * @brief Construct a proof for the IVC, which, if verified, fully establishes its correctness
 *
 * @return Proof
 */
AztecIVC::Proof AztecIVC::prove()
{
    max_block_size_tracker.print();               // print minimum structured sizes for each block
    ASSERT(verification_queue.size() == 1);       // ensure only a single fold proof remains in the queue
    ASSERT(merge_verification_queue.size() == 1); // ensure only a single merge proof remains in the queue
    FoldProof& fold_proof = verification_queue[0].proof;
    MergeProof& merge_proof = merge_verification_queue[0];
    return { fold_proof, decider_prove(), goblin.prove(merge_proof) };
};

bool AztecIVC::verify(const Proof& proof,
                      const std::shared_ptr<DeciderVerificationKey>& accumulator,
                      const std::shared_ptr<DeciderVerificationKey>& final_stack_vk,
                      const std::shared_ptr<AztecIVC::ECCVMVerificationKey>& eccvm_vk,
                      const std::shared_ptr<AztecIVC::TranslatorVerificationKey>& translator_vk)
{
    // Goblin verification (merge, eccvm, translator)
    GoblinVerifier goblin_verifier{ eccvm_vk, translator_vk };
    bool goblin_verified = goblin_verifier.verify(proof.goblin_proof);

    // Decider verification
    AztecIVC::FoldingVerifier folding_verifier({ accumulator, final_stack_vk });
    auto verifier_accumulator = folding_verifier.verify_folding_proof(proof.folding_proof);

    AztecIVC::DeciderVerifier decider_verifier(verifier_accumulator);
    bool decision = decider_verifier.verify_proof(proof.decider_proof);
    return goblin_verified && decision;
}

/**
 * @brief Verify a full proof of the IVC
 *
 * @param proof
 * @return bool
 */
bool AztecIVC::verify(const Proof& proof, const std::vector<std::shared_ptr<DeciderVerificationKey>>& vk_stack)
{
    auto eccvm_vk = std::make_shared<ECCVMVerificationKey>(goblin.get_eccvm_proving_key());
    auto translator_vk = std::make_shared<TranslatorVerificationKey>(goblin.get_translator_proving_key());
    return verify(proof, vk_stack[0], vk_stack[1], eccvm_vk, translator_vk);
}

/**
 * @brief Internal method for constructing a decider proof
 *
 * @return HonkProof
 */
HonkProof AztecIVC::decider_prove() const
{
    MegaDeciderProver decider_prover(fold_output.accumulator);
    return decider_prover.construct_proof();
}

/**
 * @brief Construct and verify a proof for the IVC
 * @note Use of this method only makes sense when the prover and verifier are the same entity, e.g. in
 * development/testing.
 *
 */
bool AztecIVC::prove_and_verify()
{
    auto proof = prove();

    ASSERT(verification_queue.size() == 1); // ensure only a single fold proof remains in the queue
    auto verifier_inst = std::make_shared<DeciderVerificationKey>(this->verification_queue[0].honk_verification_key);
    return verify(proof, { this->verifier_accumulator, verifier_inst });
}

} // namespace bb
