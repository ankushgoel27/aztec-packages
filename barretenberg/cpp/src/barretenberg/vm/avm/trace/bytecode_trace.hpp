#pragma once

#include <utility>

#include "barretenberg/vm/avm/trace/common.hpp"
#include "barretenberg/vm/avm/trace/execution_hints.hpp"
namespace bb::avm_trace {

class AvmBytecodeTraceBuilder {
  public:
    struct BytecodeTraceEntry {
        // Calculate the bytecode hash
        FF packed_bytecode{};
        FF running_hash{};
        // This is the length in fields, not bytes - max 1000 fields
        uint16_t bytecode_length_remaining = 0;

        // Derive the class Id
        FF class_id{};

        // Derive the contract address
        FF contract_address{};
    };
    // This interface will change when we start feeding in more inputs and hints
    AvmBytecodeTraceBuilder(std::vector<std::vector<FF>> all_contracts_bytecode)
        : all_contracts_bytecode(std::move(all_contracts_bytecode))
    {}
    size_t size() const { return bytecode_trace.size(); }
    void reset();
    void finalize(std::vector<AvmFullRow<FF>>& main_trace);

    static FF compute_bytecode_hash(const std::vector<FF>& packed_bytecode);
    void build_bytecode_columns();

  private:
    std::vector<BytecodeTraceEntry> bytecode_trace;
    // This will contain the bytecode as field elements
    std::vector<std::vector<FF>> all_contracts_bytecode;
    // TODO: Come back to this
    // VmPublicInputs public_inputs;
    // ExecutionHints hints;
};
} // namespace bb::avm_trace
