use dep::protocol_types::{
    abis::{function_selector::FunctionSelector, private_circuit_public_inputs::PrivateCircuitPublicInputs},
    address::AztecAddress, traits::Deserialize
};

use crate::context::inputs::{PublicContextInputs, PrivateContextInputs};
use crate::context::{packed_returns::PackedReturns, call_interfaces::CallInterface};

use crate::context::{PrivateContext, PublicContext, UnconstrainedContext};
use crate::test::helpers::{cheatcodes, utils::{apply_side_effects_private, Deployer}, keys};
use crate::keys::constants::{NULLIFIER_INDEX, INCOMING_INDEX, OUTGOING_INDEX, TAGGING_INDEX};
use crate::hash::hash_args;

use crate::note::{note_header::NoteHeader, note_interface::NoteInterface};
use crate::oracle::{execution::{get_block_number, get_contract_address}, notes::notify_created_note};

struct TestEnvironment {}

impl TestEnvironment {
    fn new() -> Self {
        cheatcodes::reset();
        Self {}
    }

    fn block_number(_self: Self) -> u32 {
        get_block_number()
    }

    fn contract_address(_self: Self) -> AztecAddress {
        get_contract_address()
    }

    fn impersonate(_self: Self, address: AztecAddress) {
        cheatcodes::set_contract_address(address)
    }

    fn advance_block_to(&mut self, block_number: u32) {
        let difference = block_number - get_block_number();
        self.advance_block_by(difference);
    }

    fn advance_block_by(_self: &mut Self, blocks: u32) {
        cheatcodes::advance_blocks_by(blocks);
    }

    fn public(_self: Self) -> PublicContext {
        let mut inputs = cheatcodes::get_public_context_inputs();
        PublicContext::new(inputs)
    }

    fn private(&mut self) -> PrivateContext {
        self.private_at(get_block_number())
    }

    // unconstrained is a key word, so we mis-spell purposefully here, like we do with contrakt
    fn unkonstrained(_self: Self) -> UnconstrainedContext {
        UnconstrainedContext::new()
    }

    fn private_at(&mut self, historical_block_number: u32) -> PrivateContext {
        if historical_block_number >= get_block_number() {
            self.advance_block_to(historical_block_number + 1);
        }

        let mut inputs = cheatcodes::get_private_context_inputs(historical_block_number);

        PrivateContext::new(inputs, 0)
    }

    fn create_account(_self: Self) -> AztecAddress {
        let test_account = cheatcodes::create_account();
        let address = test_account.address;
        let keys = test_account.keys;

        keys::store_master_key(NULLIFIER_INDEX, address, keys.npk_m);
        keys::store_master_key(INCOMING_INDEX, address, keys.ivpk_m);
        keys::store_master_key(OUTGOING_INDEX, address, keys.ovpk_m);
        keys::store_master_key(TAGGING_INDEX, address, keys.tpk_m);

        test_account.address
    }

    fn create_account_contract(&mut self, secret: Field) -> AztecAddress {
        let test_account = cheatcodes::add_account(secret);
        let address = test_account.address;
        cheatcodes::advance_blocks_by(1);

        let keys = test_account.keys;

        keys::store_master_key(NULLIFIER_INDEX, address, keys.npk_m);
        keys::store_master_key(INCOMING_INDEX, address, keys.ivpk_m);
        keys::store_master_key(OUTGOING_INDEX, address, keys.ovpk_m);
        keys::store_master_key(TAGGING_INDEX, address, keys.tpk_m);

        let selector = FunctionSelector::from_signature("constructor(Field,Field)");

        let mut context = self.private_at(get_block_number());
        let args = [test_account.keys.ivpk_m.inner.x, test_account.keys.ivpk_m.inner.y];
        let _ = context.call_private_function(address, selector, args);

        address
    }

    fn deploy<N, M>(self, path: str<N>, name: str<M>) -> Deployer<N, M> {
        Deployer { path, name, public_keys_hash: 0 }
    }

    fn deploy_self<M>(self, name: str<M>) -> Deployer<0, M> {
        Deployer { path: "", name, public_keys_hash: 0 }
    }

    fn call_private<C, let M: u32, T, Env, let N: u32>(
        _self: Self,
        call_interface: C
    ) -> T where C: CallInterface<M, PrivateContextInputs, PrivateCircuitPublicInputs, Env>, T: Deserialize<N> {
        let original_fn = call_interface.get_original();
        let original_msg_sender = cheatcodes::get_msg_sender();
        let original_contract_address = get_contract_address();
        let target_address = call_interface.get_contract_address();

        cheatcodes::set_contract_address(target_address);
        cheatcodes::set_msg_sender(original_contract_address);
        let mut inputs = cheatcodes::get_private_context_inputs(get_block_number() - 1);
        inputs.call_context.function_selector = call_interface.get_selector();
        inputs.call_context.is_static_call = call_interface.get_is_static();
        let public_inputs = original_fn(inputs);
        apply_side_effects_private(target_address, public_inputs);

        cheatcodes::set_contract_address(original_contract_address);
        cheatcodes::set_msg_sender(original_msg_sender);
        PackedReturns::new(public_inputs.returns_hash).unpack_into()
    }

    fn call_private_void<C, let M: u32, Env>(
        _self: Self,
        call_interface: C
    ) where C: CallInterface<M, PrivateContextInputs, PrivateCircuitPublicInputs, Env> {
        let original_fn = call_interface.get_original();
        let original_msg_sender = cheatcodes::get_msg_sender();
        let original_contract_address = get_contract_address();
        let target_address = call_interface.get_contract_address();

        cheatcodes::set_contract_address(target_address);
        cheatcodes::set_msg_sender(original_contract_address);
        let mut inputs = cheatcodes::get_private_context_inputs(get_block_number() - 1);
        inputs.call_context.function_selector = call_interface.get_selector();
        inputs.call_context.is_static_call = call_interface.get_is_static();
        let public_inputs = original_fn(inputs);
        apply_side_effects_private(target_address, public_inputs);

        cheatcodes::set_contract_address(original_contract_address);
        cheatcodes::set_msg_sender(original_msg_sender);
        PackedReturns::new(public_inputs.returns_hash).assert_empty();
    }

    fn call_public<C, let M: u32, T, Env>(
        _self: Self,
        call_interface: C
    ) -> T where C: CallInterface<M, PublicContextInputs, T, Env> {
        let original_fn = call_interface.get_original();
        let original_msg_sender = cheatcodes::get_msg_sender();
        let original_contract_address = get_contract_address();
        let original_fn_selector = cheatcodes::get_function_selector();
        let target_address = call_interface.get_contract_address();
        let fn_selector = call_interface.get_selector();
        let calldata = call_interface.get_args();

        cheatcodes::set_fn_selector(fn_selector);
        cheatcodes::set_contract_address(target_address);
        cheatcodes::set_msg_sender(original_contract_address);
        let mut inputs = cheatcodes::get_public_context_inputs();
        inputs.calldata_length = call_interface.get_args().len() as Field;
        inputs.is_static_call = call_interface.get_is_static();
        cheatcodes::set_calldata(calldata);

        let result = original_fn(inputs);

        cheatcodes::set_fn_selector(original_fn_selector);
        cheatcodes::set_contract_address(original_contract_address);
        cheatcodes::set_msg_sender(original_msg_sender);
        cheatcodes::set_calldata(calldata);
        result
    }

    fn assert_public_call_fails<C, let M: u32, T, Env>(
        _self: Self,
        call_interface: C
    ) where C: CallInterface<M, PublicContextInputs, T, Env> {
        cheatcodes::assert_public_call_fails(
            call_interface.get_contract_address(),
            call_interface.get_selector(),
            call_interface.get_args()
        );
    }

    fn assert_private_call_fails<C, let M: u32, T, Env>(
        _self: Self,
        call_interface: C
    ) where C: CallInterface<M, PrivateContextInputs, T, Env> {
        cheatcodes::assert_private_call_fails(
            call_interface.get_contract_address(),
            call_interface.get_selector(),
            hash_args(call_interface.get_args()),
            cheatcodes::get_side_effects_counter() as Field,
            call_interface.get_is_static(),
            false
        );
    }

    /// Manually adds a note to TXE. This needs to be called if you want to work with a note in your test with the note
    /// not having an encrypted log emitted. TXE alternative to `PXE.addNote(...)`.
    pub fn add_note<Note, let N: u32, let M: u32>(
        _self: Self,
        note: &mut Note,
        storage_slot: Field,
        contract_address: AztecAddress
    ) where Note: NoteInterface<N, M> {
        let original_contract_address = get_contract_address();
        cheatcodes::set_contract_address(contract_address);
        let note_hash_counter = cheatcodes::get_side_effects_counter();

        let header = NoteHeader { contract_address, storage_slot, nonce: 0, note_hash_counter };
        note.set_header(header);
        // TODO(#7771): inject compute_note_hash(...) func to notes with macros.
        let note_hash = note.compute_note_hiding_point().x;
        let serialized_note = Note::serialize_content(*note);
        assert(
            notify_created_note(
            storage_slot,
            Note::get_note_type_id(),
            serialized_note,
            note_hash,
            note_hash_counter
        )
            == 0
        );
        cheatcodes::set_contract_address(original_contract_address);
    }
}
