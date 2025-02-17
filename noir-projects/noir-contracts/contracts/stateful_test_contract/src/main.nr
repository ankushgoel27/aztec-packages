// A contract used for testing a random hodgepodge of small features from simulator and end-to-end tests.
contract StatefulTest {
    use dep::aztec::prelude::{Map, PublicMutable, PrivateSet, AztecAddress, FunctionSelector};
    use dep::value_note::{balance_utils, utils::{increment, decrement}, value_note::ValueNote};
    use dep::aztec::{initializer::assert_is_initialized_private};

    #[aztec(storage)]
    struct Storage {
        notes: Map<AztecAddress, PrivateSet<ValueNote>>,
        public_values: Map<AztecAddress, PublicMutable<Field>>,
    }

    #[aztec(private)]
    #[aztec(initializer)]
    fn constructor(owner: AztecAddress, outgoing_viewer: AztecAddress, value: Field) {
        StatefulTest::at(context.this_address()).create_note_no_init_check(owner, outgoing_viewer, value).call(&mut context);
    }

    #[aztec(private)]
    #[aztec(initializer)]
    fn wrong_constructor() {
        let selector = FunctionSelector::from_signature("not_exists(Field)");
        let _res = context.call_public_function(context.this_address(), selector, [42]);
    }

    // Having _ignored_arg here as it makes the params the same as for the private constructor which makes
    // contract_class_registration tests way less cluttered. This is a test contract. Don't judge me.
    #[aztec(public)]
    #[aztec(initializer)]
    fn public_constructor(owner: AztecAddress, _ignored_arg: AztecAddress, value: Field) {
        StatefulTest::at(context.this_address()).increment_public_value_no_init_check(owner, value).call(&mut context);
    }

    #[aztec(private)]
    fn create_note(owner: AztecAddress, outgoing_viewer: AztecAddress, value: Field) {
        if (value != 0) {
            let loc = storage.notes.at(owner);
            increment(loc, value, owner, outgoing_viewer);
        }
    }

    #[aztec(private)]
    #[aztec(noinitcheck)]
    fn create_note_no_init_check(owner: AztecAddress, outgoing_viewer: AztecAddress, value: Field) {
        if (value != 0) {
            let loc = storage.notes.at(owner);
            increment(loc, value, owner, outgoing_viewer);
        }
    }

    #[aztec(private)]
    fn destroy_and_create(recipient: AztecAddress, amount: Field) {
        assert_is_initialized_private(&mut context);
        let sender = context.msg_sender();

        let sender_notes = storage.notes.at(sender);
        decrement(sender_notes, amount, sender, context.msg_sender());

        let recipient_notes = storage.notes.at(recipient);
        increment(recipient_notes, amount, recipient, context.msg_sender());
    }

    #[aztec(private)]
    #[aztec(noinitcheck)]
    fn destroy_and_create_no_init_check(recipient: AztecAddress, amount: Field) {
        let sender = context.msg_sender();

        let sender_notes = storage.notes.at(sender);
        decrement(sender_notes, amount, sender, context.msg_sender());

        let recipient_notes = storage.notes.at(recipient);
        increment(recipient_notes, amount, recipient, context.msg_sender());
    }

    #[aztec(public)]
    fn increment_public_value(owner: AztecAddress, value: Field) {
        let loc = storage.public_values.at(owner);
        loc.write(loc.read() + value);
    }

    #[aztec(public)]
    #[aztec(noinitcheck)]
    fn increment_public_value_no_init_check(owner: AztecAddress, value: Field) {
        let loc = storage.public_values.at(owner);
        loc.write(loc.read() + value);
    }

    unconstrained fn summed_values(owner: AztecAddress) -> pub Field {
        let owner_balance = storage.notes.at(owner);

        // docs:start:get_balance
        // Return the sum of all notes in the set.
        balance_utils::get_balance(owner_balance)
        // docs:end:get_balance
    }

    #[aztec(public)]
    #[aztec(noinitcheck)]
    #[aztec(view)]
    fn get_public_value(owner: AztecAddress) -> pub Field {
        storage.public_values.at(owner).read()
    }
}
