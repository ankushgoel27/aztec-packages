use dep::aztec::prelude::{NoteHeader, NoteInterface, PrivateContext};
use dep::aztec::{
    note::utils::compute_note_hash_for_nullify, keys::getters::get_nsk_app,
    protocol_types::{constants::GENERATOR_INDEX__NOTE_NULLIFIER, hash::poseidon2_hash_with_separator}
};

global PUBLIC_KEY_NOTE_LEN: Field = 3;
// PUBLIC_KEY_NOTE_LEN * 32 + 32(storage_slot as bytes) + 32(note_type_id as bytes)
global PUBLIC_KEY_NOTE_BYTES_LEN: Field = 3 * 32 + 64;

// Stores a public key composed of two fields
// TODO: Do we need to include a nonce, in case we want to read/nullify/recreate with the same pubkey value?
#[aztec(note)]
struct PublicKeyNote {
    x: Field,
    y: Field,
    // We store the npk_m_hash only to get the secret key to compute the nullifier
    npk_m_hash: Field,
}

impl NoteInterface<PUBLIC_KEY_NOTE_LEN, PUBLIC_KEY_NOTE_BYTES_LEN> for PublicKeyNote {
    fn compute_nullifier(self, context: &mut PrivateContext, note_hash_for_nullify: Field) -> Field {
        let secret = context.request_nsk_app(self.npk_m_hash);
        poseidon2_hash_with_separator(
            [
            note_hash_for_nullify,
            secret
        ],
            GENERATOR_INDEX__NOTE_NULLIFIER as Field
        )
    }

    fn compute_nullifier_without_context(self) -> Field {
        let note_hash_for_nullify = compute_note_hash_for_nullify(self);
        let secret = get_nsk_app(self.npk_m_hash);
        poseidon2_hash_with_separator(
            [
            note_hash_for_nullify,
            secret
        ],
            GENERATOR_INDEX__NOTE_NULLIFIER as Field
        )
    }
}

impl PublicKeyNote {
    pub fn new(x: Field, y: Field, npk_m_hash: Field) -> Self {
        PublicKeyNote { x, y, npk_m_hash, header: NoteHeader::empty() }
    }
}
