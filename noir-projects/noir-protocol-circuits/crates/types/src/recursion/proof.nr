use crate::{
    traits::{Serialize, Deserialize, Empty},
    constants::{RECURSIVE_PROOF_LENGTH, NESTED_RECURSIVE_PROOF_LENGTH}
};

struct RecursiveProof {
    fields: [Field; RECURSIVE_PROOF_LENGTH],
}

impl Serialize<RECURSIVE_PROOF_LENGTH> for RecursiveProof {
    fn serialize(self) -> [Field; RECURSIVE_PROOF_LENGTH] {
        self.fields
    }
}

impl Deserialize<RECURSIVE_PROOF_LENGTH> for RecursiveProof {
    fn deserialize(fields: [Field; RECURSIVE_PROOF_LENGTH]) -> Self {
        RecursiveProof { fields }
    }
}

impl Empty for RecursiveProof {
    fn empty() -> Self {
        RecursiveProof { fields: [0; RECURSIVE_PROOF_LENGTH] }
    }
}

struct NestedRecursiveProof {
    fields: [Field; NESTED_RECURSIVE_PROOF_LENGTH],
}

impl Serialize<NESTED_RECURSIVE_PROOF_LENGTH> for NestedRecursiveProof {
    fn serialize(self) -> [Field; NESTED_RECURSIVE_PROOF_LENGTH] {
        self.fields
    }
}

impl Deserialize<NESTED_RECURSIVE_PROOF_LENGTH> for NestedRecursiveProof {
    fn deserialize(fields: [Field; NESTED_RECURSIVE_PROOF_LENGTH]) -> Self {
        NestedRecursiveProof { fields }
    }
}

impl Empty for NestedRecursiveProof {
    fn empty() -> Self {
        NestedRecursiveProof { fields: [0; NESTED_RECURSIVE_PROOF_LENGTH] }
    }
}

impl Eq for RecursiveProof {
    fn eq(self, other: Self) -> bool {
        (self.fields == other.fields)
    }
}

impl Eq for NestedRecursiveProof {
    fn eq(self, other: Self) -> bool {
        (self.fields == other.fields)
    }
}

#[test]
fn serialization_of_empty_recursive() {
    let item = RecursiveProof::empty();
    let serialized = item.serialize();
    let deserialized = RecursiveProof::deserialize(serialized);
    assert(item.eq(deserialized));
}

#[test]
fn serialization_of_empty_nested() {
    let item = NestedRecursiveProof::empty();
    let serialized = item.serialize();
    let deserialized = NestedRecursiveProof::deserialize(serialized);
    assert(item.eq(deserialized));
}
