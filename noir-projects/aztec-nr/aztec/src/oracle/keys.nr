use crate::keys::{PublicKeys, public_keys::{NpkM, IvpkM, OvpkM, TpkM}};
use dep::protocol_types::{address::{AztecAddress, PartialAddress}, point::Point};

#[oracle(getPublicKeysAndPartialAddress)]
unconstrained fn get_public_keys_and_partial_address_oracle(_address: AztecAddress) -> [Field; 13] {}

unconstrained pub fn get_public_keys_and_partial_address(address: AztecAddress) -> (PublicKeys, PartialAddress) {
    // let result = [0; 13];

    let keys = PublicKeys {
        npk_m: NpkM { inner: Point { x: 0x30426e64aee30e998c13c8ceecda3a77807dbead52bc2f3bf0eae851b4b710c1, y: 0x113156a068f603023240c96b4da5474667db3b8711c521c748212a15bc034ea6, is_infinite: false } },
        ivpk_m: IvpkM { inner: Point { x: 0x2825c79cc6a5cbbeef7d6a8f1b6a12b312aa338440aefeb4396148c89147c049, y: 0x129bfd1da54b7062d6b544e7e36b90736350f6fba01228c41c72099509f5701e, is_infinite: false } },
        ovpk_m: OvpkM { inner: Point { x: 0x0edb1e293c3ce91bfc04e3ceaa50d2c541fa9d091c72eb403efb1cfa2cb3357f, y: 0x1341d675fa030ece3113ad53ca34fd13b19b6e9762046734f414824c4d6ade35, is_infinite: false } },
        tpk_m: TpkM { inner: Point { x: 0x041223147b680850dc82e8a55a952d4df20256fe0593d949a9541ca00f0abf15, y: 0x0a8c72e60d0e60f5d804549d48f3044d06140b98ed717a9b532af630c1530791, is_infinite: false } }
    };

    let partial_address = PartialAddress::from_field(8);

    (keys, partial_address)
}
