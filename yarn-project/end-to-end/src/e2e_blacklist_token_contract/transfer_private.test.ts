import { Fr, computeAuthWitMessageHash } from '@aztec/aztec.js';

import { DUPLICATE_NULLIFIER_ERROR } from '../fixtures/fixtures.js';
import { BlacklistTokenContractTest } from './blacklist_token_contract_test.js';

describe('e2e_blacklist_token_contract transfer private', () => {
  const t = new BlacklistTokenContractTest('transfer_private');
  let { asset, tokenSim, wallets, blacklisted } = t;

  beforeAll(async () => {
    await t.applyBaseSnapshots();
    // Beware that we are adding the admin as minter here, which is very slow because it needs multiple blocks.
    await t.applyMintSnapshot();
    await t.setup();
    // Have to destructure again to ensure we have latest refs.
    ({ asset, tokenSim, wallets, blacklisted } = t);
  }, 600_000);

  afterAll(async () => {
    await t.teardown();
  });

  afterEach(async () => {
    // await t.tokenSim.check();
  });

  it('transfer to self', async () => {
    const balance0 = await asset.methods.balance_of_private(wallets[0].getAddress()).simulate();
    const amount = balance0 / 2n;
    expect(amount).toBeGreaterThan(0n);

    await asset.methods.transfer(wallets[0].getAddress(), wallets[0].getAddress(), amount, 0).send().wait();
  });

  it('transfer on behalf of other', async () => {
    const balance0 = await asset.methods.balance_of_private(wallets[0].getAddress()).simulate();
    const amount = balance0 / 2n;
    const nonce = Fr.random();
    expect(amount).toBeGreaterThan(0n);

    // We need to compute the message we want to sign and add it to the wallet as approved
    // docs:start:authwit_transfer_example
    // docs:start:authwit_computeAuthWitMessageHash
    const action = asset
      .withWallet(wallets[1])
      .methods.transfer(wallets[0].getAddress(), wallets[1].getAddress(), amount, nonce);
    // docs:end:authwit_computeAuthWitMessageHash
    // docs:start:create_authwit
    const witness = await wallets[0].createAuthWit({ caller: wallets[1].getAddress(), action });
    // docs:end:create_authwit
    // docs:start:add_authwit
    await wallets[1].addAuthWitness(witness);
    // docs:end:add_authwit
    // docs:end:authwit_transfer_example

    // We give wallets[1] access to wallets[0]'s notes to be able to transfer the notes.
    wallets[1].setScopes([wallets[1].getAddress(), wallets[0].getAddress()]);

    // Perform the transfer
    const {txHash} = await action.send().wait();
    console.log('wut txHash', txHash);
    // // Perform the transfer again, should fail
    // const txReplay = asset
    //   .withWallet(wallets[1])
    //   .methods.transfer(wallets[0].getAddress(), wallets[1].getAddress(), amount, nonce)
    //   .send();
    // await expect(txReplay.wait()).rejects.toThrow(DUPLICATE_NULLIFIER_ERROR);
  });
});
