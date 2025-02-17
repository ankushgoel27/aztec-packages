import { type BlockAttestation, type BlockProposal, type TxHash } from '@aztec/circuit-types';
import { type Header } from '@aztec/circuits.js';
import { type Fr } from '@aztec/foundation/fields';
import { createDebugLogger } from '@aztec/foundation/log';
import { sleep } from '@aztec/foundation/sleep';
import { type P2P } from '@aztec/p2p';

import { type ValidatorClientConfig } from './config.js';
import { ValidationService } from './duties/validation_service.js';
import { AttestationTimeoutError, TransactionsNotAvailableError } from './errors/validator.error.js';
import { type ValidatorKeyStore } from './key_store/interface.js';
import { LocalKeyStore } from './key_store/local_key_store.js';

export interface Validator {
  start(): Promise<void>;
  registerBlockProposalHandler(): void;

  // Block validation responsiblities
  createBlockProposal(header: Header, archive: Fr, txs: TxHash[]): Promise<BlockProposal>;
  attestToProposal(proposal: BlockProposal): void;

  broadcastBlockProposal(proposal: BlockProposal): void;
  collectAttestations(proposal: BlockProposal, numberOfRequiredAttestations: number): Promise<BlockAttestation[]>;
}

/** Validator Client
 */
export class ValidatorClient implements Validator {
  private validationService: ValidationService;

  constructor(
    keyStore: ValidatorKeyStore,
    private p2pClient: P2P,
    private attestationPoolingIntervalMs: number,
    private attestationWaitTimeoutMs: number,
    private log = createDebugLogger('aztec:validator'),
  ) {
    //TODO: We need to setup and store all of the currently active validators https://github.com/AztecProtocol/aztec-packages/issues/7962

    this.validationService = new ValidationService(keyStore);
    this.log.verbose('Initialized validator');
  }

  static new(config: ValidatorClientConfig, p2pClient: P2P) {
    const localKeyStore = new LocalKeyStore(config.validatorPrivateKey);

    const validator = new ValidatorClient(
      localKeyStore,
      p2pClient,
      config.attestationPoolingIntervalMs,
      config.attestationWaitTimeoutMs,
    );
    validator.registerBlockProposalHandler();
    return validator;
  }

  public start() {
    // Sync the committee from the smart contract
    // https://github.com/AztecProtocol/aztec-packages/issues/7962

    this.log.info('Validator started');
    return Promise.resolve();
  }

  public registerBlockProposalHandler() {
    const handler = (block: BlockProposal): Promise<BlockAttestation | undefined> => {
      return this.attestToProposal(block);
    };
    this.p2pClient.registerBlockProposalHandler(handler);
  }

  async attestToProposal(proposal: BlockProposal): Promise<BlockAttestation | undefined> {
    // Check that all of the tranasctions in the proposal are available in the tx pool before attesting
    try {
      await this.ensureTransactionsAreAvailable(proposal);
    } catch (error: any) {
      if (error instanceof TransactionsNotAvailableError) {
        this.log.error(`Transactions not available, skipping attestation ${error.message}`);
      }
      return undefined;
    }
    this.log.debug(`Transactions available, attesting to proposal with ${proposal.txs.length} transactions`);

    // If the above function does not throw an error, then we can attest to the proposal
    return this.validationService.attestToProposal(proposal);
  }

  /**
   * Ensure that all of the transactions in the proposal are available in the tx pool before attesting
   *
   * 1. Check if the local tx pool contains all of the transactions in the proposal
   * 2. If any transactions are not in the local tx pool, request them from the network
   * 3. If we cannot retrieve them from the network, throw an error
   * @param proposal - The proposal to attest to
   */
  async ensureTransactionsAreAvailable(proposal: BlockProposal) {
    const txHashes: TxHash[] = proposal.txs;
    const transactionStatuses = await Promise.all(txHashes.map(txHash => this.p2pClient.getTxStatus(txHash)));

    const missingTxs = txHashes.filter((_, index) => !['pending', 'mined'].includes(transactionStatuses[index] ?? ''));

    if (missingTxs.length === 0) {
      return; // All transactions are available
    }

    this.log.verbose(`Missing ${missingTxs.length} transactions in the tx pool, requesting from the network`);

    const requestedTxs = await this.p2pClient.requestTxs(missingTxs);
    if (requestedTxs.some(tx => tx === undefined)) {
      this.log.error(`Failed to request transactions from the network: ${missingTxs.join(', ')}`);
      throw new TransactionsNotAvailableError(missingTxs);
    }
  }

  createBlockProposal(header: Header, archive: Fr, txs: TxHash[]): Promise<BlockProposal> {
    return this.validationService.createBlockProposal(header, archive, txs);
  }

  broadcastBlockProposal(proposal: BlockProposal): void {
    this.p2pClient.broadcastProposal(proposal);
  }

  // TODO(https://github.com/AztecProtocol/aztec-packages/issues/7962)
  async collectAttestations(
    proposal: BlockProposal,
    numberOfRequiredAttestations: number,
  ): Promise<BlockAttestation[]> {
    // Wait and poll the p2pClient's attestation pool for this block until we have enough attestations
    const slot = proposal.header.globalVariables.slotNumber.toBigInt();
    this.log.info(`Waiting for ${numberOfRequiredAttestations} attestations for slot: ${slot}`);

    const myAttestation = await this.validationService.attestToProposal(proposal);

    const startTime = Date.now();

    while (true) {
      const attestations = [myAttestation, ...(await this.p2pClient.getAttestationsForSlot(slot))];

      if (attestations.length >= numberOfRequiredAttestations) {
        this.log.info(`Collected all ${numberOfRequiredAttestations} attestations for slot, ${slot}`);
        return attestations;
      }

      const elapsedTime = Date.now() - startTime;
      if (elapsedTime > this.attestationWaitTimeoutMs) {
        this.log.error(`Timeout waiting for ${numberOfRequiredAttestations} attestations for slot, ${slot}`);
        throw new AttestationTimeoutError(numberOfRequiredAttestations, slot);
      }

      this.log.verbose(
        `Collected ${attestations.length} attestations so far, waiting ${this.attestationPoolingIntervalMs}ms for more...`,
      );
      await sleep(this.attestationPoolingIntervalMs);
    }
  }
}
