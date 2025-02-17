import { L2Block, type ViemSignature } from '@aztec/circuit-types';
import { EthAddress } from '@aztec/circuits.js';
import { sleep } from '@aztec/foundation/sleep';
import { RollupAbi } from '@aztec/l1-artifacts';
import { NoopTelemetryClient } from '@aztec/telemetry-client/noop';

import { type MockProxy, mock } from 'jest-mock-extended';
import { type GetTransactionReceiptReturnType, type PrivateKeyAccount } from 'viem';

import { type PublisherConfig, type TxSenderConfig } from './config.js';
import { L1Publisher } from './l1-publisher.js';

interface MockPublicClient {
  getTransactionReceipt: ({ hash }: { hash: '0x${string}' }) => Promise<GetTransactionReceiptReturnType>;
  getBlock(): Promise<{ timestamp: bigint }>;
  getTransaction: ({ hash }: { hash: '0x${string}' }) => Promise<{ input: `0x${string}`; hash: `0x${string}` }>;
  estimateGas: ({ to, data }: { to: '0x${string}'; data: '0x${string}' }) => Promise<bigint>;
}

interface MockRollupContractWrite {
  propose: (
    args: readonly [`0x${string}`, `0x${string}`] | readonly [`0x${string}`, `0x${string}`, `0x${string}`],
    options: { account: PrivateKeyAccount },
  ) => Promise<`0x${string}`>;
}

interface MockRollupContractRead {
  archive: () => Promise<`0x${string}`>;
  getCurrentSlot(): Promise<bigint>;
  validateHeader: (
    args: readonly [
      `0x${string}`,
      ViemSignature[],
      `0x${string}`,
      bigint,
      { ignoreDA: boolean; ignoreSignatures: boolean },
    ],
  ) => Promise<void>;
}

class MockRollupContract {
  constructor(public write: MockRollupContractWrite, public read: MockRollupContractRead, public abi = RollupAbi) {}
}

describe('L1Publisher', () => {
  let rollupContractRead: MockProxy<MockRollupContractRead>;
  let rollupContractWrite: MockProxy<MockRollupContractWrite>;
  let rollupContract: MockRollupContract;

  let publicClient: MockProxy<MockPublicClient>;

  let proposeTxHash: `0x${string}`;
  let proposeTxReceipt: GetTransactionReceiptReturnType;
  let l2Block: L2Block;

  let header: Buffer;
  let archive: Buffer;
  let blockHash: Buffer;
  let body: Buffer;

  let account: PrivateKeyAccount;

  let publisher: L1Publisher;

  const GAS_GUESS = 300_000n;

  beforeEach(() => {
    l2Block = L2Block.random(42);

    header = l2Block.header.toBuffer();
    archive = l2Block.archive.root.toBuffer();
    blockHash = l2Block.header.hash().toBuffer();
    body = l2Block.body.toBuffer();

    proposeTxHash = `0x${Buffer.from('txHashPropose').toString('hex')}`; // random tx hash

    proposeTxReceipt = {
      transactionHash: proposeTxHash,
      status: 'success',
      logs: [],
    } as unknown as GetTransactionReceiptReturnType;

    rollupContractWrite = mock<MockRollupContractWrite>();
    rollupContractRead = mock<MockRollupContractRead>();
    rollupContract = new MockRollupContract(rollupContractWrite, rollupContractRead);

    publicClient = mock<MockPublicClient>();

    const config = {
      l1RpcUrl: `http://127.0.0.1:8545`,
      l1ChainId: 1,
      publisherPrivateKey: `0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80`,
      l1Contracts: {
        rollupAddress: EthAddress.ZERO.toString(),
      },
      l1PublishRetryIntervalMS: 1,
    } as unknown as TxSenderConfig & PublisherConfig;

    publisher = new L1Publisher(config, new NoopTelemetryClient());

    (publisher as any)['rollupContract'] = rollupContract;
    (publisher as any)['publicClient'] = publicClient;

    account = (publisher as any)['account'];

    rollupContractRead.getCurrentSlot.mockResolvedValue(l2Block.header.globalVariables.slotNumber.toBigInt());
    publicClient.getBlock.mockResolvedValue({ timestamp: 12n });
    publicClient.estimateGas.mockResolvedValue(GAS_GUESS);
  });

  it('publishes and propose l2 block to l1', async () => {
    rollupContractRead.archive.mockResolvedValue(l2Block.header.lastArchive.root.toString() as `0x${string}`);
    rollupContractWrite.propose.mockResolvedValueOnce(proposeTxHash);

    publicClient.getTransactionReceipt.mockResolvedValueOnce(proposeTxReceipt);

    const result = await publisher.proposeL2Block(l2Block);

    expect(result).toEqual(true);

    const args = [
      `0x${header.toString('hex')}`,
      `0x${archive.toString('hex')}`,
      `0x${blockHash.toString('hex')}`,
      [],
      [],
      `0x${body.toString('hex')}`,
    ] as const;
    expect(rollupContractWrite.propose).toHaveBeenCalledWith(args, {
      account: account,
      gas: L1Publisher.PROPOSE_GAS_GUESS + GAS_GUESS,
    });
    expect(publicClient.getTransactionReceipt).toHaveBeenCalledWith({ hash: proposeTxHash });
  });

  it('does not retry if sending a propose tx fails', async () => {
    rollupContractRead.archive.mockResolvedValue(l2Block.header.lastArchive.root.toString() as `0x${string}`);
    rollupContractWrite.propose.mockRejectedValueOnce(new Error()).mockResolvedValueOnce(proposeTxHash);

    const result = await publisher.proposeL2Block(l2Block);

    expect(result).toEqual(false);
    expect(rollupContractWrite.propose).toHaveBeenCalledTimes(1);
  });

  it('does not retry if simulating a publish and propose tx fails', async () => {
    rollupContractRead.archive.mockResolvedValue(l2Block.header.lastArchive.root.toString() as `0x${string}`);
    rollupContractRead.validateHeader.mockRejectedValueOnce(new Error('Test error'));

    await expect(publisher.proposeL2Block(l2Block)).rejects.toThrow();

    expect(rollupContractRead.validateHeader).toHaveBeenCalledTimes(1);
    expect(rollupContractWrite.propose).toHaveBeenCalledTimes(0);
  });

  it('does not retry if sending a publish and propose tx fails', async () => {
    rollupContractRead.archive.mockResolvedValue(l2Block.header.lastArchive.root.toString() as `0x${string}`);
    rollupContractWrite.propose.mockRejectedValueOnce(new Error());

    const result = await publisher.proposeL2Block(l2Block);

    expect(result).toEqual(false);
    expect(rollupContractWrite.propose).toHaveBeenCalledTimes(1);
  });

  it('retries if fetching the receipt fails (propose)', async () => {
    rollupContractRead.archive.mockResolvedValue(l2Block.header.lastArchive.root.toString() as `0x${string}`);
    rollupContractWrite.propose.mockResolvedValueOnce(proposeTxHash);
    publicClient.getTransactionReceipt.mockRejectedValueOnce(new Error()).mockResolvedValueOnce(proposeTxReceipt);

    const result = await publisher.proposeL2Block(l2Block);

    expect(result).toEqual(true);
    expect(publicClient.getTransactionReceipt).toHaveBeenCalledTimes(2);
  });

  it('retries if fetching the receipt fails (publish propose)', async () => {
    rollupContractRead.archive.mockResolvedValue(l2Block.header.lastArchive.root.toString() as `0x${string}`);
    rollupContractWrite.propose.mockResolvedValueOnce(proposeTxHash as `0x${string}`);
    publicClient.getTransactionReceipt.mockRejectedValueOnce(new Error()).mockResolvedValueOnce(proposeTxReceipt);

    const result = await publisher.proposeL2Block(l2Block);

    expect(result).toEqual(true);
    expect(publicClient.getTransactionReceipt).toHaveBeenCalledTimes(2);
  });

  it('returns false if publish and propose tx reverts', async () => {
    rollupContractRead.archive.mockResolvedValue(l2Block.header.lastArchive.root.toString() as `0x${string}`);
    rollupContractWrite.propose.mockResolvedValueOnce(proposeTxHash);
    publicClient.getTransactionReceipt.mockResolvedValueOnce({ ...proposeTxReceipt, status: 'reverted' });

    const result = await publisher.proposeL2Block(l2Block);

    expect(result).toEqual(false);
  });

  it('returns false if propose tx reverts', async () => {
    rollupContractRead.archive.mockResolvedValue(l2Block.header.lastArchive.root.toString() as `0x${string}`);

    publicClient.getTransactionReceipt.mockResolvedValueOnce({ ...proposeTxReceipt, status: 'reverted' });

    const result = await publisher.proposeL2Block(l2Block);

    expect(result).toEqual(false);
  });

  it('returns false if sending publish and progress tx is interrupted', async () => {
    rollupContractRead.archive.mockResolvedValue(l2Block.header.lastArchive.root.toString() as `0x${string}`);
    rollupContractWrite.propose.mockImplementationOnce(() => sleep(10, proposeTxHash) as Promise<`0x${string}`>);

    const resultPromise = publisher.proposeL2Block(l2Block);
    publisher.interrupt();
    const result = await resultPromise;

    expect(result).toEqual(false);
    expect(publicClient.getTransactionReceipt).not.toHaveBeenCalled();
  });

  it('returns false if sending propose tx is interrupted', async () => {
    rollupContractRead.archive.mockResolvedValue(l2Block.header.lastArchive.root.toString() as `0x${string}`);
    rollupContractWrite.propose.mockImplementationOnce(() => sleep(10, proposeTxHash) as Promise<`0x${string}`>);

    const resultPromise = publisher.proposeL2Block(l2Block);
    publisher.interrupt();
    const result = await resultPromise;

    expect(result).toEqual(false);
    expect(publicClient.getTransactionReceipt).not.toHaveBeenCalled();
  });
});
