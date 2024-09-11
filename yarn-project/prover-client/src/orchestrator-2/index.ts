import {
  Body,
  L2Block,
  MerkleTreeId,
  MerkleTreeOperations,
  ProcessedTx,
  ProvingRequest,
  ProvingRequestResult,
  ProvingRequestType,
  PublicInputsAndRecursiveProof,
  ServerCircuitProver,
  SimulationRequestResult,
  TxEffect,
  toTxEffect,
} from '@aztec/circuit-types';
import {
  ARCHIVE_HEIGHT,
  AppendOnlyTreeSnapshot,
  BaseOrMergeRollupPublicInputs,
  BaseParityInputs,
  BaseRollupInputs,
  BlockMergeRollupInputs,
  BlockRootOrBlockMergePublicInputs,
  BlockRootRollupInputs,
  ContentCommitment,
  Fr,
  GasFees,
  GlobalVariables,
  Header,
  L1_TO_L2_MSG_SUBTREE_HEIGHT,
  L1_TO_L2_MSG_SUBTREE_SIBLING_PATH_LENGTH,
  NESTED_RECURSIVE_PROOF_LENGTH,
  NUMBER_OF_L1_L2_MESSAGES_PER_ROLLUP,
  NUM_BASE_PARITY_PER_ROOT_PARITY,
  NUM_MSGS_PER_BASE_PARITY,
  ParityPublicInputs,
  PreviousRollupBlockData,
  PreviousRollupData,
  RECURSIVE_PROOF_LENGTH,
  RecursiveProof,
  RootParityInput,
  RootParityInputs,
  RootRollupInputs,
  RootRollupPublicInputs,
  StateReference,
  VerificationKey,
  VerificationKeyData,
} from '@aztec/circuits.js';
import { makeTuple } from '@aztec/foundation/array';
import { padArrayEnd, times } from '@aztec/foundation/collection';
import { sha256Trunc } from '@aztec/foundation/crypto';
import { createDebugLogger } from '@aztec/foundation/log';
import { type PromiseWithResolvers, promiseWithResolvers } from '@aztec/foundation/promise';
import { Tuple, assertLength, mapTuple } from '@aztec/foundation/serialize';
import { RequiredBy } from '@aztec/foundation/types';
import { getVKMembershipWitness, getVKTreeRoot } from '@aztec/noir-protocol-circuits-types';
import { TelemetryClient } from '@aztec/telemetry-client';

import {
  getPreviousRollupDataFromPublicInputs,
  getRootTreeSiblingPath,
  getSubtreeSiblingPath,
  getTreeSnapshot,
  validateBlockRootOutput,
} from '../orchestrator/block-building-helpers.js';

// as usual, I hate all names here

type Proof = Buffer;

type PayloadUrl = string;

interface Prover {
  prove<Request extends ProvingRequest>(request: Request): Promise<ProvingRequestResult<Request['type']>>;
}

interface Simulator {
  simulate<Request extends ProvingRequest>(request: Request): Promise<SimulationRequestResult<Request['type']>>;
}

interface PayloadStore {
  save(id: string, payload: Buffer): Promise<PayloadUrl>;
  load(payloadUrl: PayloadUrl): Promise<Buffer | undefined>;
}

interface MetadataStore {
  save(id: string, metadata: any): Promise<void>;
  load(id: string): Promise<any | undefined>;
  list(parentId: string): Promise<any[]>;
}

type OrchestratorContext = {
  db: MerkleTreeOperations;
  simulator: Simulator;
  prover: Prover;
  telemetryClient: TelemetryClient;
  payloadStore: PayloadStore;
  metadataStore: MetadataStore;
  proverId: Fr;
  options: {
    checkSimulationMatchesProof: boolean;
  };
};

// public inputs for root rollup circuit
type L2Epoch = {
  data: PublicInputsAndRecursiveProof<RootRollupPublicInputs>;
};

class TxOrchestrator {
  public readonly level = 0;

  constructor(
    public readonly tx: ProcessedTx,
    public readonly index: number,
    private readonly gasFees: GasFees,
    private readonly context: OrchestratorContext,
  ) {}

  // return an identifier
  // randomly generated? hierarchical? depends on blocknum and txindex?
  public getId() {}

  public getTxEffect() {
    return toTxEffect(this.tx, this.gasFees);
  }

  // updates world state!
  @memoize
  public updateState(): Promise<BaseRollupInputs> {
    throw new Error('Unimplemented');
    // inputs = buildBaseRollupInputs()
    // save inputs to db
    // commit changes to world-state
    // q: do we want to save w-s here, or on block orchestrator end block? can we resume a public-processor halfway through??
  }

  // returns output of base rollup
  @memoize
  public simulate(): Promise<BaseOrMergeRollupPublicInputs> {
    throw new Error('Unimplemented');
    // await buildInput
    // run base rollup sim
    // save output to db
  }

  // returns output and proof
  @memoize
  public prove(): Promise<PublicInputsAndRecursiveProof<BaseOrMergeRollupPublicInputs>> {
    throw new Error('Unimplemented');
    // await buildInput
    // run prover
    // validate output?
    // store proof in db?
    // return proof!
  }

  public async clear() {
    // delete eveyrthing from db
  }

  public async save() {
    // saves its current state to db
    // should save its identifier and status to db
    // and payload to storage shared with agents
    // so it can share payload via url
  }

  public static async load(id: string, context: OrchestratorContext): Promise<TxOrchestrator> {
    // load from db given its identifier
    throw new Error('Unimplemented');
  }
}

type ParityState = {
  l1ToL2Messages: Tuple<Fr, typeof NUMBER_OF_L1_L2_MESSAGES_PER_ROLLUP>;
  newL1ToL2MessageTreeRootSiblingPath: Tuple<Fr, typeof L1_TO_L2_MSG_SUBTREE_SIBLING_PATH_LENGTH>;
  messageTreeSnapshot: AppendOnlyTreeSnapshot;
};

class ParityOrchestrator {
  private baseParityJobs?: BaseParityCircuit[];
  private rootParityJob?: RootParityCircuit;

  private simulationPromise = promiseWithResolvers<ParityPublicInputs>();
  private proofPromise = promiseWithResolvers<RootParityInput<typeof NESTED_RECURSIVE_PROOF_LENGTH>>();

  constructor(private readonly unpaddedl1ToL2Messages: Fr[], private readonly context: OrchestratorContext) {
    this.handleError = this.handleError.bind(this);
  }

  @memoize
  public async updateState() {
    const messageTreeSnapshot = await getTreeSnapshot(MerkleTreeId.L1_TO_L2_MESSAGE_TREE, this.context.db);

    const newL1ToL2MessageTreeRootSiblingPath = padArrayEnd(
      await getSubtreeSiblingPath(MerkleTreeId.L1_TO_L2_MESSAGE_TREE, L1_TO_L2_MSG_SUBTREE_HEIGHT, this.context.db),
      Fr.ZERO,
      L1_TO_L2_MSG_SUBTREE_SIBLING_PATH_LENGTH,
    );

    // Update the local trees to include the new l1 to l2 messages
    await this.context.db.appendLeaves(MerkleTreeId.L1_TO_L2_MESSAGE_TREE, this.l1ToL2Messages);

    return {
      l1ToL2Messages: this.l1ToL2Messages,
      messageTreeSnapshot,
      newL1ToL2MessageTreeRootSiblingPath,
    };
  }

  @memoize
  public simulate() {
    this.start();
    return this.simulationPromise.promise;
  }

  @memoize
  public prove() {
    this.start();
    return this.proofPromise.promise;
  }

  private get l1ToL2Messages() {
    return padArrayEnd(
      this.unpaddedl1ToL2Messages,
      Fr.ZERO,
      NUMBER_OF_L1_L2_MESSAGES_PER_ROLLUP,
      'Too many L1 to L2 messages',
    );
  }

  @memoize
  private start() {
    const rootParityJob = this.createRootParityProvingJob();
    this.rootParityJob = rootParityJob;

    const count = NUM_BASE_PARITY_PER_ROOT_PARITY;
    const messageBatches = times(count, i => BaseParityInputs.sliceMessages(this.l1ToL2Messages, i));

    this.baseParityJobs = messageBatches.map((messages, index) => {
      return this.createBaseParityProvingJob(messages, index, rootParityJob);
    });
  }

  private createBaseParityProvingJob(
    messages: Tuple<Fr, typeof NUM_BASE_PARITY_PER_ROOT_PARITY>,
    index: number,
    rootParityJob: RootParityCircuit,
  ) {
    const job = new BaseParityCircuit(messages, index, this.context);
    void job
      .prove()
      .then(proof => rootParityJob.setNested({ proof }, index))
      .catch(this.handleError);
    void job
      .simulate()
      .then(simulation => rootParityJob.setNested({ simulation }, index))
      .catch(this.handleError);
    return job;
  }

  private createRootParityProvingJob() {
    const rootParityJob = new RootParityCircuit(this.context);
    void rootParityJob
      .prove()
      .then(proof => this.proofPromise.resolve(proof))
      .catch(this.handleError);
    void rootParityJob
      .simulate()
      .then(simulation => this.simulationPromise.resolve(simulation))
      .catch(this.handleError);
    return rootParityJob;
  }

  private handleError(err: Error) {
    throw err; // TODO: proper error handling
  }
}

type BlockOrchestratorStatus = 'created' | 'processed-txs' | 'updated-state' | 'proven';

type BlockOrchestratorMetadata = {
  index: number;
  numTxs: number;
  globalVariables: GlobalVariables;
  l1ToL2Messages: Fr[];
  status: BlockOrchestratorStatus;
  body?: Body;
};

type NestedRecursiveProof = {
  proof: RecursiveProof<typeof NESTED_RECURSIVE_PROOF_LENGTH>;
  verificationKey: VerificationKeyData;
};

class BlockOrchestrator {
  public readonly level = 0;

  private readonly txs: TxOrchestrator[] = [];
  private readonly merges: MergeRollupProvingJob[] = [];

  private parity?: ParityOrchestrator;
  private body?: Body;

  private simulationPromise = promiseWithResolvers<BlockRootOrBlockMergePublicInputs>();
  private blockPromise = promiseWithResolvers<L2Block>();
  private proofPromise = promiseWithResolvers<PublicInputsAndRecursiveProof<BlockRootOrBlockMergePublicInputs>>();

  private status: BlockOrchestratorStatus = 'created'; // we need to move this forward, it's needed for rehydration

  constructor(
    /** Index of the block within the epoch (zero-based) */
    public readonly index: number,
    private readonly numTxs: number,
    private readonly globalVariables: GlobalVariables,
    private readonly l1ToL2Messages: Fr[],
    private readonly context: OrchestratorContext,
  ) {
    this.handleError = this.handleError.bind(this);
  }

  static load(_id: string, _context: OrchestratorContext) {
    throw new Error('Unimplemented');
    // const metadata: BlockOrchestratorMetadata = await context.metadataStore.load(id);
    // if (!metadata) {
    //   throw new Error('Block not found');
    // }

    // const { index, numTxs, globalVariables, l1ToL2Messages } = metadata;
    // const orchestrator = new BlockOrchestrator(index, numTxs, globalVariables, l1ToL2Messages, context);

    // if (metadata.status === 'proven') {
    //   const proofKey = id + 'proof'; // need to devise a proper scheme for this
    //   const proof = (await context.payloadStore.load(
    //     proofKey,
    //   )) as unknown as PublicInputsAndRecursiveProof<BlockRootOrBlockMergePublicInputs>; // deserialize!
    //   if (!proof) {
    //     // fall back to no-proof
    //   }
    //   // orchestrator.handleBlockRootProof(proof); // should also set the block itself..?
    //   return orchestrator; // we're good!
    // }

    // if (metadata.status === 'updated-state' || metadata.status === 'processed-txs') {
    //   // set l2 block body if all txs have been processed
    //   orchestrator.body = metadata.body;
    // }

    // // now, we need to rehydrate the orchestrator with partial proving state
    // // let's start with the merges
    // // what are we loading here? just the ids?
    // const merges = await context.metadataStore.list(id + 'merges'); // again, proper key scheme!

    // // we actually only need the highest full level of merges, can forget about the rest
    // // need to get that from the list of merges given their level and index, or return undefined if none
    // const highestLevel = merges; // getHighestCompleteLevel(merges);

    // if (highestLevel) {
    //   for (const merge of merges) {
    //     const mergeOrch = await MergeRollupProvingJob.load(merge, context);
    //     orchestrator.merges.push(mergeOrch); // set them to the proper index!
    //     // wire them up!
    //   }
    // }

    // // if we don't have a full level of merges, we need to rehydrate the txs
    // const txs = await context.metadataStore.list(id + 'txs');
    // for (const tx of txs) {
    //   const txOrch = await TxOrchestrator.load(tx, context);
    //   orchestrator.txs.push(txOrch); // set them to the proper index!
    //   // wire them up!
    // }

    // return orchestrator;
  }

  public async start() {
    const parity = this.createParityOrchestrator();
    const state = await parity.updateState();
    this.getBlockRoot().setRootParity({ state });
    this.parity = parity;
  }

  private createParityOrchestrator() {
    const parity = new ParityOrchestrator(this.l1ToL2Messages, this.context);
    void parity
      .simulate()
      .then(simulation => this.getBlockRoot().setRootParity({ simulation }))
      .catch(this.handleError);
    void parity
      .prove()
      .then(proof => this.getBlockRoot().setRootParity({ proof }))
      .catch(this.handleError);
    return parity;
  }

  // creates new tx orch
  // calls build-input to update world state
  public async addTx(processedTx: ProcessedTx): Promise<void> {
    const index = this.txs.length;
    const txOrchestrator = this.createTxOrchestrator(processedTx, index);
    this.txs.push(txOrchestrator);
    await txOrchestrator.updateState();
  }

  private createTxOrchestrator(processedTx: ProcessedTx, index: number) {
    const txOrchestrator = new TxOrchestrator(processedTx, index, this.globalVariables.gasFees, this.context);
    this.wireBaseOrMergeOutputs(txOrchestrator);
    return txOrchestrator;
  }

  public async padBlock(): Promise<void> {
    // TODO: create padding txs and await them!

    const nonEmptyTxEffects: TxEffect[] = this.txs.map(tx => tx.getTxEffect()).filter(txEffect => !txEffect.isEmpty());
    const body = new Body(nonEmptyTxEffects);
    this.body = body;
    this.status = 'processed-txs';
  }

  // updates world state with the result of the block
  // calls getBlock under the hood
  @memoize
  public async updateState(): Promise<void> {
    const block = await this.getBlock();
    await this.context.db.updateArchive(block.header);
    this.status = 'updated-state';

    const rootOutput = await this.getBlockRoot().simulate();
    await validateBlockRootOutput(rootOutput, block.header, this.context.db);
  }

  public simulate(): Promise<BlockRootOrBlockMergePublicInputs> {
    return this.simulationPromise.promise;
  }

  // returns full l2 block
  // updates world state with txs, but does not yet touch the archive tree (updateState takes care of that)
  public getBlock(): Promise<L2Block> {
    return this.blockPromise.promise;
  }

  public prove(): Promise<PublicInputsAndRecursiveProof<BlockRootOrBlockMergePublicInputs>> {
    return this.proofPromise.promise;
  }

  // this would be better served by a "tree-manager" component
  private wireBaseOrMergeOutputs(source: TxOrchestrator | MergeRollupProvingJob) {
    const { level, index } = source;

    const handler = (
      out:
        | { simulation: BaseOrMergeRollupPublicInputs }
        | { proof: PublicInputsAndRecursiveProof<BaseOrMergeRollupPublicInputs> },
    ) => {
      const parent = this.getParentMerge(level, index);
      const isLeft = index % 2 === 0;
      parent.setNested(out, isLeft);
    };

    void source
      .prove()
      .then(proof => handler({ proof }))
      .catch(this.handleError);

    void source
      .simulate()
      .then(simulation => handler({ simulation }))
      .catch(this.handleError);
  }

  // this too
  private getParentMerge(level: number, index: number): MergeRollupProvingJob | BlockRootCircuit {
    // check this math and off-by-ones
    const depth = Math.ceil(Math.log2(this.numTxs));
    if (level === depth) {
      return this.getBlockRoot();
    }

    // yep, definitely check this math
    const position = (1 << level) + (index << 1);
    if (!this.merges[position]) {
      const merge = new MergeRollupProvingJob(level, index);
      this.wireBaseOrMergeOutputs(merge);
      this.merges[position] = merge;
    }
    return this.merges[position];
  }

  @memoize
  private getBlockRoot() {
    const blockRoot = new BlockRootCircuit(this.context);
    void blockRoot
      .simulate()
      .then(output => {
        this.simulationPromise.resolve(output);
        return this.makeBlock(output);
      })
      .then(block => this.blockPromise.resolve(block))
      .catch(this.handleError);

    void blockRoot
      .prove()
      .then(proof => {
        this.status = 'proven';
        this.proofPromise.resolve(proof);
      })
      .catch(this.handleError);

    return blockRoot;
  }

  private async makeBlock(blockRootOutputs: BlockRootOrBlockMergePublicInputs): Promise<L2Block> {
    const archive = blockRootOutputs.newArchive;

    const header = await this.makeHeader(blockRootOutputs);
    if (!header.hash().equals(blockRootOutputs.endBlockHash)) {
      throw new Error(
        `Block header hash mismatch: ${header.hash().toString()} !== ${blockRootOutputs.endBlockHash.toString()}`,
      );
    }

    // build the body
    // note we may want to save the tx effect separately, so we don't need to rehydrate all tx orchestrators if this block orch goes down
    const nonEmptyTxEffects: TxEffect[] = this.txs.map(tx => tx.getTxEffect()).filter(txEffect => !txEffect.isEmpty());
    const body = new Body(nonEmptyTxEffects);
    if (!body.getTxsEffectsHash().equals(header.contentCommitment.txsEffectsHash)) {
      const bodyTxEffectsHex = body.getTxsEffectsHash().toString('hex');
      const headerTxEffectsHex = header.contentCommitment.txsEffectsHash.toString('hex');
      throw new Error(`Txs effects hash mismatch: ${bodyTxEffectsHex} != ${headerTxEffectsHex}`);
    }

    return L2Block.fromFields({ archive, header, body });
  }

  private async makeHeader(blockRootOutputs: BlockRootOrBlockMergePublicInputs) {
    const {
      leftMerge: { inputs: leftMerge },
      rightMerge: { inputs: rightMerge },
      rootParity: { inputs: rootParity },
    } = this.getBlockRoot() as BlockRootReadyForSimulation;

    const contentCommitment = new ContentCommitment(
      new Fr(leftMerge.numTxs + rightMerge.numTxs),
      sha256Trunc(Buffer.concat([leftMerge.txsEffectsHash.toBuffer(), rightMerge.txsEffectsHash.toBuffer()])),
      rootParity.shaRoot.toBuffer(),
      sha256Trunc(Buffer.concat([leftMerge.outHash.toBuffer(), rightMerge.outHash.toBuffer()])),
    );
    const state = new StateReference(
      await getTreeSnapshot(MerkleTreeId.L1_TO_L2_MESSAGE_TREE, this.context.db),
      rightMerge.end,
    );

    const fees = leftMerge.accumulatedFees.add(rightMerge.accumulatedFees);
    const header = new Header(blockRootOutputs.previousArchive, contentCommitment, state, this.globalVariables, fees);
    return header;
  }

  private handleError(_err: Error) {
    throw new Error('Unimplemented');
    // cancel all outstanding work
    // reject all outstanding promises
    // log loudly
    // this will be repeated across all three orchestrators, maybe refactor into helper?
  }
}

class EpochOrchestrator {
  private readonly blocks: BlockOrchestrator[] = [];

  constructor(public readonly numBlocks: number, private readonly context: OrchestratorContext) {
    this.handleError = this.handleError.bind(this);
  }

  public addBlock(numTxs: number, globalVariables: GlobalVariables, l1ToL2Messages: Fr[]) {
    const index = this.blocks.length;
    const blockOrchestrator = new BlockOrchestrator(index, numTxs, globalVariables, l1ToL2Messages, this.context);
    this.wireBlockOrMergeOutputs(blockOrchestrator);
    this.blocks.push(blockOrchestrator);
  }

  public async addTx(processedTx: ProcessedTx): Promise<void> {
    await this.currentBlock.addTx(processedTx);
  }

  public async endBlock(): Promise<void> {
    await this.currentBlock.padBlock();
    await this.currentBlock.updateState();
  }

  private get currentBlock() {
    return this.blocks[this.blocks.length - 1];
  }

  // this is dup from the block orchestrator, should be refactored
  private wireBlockOrMergeOutputs(source: BlockOrchestrator | BlockMergeCircuit) {
    const { level, index } = source;

    const handler = (
      out:
        | { simulation: BlockRootOrBlockMergePublicInputs }
        | { proof: PublicInputsAndRecursiveProof<BlockRootOrBlockMergePublicInputs> },
    ) => {
      const parent = this.getParentMerge(level, index);
      const isLeft = index % 2 === 0;
      parent.setNested(out, isLeft);
    };

    void source
      .prove()
      .then(proof => handler({ proof }))
      .catch(this.handleError);

    void source
      .simulate()
      .then(simulation => handler({ simulation }))
      .catch(this.handleError);
  }

  // this is dup from the block orchestrator, should be refactored
  private getParentMerge(level: number, index: number): BlockMergeCircuit | RootRollupCircuit {
    // check this math and off-by-ones
    const depth = Math.ceil(Math.log2(this.numBlocks));
    if (level === depth) {
      return this.getRootRollup();
    }

    // yep, definitely check this math
    const position = (1 << level) + (index << 1);
    if (!this.merges[position]) {
      const merge = new MergeRollupProvingJob(level, index);
      this.wireBaseOrMergeOutputs(merge);
      this.merges[position] = merge;
    }
    return this.merges[position];
  }

  @memoize
  private getRootRollup() {
    return new RootRollupCircuit(this.context);
  }

  // pads epoch with empty blocks if needed
  // do we really need this method? don't we know block count in advance?
  public async endEpoch(): Promise<void> {}

  public simulate(): Promise<RootRollupPublicInputs> {
    return this.getRootRollup().simulate();
  }

  public prove(): Promise<PublicInputsAndRecursiveProof<RootRollupPublicInputs>> {
    return this.getRootRollup().prove();
  }

  private handleError(_err: Error) {
    throw new Error('Unimplemented');
  }
}

// kinda? needs a rename
// have all little classes below implement this
interface Circuit<Type extends ProvingRequestType> {
  simulate(): Promise<SimulationRequestResult<Type>>;
  prove(): Promise<ProvingRequestResult<Type>>;
}

class BaseProvingJob {}

class AvmProvingJob {}

class PublicKernelProvingJob {}

class TubeProvingJob {}

class BaseRollupProvingJob {}

class BaseParityCircuit implements Circuit<typeof ProvingRequestType.BASE_PARITY> {
  constructor(
    public readonly messages: Tuple<Fr, typeof NUM_MSGS_PER_BASE_PARITY>,
    public readonly index: number,
    private readonly context: OrchestratorContext,
  ) {}

  @memoize
  public buildInputs() {
    return new BaseParityInputs(this.messages, getVKTreeRoot());
  }

  public simulate(): Promise<ParityPublicInputs> {
    return this.context.simulator.simulate({ type: ProvingRequestType.BASE_PARITY, inputs: this.buildInputs() });
  }

  public prove(): Promise<RootParityInput<439>> {
    return this.context.prover.prove({ type: ProvingRequestType.BASE_PARITY, inputs: this.buildInputs() });
  }
}

type RootParityReadyForSimulation = RootParityCircuit & {
  children: Tuple<{ publicInputs: ParityPublicInputs }, typeof NUM_BASE_PARITY_PER_ROOT_PARITY>;
};

type RootParityReadyForProving = RootParityCircuit & {
  children: Tuple<RootParityInput<typeof RECURSIVE_PROOF_LENGTH>, typeof NUM_BASE_PARITY_PER_ROOT_PARITY>;
};

class RootParityCircuit implements Circuit<typeof ProvingRequestType.ROOT_PARITY> {
  // Every child circuit can be pending, simulated, or proven
  public readonly children: Tuple<
    RootParityInput<typeof RECURSIVE_PROOF_LENGTH> | { publicInputs: ParityPublicInputs } | undefined,
    typeof NUM_BASE_PARITY_PER_ROOT_PARITY
  >;

  // These promises are resolved once the nested inputs are ready
  private simulationReadyPromise = promiseWithResolvers<RootParityReadyForSimulation['children']>();
  private provingReadyPromise = promiseWithResolvers<RootParityReadyForProving['children']>();

  constructor(private context: OrchestratorContext) {
    this.children = makeTuple(NUM_BASE_PARITY_PER_ROOT_PARITY, () => undefined);
  }

  public setNested(
    input: { proof: RootParityInput<typeof RECURSIVE_PROOF_LENGTH> } | { simulation: ParityPublicInputs },
    index: number,
  ) {
    if (index > NUM_BASE_PARITY_PER_ROOT_PARITY) {
      throw new Error('Invalid child parity index.');
    }

    if ('proof' in input) {
      this.children[index] = input.proof;
    } else if (this.children[index] === undefined) {
      this.children[index] = { publicInputs: input.simulation };
    }

    if (this.isReadyForSimulation()) {
      this.simulationReadyPromise.resolve(this.children);
    }

    if (this.isReadyForProving()) {
      this.provingReadyPromise.resolve(this.children);
    }
  }

  @memoize
  private async getSimulationInputs() {
    const children = await this.simulationReadyPromise.promise;
    return new RootParityInputs(
      mapTuple(children, child => RootParityInput.withEmptyProof(child, NESTED_RECURSIVE_PROOF_LENGTH)),
    );
  }

  @memoize
  private async getProvingInputs() {
    const children = await this.provingReadyPromise.promise;
    return new RootParityInputs(children);
  }

  private isReadyForProving(): this is RootParityCircuit & {
    children: Tuple<RootParityInput<typeof RECURSIVE_PROOF_LENGTH>, typeof NUM_BASE_PARITY_PER_ROOT_PARITY>;
  } {
    return this.isReadyForSimulation() && this.children.every(child => child && 'proof' in child);
  }

  private isReadyForSimulation(): this is RootParityCircuit & {
    children: Tuple<{ publicInputs: ParityPublicInputs }, typeof NUM_BASE_PARITY_PER_ROOT_PARITY>;
  } {
    return this.children.filter(value => !!value).length === NUM_BASE_PARITY_PER_ROOT_PARITY;
  }

  @memoize
  public async simulate(): Promise<ParityPublicInputs> {
    return this.context.simulator.simulate({
      type: ProvingRequestType.ROOT_PARITY,
      inputs: await this.getSimulationInputs(),
    });
  }

  @memoize
  public async prove(): Promise<RootParityInput<typeof NESTED_RECURSIVE_PROOF_LENGTH>> {
    const result = await this.context.prover.prove({
      type: ProvingRequestType.ROOT_PARITY,
      inputs: await this.getProvingInputs(),
    });

    if (this.context.options.checkSimulationMatchesProof && !result.publicInputs.equals(await this.simulate())) {
      throw new Error(`Simulation output and proof public inputs do not match`);
    }

    return result;
  }
}

class MergeRollupProvingJob {
  static load(merge: any, context: OrchestratorContext): Promise<MergeRollupProvingJob> {
    throw new Error('Method not implemented.');
  }

  constructor(public readonly level: number, public readonly index: number) {}
  public setNested(
    input:
      | { simulation: BaseOrMergeRollupPublicInputs }
      | { proof: PublicInputsAndRecursiveProof<BaseOrMergeRollupPublicInputs> },
    isLeft: boolean,
  ) {}

  @memoize
  public simulate(): Promise<BaseOrMergeRollupPublicInputs> {
    throw new Error('Unimplemented');
    // await buildInput
    // run base rollup sim
    // save output to db
  }

  // returns output and proof
  @memoize
  public prove(): Promise<PublicInputsAndRecursiveProof<BaseOrMergeRollupPublicInputs>> {
    throw new Error('Unimplemented');
    // await buildInput
    // run prover
    // validate output?
    // store proof in db?
    // return proof!
  }
}

// abstract class ParentCircuit<
//   TState,
//   TIndex,
//   TSimulationReadyState,
//   TProofReadyState,
//   TSimulationInputs,
//   TProvingInputs,
// > {
//   private simulationReadyPromise = promiseWithResolvers<TSimulationReadyState>();
//   private provingReadyPromise = promiseWithResolvers<TProofReadyState>();

//   protected state: TState;

//   protected abstract makeSimulationInputs(): Promise<TSimulationInputs>;

//   protected abstract makeProvingInputs(): Promise<TProvingInputs>;

//   protected abstract isReadyForSimulation(): this is TSimulationReadyState;

//   protected abstract isReadyForProving(): this is TProofReadyState;
// }

type BlockRootReadyForSimulation = BlockRootCircuit & {
  leftMerge: RequiredBy<BlockRootCircuit['leftMerge'], 'inputs'>;
  rightMerge: RequiredBy<BlockRootCircuit['rightMerge'], 'inputs'>;
  rootParity: RequiredBy<BlockRootCircuit['rootParity'], 'inputs' | 'state'>;
};

type BlockRootReadyForProving = BlockRootCircuit & {
  leftMerge: Required<BlockRootCircuit['leftMerge']>;
  rightMerge: Required<BlockRootCircuit['rightMerge']>;
  rootParity: Required<BlockRootCircuit['rootParity']>;
};

class BlockRootCircuit implements Circuit<ProvingRequestType.BLOCK_ROOT_ROLLUP> {
  constructor(private context: OrchestratorContext) {}

  public leftMerge: Partial<PublicInputsAndRecursiveProof<BaseOrMergeRollupPublicInputs>> = {};
  public rightMerge: Partial<PublicInputsAndRecursiveProof<BaseOrMergeRollupPublicInputs>> = {};

  public rootParity: {
    inputs?: ParityPublicInputs;
    proof?: RootParityInput<typeof NESTED_RECURSIVE_PROOF_LENGTH>;
    state?: ParityState;
  } = {};

  public setNested(
    input:
      | { simulation: BaseOrMergeRollupPublicInputs }
      | { proof: PublicInputsAndRecursiveProof<BaseOrMergeRollupPublicInputs> },
    isLeft: boolean,
  ) {
    const accessor = isLeft ? ('leftMerge' as const) : ('rightMerge' as const);
    if ('simulation' in input) {
      this[accessor].inputs = input.simulation;
    } else {
      this[accessor] = input.proof;
    }
  }

  public setRootParity(
    input: { state: ParityState } | { simulation: ParityPublicInputs } | { proof: RootParityInput<439> },
  ) {
    if ('state' in input) {
      this.rootParity.state = input.state;
    } else if ('simulation' in input) {
      this.rootParity.inputs = input.simulation;
    } else {
      this.rootParity.proof = input.proof;
    }
  }

  private isReadyForSimulation(): this is BlockRootReadyForSimulation {
    return Boolean(this.leftMerge.inputs && this.rightMerge.inputs && this.rootParity.state && this.rootParity.inputs);
  }

  private isReadyForProving(): this is BlockRootReadyForProving {
    return Boolean(this.leftMerge.proof && this.rightMerge.proof && this.rootParity.state && this.rootParity.proof);
  }

  @memoize
  private async buildCommonInputs() {
    const db = this.context.db;

    const startArchiveSnapshot = await getTreeSnapshot(MerkleTreeId.ARCHIVE, db);
    const newArchiveSiblingPath = await getRootTreeSiblingPath(MerkleTreeId.ARCHIVE, db);

    return { startArchiveSnapshot, newArchiveSiblingPath };
  }

  @memoize
  private async buildSimulationInputs() {
    if (!this.isReadyForSimulation()) {
      throw new Error(`Block root not ready for simulation.`);
    }

    const {
      inputs: rootParityInput,
      state: {
        l1ToL2Messages: newL1ToL2Messages,
        messageTreeSnapshot: startL1ToL2MessageTreeSnapshot,
        newL1ToL2MessageTreeRootSiblingPath,
      },
    } = this.rootParity;

    const { startArchiveSnapshot, newArchiveSiblingPath } = await this.buildCommonInputs();

    const previousRollupData: BlockRootRollupInputs['previousRollupData'] = mapTuple(
      [this.leftMerge, this.rightMerge],
      ({ inputs }) => PreviousRollupData.withEmptyProof(inputs),
    );

    const l1ToL2Roots = RootParityInput.withEmptyProof(rootParityInput, NESTED_RECURSIVE_PROOF_LENGTH);

    return BlockRootRollupInputs.from({
      previousRollupData,
      l1ToL2Roots,
      newL1ToL2Messages,
      newL1ToL2MessageTreeRootSiblingPath,
      startL1ToL2MessageTreeSnapshot,
      startArchiveSnapshot,
      newArchiveSiblingPath,
      // TODO(#7346): Inject previous block hash (required when integrating batch rollup circuits)
      previousBlockHash: Fr.ZERO,
      proverId: this.context.proverId,
    });
  }

  @memoize
  private async buildProvingInputs() {
    if (!this.isReadyForProving()) {
      throw new Error(`Block root not ready for proof.`);
    }

    const {
      inputs: rootParityInput,
      proof: rootParityProof,
      state: {
        l1ToL2Messages: newL1ToL2Messages,
        messageTreeSnapshot: startL1ToL2MessageTreeSnapshot,
        newL1ToL2MessageTreeRootSiblingPath,
      },
    } = this.rootParity;

    const { startArchiveSnapshot, newArchiveSiblingPath } = await this.buildCommonInputs();

    const previousRollupData: BlockRootRollupInputs['previousRollupData'] = mapTuple(
      [this.leftMerge, this.rightMerge],
      ({ inputs, proof, verificationKey }) =>
        new PreviousRollupData(inputs, proof, verificationKey.keyAsFields, getVKMembershipWitness(verificationKey)),
    );

    const l1ToL2Roots = new RootParityInput(
      rootParityProof.proof,
      rootParityProof.verificationKey,
      rootParityProof.vkPath,
      rootParityInput,
    );

    return BlockRootRollupInputs.from({
      previousRollupData,
      l1ToL2Roots,
      newL1ToL2Messages,
      newL1ToL2MessageTreeRootSiblingPath,
      startL1ToL2MessageTreeSnapshot,
      startArchiveSnapshot,
      newArchiveSiblingPath,
      // TODO(#7346): Inject previous block hash (required when integrating batch rollup circuits)
      previousBlockHash: Fr.ZERO,
      proverId: this.context.proverId,
    });
  }

  @memoize
  public async simulate(): Promise<BlockRootOrBlockMergePublicInputs> {
    const inputs = await this.buildSimulationInputs();
    return this.context.simulator.simulate({ type: ProvingRequestType.BLOCK_ROOT_ROLLUP, inputs });
  }

  @memoize
  public async prove(): Promise<PublicInputsAndRecursiveProof<BlockRootOrBlockMergePublicInputs>> {
    const inputs = await this.buildProvingInputs();
    const result = await this.context.prover.prove({ type: ProvingRequestType.BLOCK_ROOT_ROLLUP, inputs });

    if (this.context.options.checkSimulationMatchesProof && !result.inputs.equals(await this.simulate())) {
      throw new Error(`Simulation output and proof public inputs do not match`);
    }

    return result;
  }
}

class BlockMergeCircuit implements Circuit<ProvingRequestType.BLOCK_MERGE_ROLLUP> {
  public leftMerge: Partial<PublicInputsAndRecursiveProof<BlockRootOrBlockMergePublicInputs>> = {};
  public rightMerge: Partial<PublicInputsAndRecursiveProof<BlockRootOrBlockMergePublicInputs>> = {};

  constructor(
    public readonly level: number,
    public readonly index: number,
    private readonly context: OrchestratorContext,
  ) {}

  public setNested(
    input:
      | { simulation: BlockRootOrBlockMergePublicInputs }
      | { proof: PublicInputsAndRecursiveProof<BlockRootOrBlockMergePublicInputs> },
    isLeft: boolean,
  ) {
    const accessor = isLeft ? ('leftMerge' as const) : ('rightMerge' as const);
    if ('simulation' in input) {
      this[accessor].inputs = input.simulation;
    } else {
      this[accessor] = input.proof;
    }

    // TODO: kick off promises! here and in all similar ones!!!!!! CONTINUE HERE
  }

  private isReadyForSimulation(): this is BlockMergeCircuit & {
    leftMerge: RequiredBy<BlockMergeCircuit['leftMerge'], 'inputs'>;
    rightMerge: RequiredBy<BlockMergeCircuit['rightMerge'], 'inputs'>;
  } {
    return Boolean(this.leftMerge.inputs && this.rightMerge.inputs);
  }

  private isReadyForProving(): this is BlockMergeCircuit & {
    leftMerge: Required<BlockMergeCircuit['leftMerge']>;
    rightMerge: Required<BlockMergeCircuit['rightMerge']>;
  } {
    return Boolean(this.leftMerge.proof && this.rightMerge.proof);
  }

  @memoize
  private buildSimulationInputs(): BlockMergeRollupInputs {
    if (!this.isReadyForSimulation()) {
      throw new Error(`Block merge not ready for simulation.`);
    }

    return new BlockMergeRollupInputs(
      mapTuple([this.leftMerge, this.rightMerge], merge => PreviousRollupBlockData.withEmptyProof(merge.inputs)),
    );
  }

  @memoize
  private buildProvingInputs(): BlockMergeRollupInputs {
    if (!this.isReadyForProving()) {
      throw new Error(`Block merge not ready for proving.`);
    }

    return new BlockMergeRollupInputs(
      mapTuple(
        [this.leftMerge, this.rightMerge],
        ({ inputs, proof, verificationKey }) =>
          new PreviousRollupBlockData(
            inputs,
            proof,
            verificationKey.keyAsFields,
            getVKMembershipWitness(verificationKey),
          ),
      ),
    );
  }

  @memoize
  public simulate(): Promise<BlockRootOrBlockMergePublicInputs> {
    const inputs = this.buildSimulationInputs();
    return this.context.simulator.simulate({ type: ProvingRequestType.BLOCK_MERGE_ROLLUP, inputs });
  }

  @memoize
  public async prove(): Promise<PublicInputsAndRecursiveProof<BlockRootOrBlockMergePublicInputs>> {
    const inputs = this.buildProvingInputs();
    const result = await this.context.prover.prove({ type: ProvingRequestType.BLOCK_MERGE_ROLLUP, inputs });

    if (this.context.options.checkSimulationMatchesProof && !result.inputs.equals(await this.simulate())) {
      throw new Error(`Simulation output and proof public inputs do not match`);
    }
    return result;
  }
}

class RootRollupCircuit implements Circuit<ProvingRequestType.ROOT_ROLLUP> {
  public leftMerge: Partial<PublicInputsAndRecursiveProof<BlockRootOrBlockMergePublicInputs>> = {};
  public rightMerge: Partial<PublicInputsAndRecursiveProof<BlockRootOrBlockMergePublicInputs>> = {};

  constructor(private context: OrchestratorContext) {}

  public setNested(
    input:
      | { simulation: BlockRootOrBlockMergePublicInputs }
      | { proof: PublicInputsAndRecursiveProof<BlockRootOrBlockMergePublicInputs> },
    isLeft: boolean,
  ) {
    const accessor = isLeft ? ('leftMerge' as const) : ('rightMerge' as const);
    if ('simulation' in input) {
      this[accessor].inputs = input.simulation;
    } else {
      this[accessor] = input.proof;
    }
  }

  private isReadyForSimulation(): this is RootRollupCircuit & {
    leftMerge: RequiredBy<RootRollupCircuit['leftMerge'], 'inputs'>;
    rightMerge: RequiredBy<RootRollupCircuit['rightMerge'], 'inputs'>;
  } {
    return Boolean(this.leftMerge.inputs && this.rightMerge.inputs);
  }

  private isReadyForProving(): this is RootRollupCircuit & {
    leftMerge: Required<RootRollupCircuit['leftMerge']>;
    rightMerge: Required<RootRollupCircuit['rightMerge']>;
  } {
    return Boolean(this.leftMerge.proof && this.rightMerge.proof);
  }

  @memoize
  private buildSimulationInputs(): RootRollupInputs {
    if (!this.isReadyForSimulation()) {
      throw new Error(`Block merge not ready for simulation.`);
    }

    return new RootRollupInputs(
      mapTuple([this.leftMerge, this.rightMerge], merge => PreviousRollupBlockData.withEmptyProof(merge.inputs)),
      this.context.proverId,
    );
  }

  @memoize
  private buildProvingInputs(): RootRollupInputs {
    if (!this.isReadyForProving()) {
      throw new Error(`Block merge not ready for proving.`);
    }

    return new RootRollupInputs(
      mapTuple(
        [this.leftMerge, this.rightMerge],
        ({ inputs, proof, verificationKey }) =>
          new PreviousRollupBlockData(
            inputs,
            proof,
            verificationKey.keyAsFields,
            getVKMembershipWitness(verificationKey),
          ),
      ),
      this.context.proverId,
    );
  }

  @memoize
  public simulate(): Promise<RootRollupPublicInputs> {
    const inputs = this.buildSimulationInputs();
    return this.context.simulator.simulate({ type: ProvingRequestType.ROOT_ROLLUP, inputs });
  }

  @memoize
  public async prove(): Promise<PublicInputsAndRecursiveProof<RootRollupPublicInputs>> {
    const inputs = this.buildProvingInputs();
    const result = await this.context.prover.prove({ type: ProvingRequestType.ROOT_ROLLUP, inputs });

    if (this.context.options.checkSimulationMatchesProof && !result.inputs.equals(await this.simulate())) {
      throw new Error(`Simulation output and proof public inputs do not match`);
    }
    return result;
  }
}

function memoize<This extends object, Result>(fn: () => Result, context: ClassMethodDecoratorContext) {
  return function (this: This) {
    const key = `__${String(context.name)}_value`;
    const thisWithKey = this as { [key: string]: Result };
    if (!(key in this)) {
      const result = fn.call(this);
      thisWithKey[key] = result;
    }
    return thisWithKey[key];
  };
}
