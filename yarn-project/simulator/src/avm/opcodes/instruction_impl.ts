import { type AvmContext } from '../avm_context.js';
import { type MemoryValue } from '../avm_memory_types.js';
import { OperandType } from '../serialization/instruction_serialization.js';
import { Addressing } from './addressing_mode.js';
import { Instruction } from './instruction.js';

/** Wire format that informs deserialization for instructions with two operands. */
export const TwoOperandWireFormat8 = [
  OperandType.UINT8,
  OperandType.UINT8,
  OperandType.UINT8,
  OperandType.UINT8,
  OperandType.UINT8,
];
export const TwoOperandWireFormat16 = [
  OperandType.UINT8,
  OperandType.UINT8,
  OperandType.UINT8,
  OperandType.UINT16,
  OperandType.UINT16,
];

/** Wire format that informs deserialization for instructions with three operands. */
export const ThreeOperandWireFormat8 = [
  OperandType.UINT8,
  OperandType.UINT8,
  OperandType.UINT8,
  OperandType.UINT8,
  OperandType.UINT8,
  OperandType.UINT8,
];
export const ThreeOperandWireFormat16 = [
  OperandType.UINT8,
  OperandType.UINT8,
  OperandType.UINT8,
  OperandType.UINT16,
  OperandType.UINT16,
  OperandType.UINT16,
];

/**
 * Covers (de)serialization for an instruction with:
 * indirect, inTag, and two operands.
 */
export abstract class TwoOperandInstruction extends Instruction {
  // Informs (de)serialization. See Instruction.deserialize.
  static readonly wireFormat8: OperandType[] = TwoOperandWireFormat8;
  static readonly wireFormat16: OperandType[] = TwoOperandWireFormat16;

  constructor(
    protected indirect: number,
    protected inTag: number,
    protected aOffset: number,
    protected dstOffset: number,
  ) {
    super();
  }
}

/**
 * Covers (de)serialization for an instruction with:
 * indirect, inTag, and three operands.
 */
export abstract class ThreeOperandInstruction extends Instruction {
  static readonly wireFormat8: OperandType[] = ThreeOperandWireFormat8;
  static readonly wireFormat16: OperandType[] = ThreeOperandWireFormat16;

  constructor(
    protected indirect: number,
    protected inTag: number,
    protected aOffset: number,
    protected bOffset: number,
    protected dstOffset: number,
  ) {
    super();
  }
}

export abstract class GetterInstruction extends Instruction {
  // Informs (de)serialization. See Instruction.deserialize.
  static readonly wireFormat: OperandType[] = [OperandType.UINT8, OperandType.UINT8, OperandType.UINT32];

  constructor(protected indirect: number, protected dstOffset: number) {
    super();
  }

  public async execute(context: AvmContext): Promise<void> {
    const memoryOperations = { writes: 1, indirect: this.indirect };
    const memory = context.machineState.memory.track(this.type);
    context.machineState.consumeGas(this.gasCost(memoryOperations));

    const [dstOffset] = Addressing.fromWire(this.indirect).resolve([this.dstOffset], memory);

    memory.set(dstOffset, this.getValue(context));

    memory.assert(memoryOperations);
    context.machineState.incrementPc();
  }

  protected abstract getValue(env: AvmContext): MemoryValue;
}
