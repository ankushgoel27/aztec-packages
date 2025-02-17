import { type L2BlockSource } from '@aztec/circuit-types';
import { createDebugLogger } from '@aztec/foundation/log';
import { type AztecKVStore } from '@aztec/kv-store';
import { type DataStoreConfig, createStore } from '@aztec/kv-store/utils';
import { type TelemetryClient } from '@aztec/telemetry-client';
import { NoopTelemetryClient } from '@aztec/telemetry-client/noop';

import { type AttestationPool } from '../attestation_pool/attestation_pool.js';
import { P2PClient } from '../client/p2p_client.js';
import { type P2PConfig } from '../config.js';
import { DiscV5Service } from '../service/discV5_service.js';
import { DummyP2PService } from '../service/dummy_service.js';
import { LibP2PService, createLibP2PPeerId } from '../service/index.js';
import { AztecKVTxPool, type TxPool } from '../tx_pool/index.js';
import { getPublicIp, resolveAddressIfNecessary, splitAddressPort } from '../util.js';

export * from './p2p_client.js';

export const createP2PClient = async (
  config: P2PConfig & DataStoreConfig,
  attestationsPool: AttestationPool,
  l2BlockSource: L2BlockSource,
  telemetry: TelemetryClient = new NoopTelemetryClient(),
  deps: { txPool?: TxPool; store?: AztecKVStore } = {},
) => {
  const store = deps.store ?? (await createStore('p2p', config, createDebugLogger('aztec:p2p:lmdb')));
  const txPool = deps.txPool ?? new AztecKVTxPool(store, telemetry);

  let p2pService;

  if (config.p2pEnabled) {
    // If announceTcpAddress or announceUdpAddress are not provided, query for public IP if config allows

    const {
      tcpAnnounceAddress: configTcpAnnounceAddress,
      udpAnnounceAddress: configUdpAnnounceAddress,
      queryForIp,
    } = config;

    config.tcpAnnounceAddress = configTcpAnnounceAddress
      ? await resolveAddressIfNecessary(configTcpAnnounceAddress)
      : undefined;
    config.udpAnnounceAddress = configUdpAnnounceAddress
      ? await resolveAddressIfNecessary(configUdpAnnounceAddress)
      : undefined;

    // create variable for re-use if needed
    let publicIp;

    // check if no announce IP was provided
    const splitTcpAnnounceAddress = splitAddressPort(configTcpAnnounceAddress || '', true);
    if (splitTcpAnnounceAddress.length == 2 && splitTcpAnnounceAddress[0] === '') {
      if (queryForIp) {
        publicIp = await getPublicIp();
        const tcpAnnounceAddress = `${publicIp}:${splitTcpAnnounceAddress[1]}`;
        config.tcpAnnounceAddress = tcpAnnounceAddress;
      } else {
        throw new Error(
          `Invalid announceTcpAddress provided: ${configTcpAnnounceAddress}. Expected format: <addr>:<port>`,
        );
      }
    }

    const splitUdpAnnounceAddress = splitAddressPort(configUdpAnnounceAddress || '', true);
    if (splitUdpAnnounceAddress.length == 2 && splitUdpAnnounceAddress[0] === '') {
      // If announceUdpAddress is not provided, use announceTcpAddress
      if (!queryForIp && config.tcpAnnounceAddress) {
        config.udpAnnounceAddress = config.tcpAnnounceAddress;
      } else if (queryForIp) {
        const udpPublicIp = publicIp || (await getPublicIp());
        const udpAnnounceAddress = `${udpPublicIp}:${splitUdpAnnounceAddress[1]}`;
        config.udpAnnounceAddress = udpAnnounceAddress;
      }
    }

    // Create peer discovery service
    const peerId = await createLibP2PPeerId(config.peerIdPrivateKey);
    const discoveryService = new DiscV5Service(peerId, config);

    p2pService = await LibP2PService.new(config, discoveryService, peerId, txPool, attestationsPool, store);
  } else {
    p2pService = new DummyP2PService();
  }
  return new P2PClient(store, l2BlockSource, txPool, attestationsPool, p2pService, config.keepProvenTxsInPoolFor);
};
