#
# Generated by erpcgen 1.9.1 on Thu Sep 15 09:19:22 2022.
#
# AUTOGENERATED - DO NOT EDIT
#

from .. import erpc

from . import common, interface


# Client for pc_to_evb
class pc_to_evbService(erpc.server.Service):
    def __init__(self, handler):
        super(pc_to_evbService, self).__init__(interface.Ipc_to_evb.SERVICE_ID)
        self._handler = handler
        self._methods = {
            interface.Ipc_to_evb.NS_RPC_DATA_SENDBLOCKTOEVB_ID: self._handle_ns_rpc_data_sendBlockToEVB,
            interface.Ipc_to_evb.NS_RPC_DATA_FETCHBLOCKFROMEVB_ID: self._handle_ns_rpc_data_fetchBlockFromEVB,
            interface.Ipc_to_evb.NS_RPC_DATA_COMPUTEONEVB_ID: self._handle_ns_rpc_data_computeOnEVB,
        }

    def _handle_ns_rpc_data_sendBlockToEVB(self, sequence, codec):
        # Read incoming parameters.
        block = common.dataBlock()._read(codec)

        # Invoke user implementation of remote function.
        _result = self._handler.ns_rpc_data_sendBlockToEVB(block)

        # Prepare codec for reply message.
        codec.reset()

        # Construct reply message.
        codec.start_write_message(
            erpc.codec.MessageInfo(
                type=erpc.codec.MessageType.kReplyMessage,
                service=interface.Ipc_to_evb.SERVICE_ID,
                request=interface.Ipc_to_evb.NS_RPC_DATA_SENDBLOCKTOEVB_ID,
                sequence=sequence,
            )
        )
        codec.write_uint32(_result)

    def _handle_ns_rpc_data_fetchBlockFromEVB(self, sequence, codec):
        # Create reference objects to pass into handler for out/inout parameters.
        block = erpc.Reference()

        # Read incoming parameters.

        # Invoke user implementation of remote function.
        _result = self._handler.ns_rpc_data_fetchBlockFromEVB(block)

        # Prepare codec for reply message.
        codec.reset()

        # Construct reply message.
        codec.start_write_message(
            erpc.codec.MessageInfo(
                type=erpc.codec.MessageType.kReplyMessage,
                service=interface.Ipc_to_evb.SERVICE_ID,
                request=interface.Ipc_to_evb.NS_RPC_DATA_FETCHBLOCKFROMEVB_ID,
                sequence=sequence,
            )
        )
        if block.value is None:
            raise ValueError("block.value is None")
        block.value._write(codec)
        codec.write_uint32(_result)

    def _handle_ns_rpc_data_computeOnEVB(self, sequence, codec):
        # Create reference objects to pass into handler for out/inout parameters.
        result_block = erpc.Reference()

        # Read incoming parameters.
        in_block = common.dataBlock()._read(codec)

        # Invoke user implementation of remote function.
        _result = self._handler.ns_rpc_data_computeOnEVB(in_block, result_block)

        # Prepare codec for reply message.
        codec.reset()

        # Construct reply message.
        codec.start_write_message(
            erpc.codec.MessageInfo(
                type=erpc.codec.MessageType.kReplyMessage,
                service=interface.Ipc_to_evb.SERVICE_ID,
                request=interface.Ipc_to_evb.NS_RPC_DATA_COMPUTEONEVB_ID,
                sequence=sequence,
            )
        )
        if result_block.value is None:
            raise ValueError("result_block.value is None")
        result_block.value._write(codec)
        codec.write_uint32(_result)
