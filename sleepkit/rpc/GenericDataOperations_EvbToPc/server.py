#
# Generated by erpcgen 1.9.1 on Thu Sep 15 09:19:22 2022.
#
# AUTOGENERATED - DO NOT EDIT
#

from .. import erpc

from . import common, interface


# Client for evb_to_pc
class evb_to_pcService(erpc.server.Service):
    def __init__(self, handler):
        super(evb_to_pcService, self).__init__(interface.Ievb_to_pc.SERVICE_ID)
        self._handler = handler
        self._methods = {
            interface.Ievb_to_pc.NS_RPC_DATA_SENDBLOCKTOPC_ID: self._handle_ns_rpc_data_sendBlockToPC,
            interface.Ievb_to_pc.NS_RPC_DATA_FETCHBLOCKFROMPC_ID: self._handle_ns_rpc_data_fetchBlockFromPC,
            interface.Ievb_to_pc.NS_RPC_DATA_COMPUTEONPC_ID: self._handle_ns_rpc_data_computeOnPC,
            interface.Ievb_to_pc.NS_RPC_DATA_REMOTEPRINTONPC_ID: self._handle_ns_rpc_data_remotePrintOnPC,
        }

    def _handle_ns_rpc_data_sendBlockToPC(self, sequence, codec):
        # Read incoming parameters.
        block = common.dataBlock()._read(codec)

        # Invoke user implementation of remote function.
        _result = self._handler.ns_rpc_data_sendBlockToPC(block)

        # Prepare codec for reply message.
        codec.reset()

        # Construct reply message.
        codec.start_write_message(
            erpc.codec.MessageInfo(
                type=erpc.codec.MessageType.kReplyMessage,
                service=interface.Ievb_to_pc.SERVICE_ID,
                request=interface.Ievb_to_pc.NS_RPC_DATA_SENDBLOCKTOPC_ID,
                sequence=sequence,
            )
        )
        codec.write_uint32(_result)

    def _handle_ns_rpc_data_fetchBlockFromPC(self, sequence, codec):
        # Create reference objects to pass into handler for out/inout parameters.
        block = erpc.Reference()

        # Read incoming parameters.

        # Invoke user implementation of remote function.
        _result = self._handler.ns_rpc_data_fetchBlockFromPC(block)

        # Prepare codec for reply message.
        codec.reset()

        # Construct reply message.
        codec.start_write_message(
            erpc.codec.MessageInfo(
                type=erpc.codec.MessageType.kReplyMessage,
                service=interface.Ievb_to_pc.SERVICE_ID,
                request=interface.Ievb_to_pc.NS_RPC_DATA_FETCHBLOCKFROMPC_ID,
                sequence=sequence,
            )
        )
        if block.value is None:
            raise ValueError("block.value is None")
        block.value._write(codec)
        codec.write_uint32(_result)

    def _handle_ns_rpc_data_computeOnPC(self, sequence, codec):
        # Create reference objects to pass into handler for out/inout parameters.
        result_block = erpc.Reference()

        # Read incoming parameters.
        in_block = common.dataBlock()._read(codec)

        # Invoke user implementation of remote function.
        _result = self._handler.ns_rpc_data_computeOnPC(in_block, result_block)

        # Prepare codec for reply message.
        codec.reset()

        # Construct reply message.
        codec.start_write_message(
            erpc.codec.MessageInfo(
                type=erpc.codec.MessageType.kReplyMessage,
                service=interface.Ievb_to_pc.SERVICE_ID,
                request=interface.Ievb_to_pc.NS_RPC_DATA_COMPUTEONPC_ID,
                sequence=sequence,
            )
        )
        if result_block.value is None:
            raise ValueError("result_block.value is None")
        result_block.value._write(codec)
        codec.write_uint32(_result)

    def _handle_ns_rpc_data_remotePrintOnPC(self, sequence, codec):
        # Read incoming parameters.
        msg = codec.read_string()

        # Invoke user implementation of remote function.
        _result = self._handler.ns_rpc_data_remotePrintOnPC(msg)

        # Prepare codec for reply message.
        codec.reset()

        # Construct reply message.
        codec.start_write_message(
            erpc.codec.MessageInfo(
                type=erpc.codec.MessageType.kReplyMessage,
                service=interface.Ievb_to_pc.SERVICE_ID,
                request=interface.Ievb_to_pc.NS_RPC_DATA_REMOTEPRINTONPC_ID,
                sequence=sequence,
            )
        )
        codec.write_uint32(_result)
