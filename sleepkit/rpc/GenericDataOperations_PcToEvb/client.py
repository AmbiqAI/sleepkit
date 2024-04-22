#
# Generated by erpcgen 1.9.1 on Thu Sep 15 09:19:22 2022.
#
# AUTOGENERATED - DO NOT EDIT
#

from .. import erpc

from . import common, interface

# import callbacks declaration from other groups
# from ..GenericDataOperations_EvbToPc import interface as interface_EvbToPc


# Client for pc_to_evb
class pc_to_evbClient(interface.Ipc_to_evb):
    def __init__(self, manager):
        super(pc_to_evbClient, self).__init__()
        self._clientManager = manager

    def ns_rpc_data_sendBlockToEVB(self, block):
        # Build remote function invocation message.
        request = self._clientManager.create_request()
        codec = request.codec
        codec.start_write_message(
            erpc.codec.MessageInfo(
                type=erpc.codec.MessageType.kInvocationMessage,
                service=self.SERVICE_ID,
                request=self.NS_RPC_DATA_SENDBLOCKTOEVB_ID,
                sequence=request.sequence,
            )
        )
        if block is None:
            raise ValueError("block is None")
        block._write(codec)

        # Send request and process reply.
        self._clientManager.perform_request(request)
        _result = codec.read_uint32()
        return _result

    def ns_rpc_data_fetchBlockFromEVB(self, block):
        assert type(block) is erpc.Reference, "out parameter must be a Reference object"

        # Build remote function invocation message.
        request = self._clientManager.create_request()
        codec = request.codec
        codec.start_write_message(
            erpc.codec.MessageInfo(
                type=erpc.codec.MessageType.kInvocationMessage,
                service=self.SERVICE_ID,
                request=self.NS_RPC_DATA_FETCHBLOCKFROMEVB_ID,
                sequence=request.sequence,
            )
        )

        # Send request and process reply.
        self._clientManager.perform_request(request)
        block.value = common.dataBlock()._read(codec)
        _result = codec.read_uint32()
        return _result

    def ns_rpc_data_computeOnEVB(self, in_block, result_block):
        assert type(result_block) is erpc.Reference, "out parameter must be a Reference object"

        # Build remote function invocation message.
        request = self._clientManager.create_request()
        codec = request.codec
        codec.start_write_message(
            erpc.codec.MessageInfo(
                type=erpc.codec.MessageType.kInvocationMessage,
                service=self.SERVICE_ID,
                request=self.NS_RPC_DATA_COMPUTEONEVB_ID,
                sequence=request.sequence,
            )
        )
        if in_block is None:
            raise ValueError("in_block is None")
        in_block._write(codec)

        # Send request and process reply.
        self._clientManager.perform_request(request)
        result_block.value = common.dataBlock()._read(codec)
        _result = codec.read_uint32()
        return _result
