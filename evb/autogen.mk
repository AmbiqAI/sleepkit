# Autogenerated file - created by NeuralSPOT make nest
INCLUDES +=  neuralspot/ns-core/includes-api neuralspot/ns-harness/includes-api neuralspot/ns-peripherals/includes-api neuralspot/ns-ipc/includes-api neuralspot/ns-audio/includes-api neuralspot/ns-utils/includes-api neuralspot/ns-i2c/includes-api neuralspot/ns-nnsp/includes-api neuralspot/ns-usb/includes-api neuralspot/ns-usb/includes-api neuralspot/ns-rpc/includes-api extern/AmbiqSuite/R4.4.1/boards/apollo4p_evb/bsp extern/AmbiqSuite/R4.4.1/CMSIS/ARM/Include extern/AmbiqSuite/R4.4.1/CMSIS/AmbiqMicro/Include extern/AmbiqSuite/R4.4.1/devices extern/AmbiqSuite/R4.4.1/mcu/apollo4p extern/AmbiqSuite/R4.4.1/mcu/apollo4p/hal/mcu extern/AmbiqSuite/R4.4.1/utils extern/AmbiqSuite/R4.4.1/third_party/FreeRTOSv10.5.1/Source/include extern/AmbiqSuite/R4.4.1/third_party/FreeRTOSv10.5.1/Source/portable/GCC/AMapollo4 extern/AmbiqSuite/R4.4.1/third_party/tinyusb/src extern/AmbiqSuite/R4.4.1/third_party/tinyusb/source/src extern/AmbiqSuite/R4.4.1/third_party/tinyusb/source/src/common extern/AmbiqSuite/R4.4.1/third_party/tinyusb/source/src/osal extern/AmbiqSuite/R4.4.1/third_party/tinyusb/source/src/class/cdc extern/AmbiqSuite/R4.4.1/third_party/tinyusb/source/src/device extern/CMSIS/CMSIS-DSP-1.15.0/Include extern/CMSIS/CMSIS-DSP-1.15.0/PrivateInclude extern/tensorflow/0264234_Nov_15_2023/. extern/tensorflow/0264234_Nov_15_2023/third_party extern/tensorflow/0264234_Nov_15_2023/third_party/flatbuffers/include extern/tensorflow/0264234_Nov_15_2023/third_party/gemmlowp extern/SEGGER_RTT/R7.70a/RTT extern/SEGGER_RTT/R7.70a/Config extern/codecs/opus-precomp/includes-api extern/erpc/R1.9.1/includes-api
libraries += libs/ns-core.a libs/ns-harness.a libs/ns-peripherals.a libs/ns-ipc.a libs/ns-audio.a libs/ns-utils.a libs/ns-i2c.a libs/ns-nnsp.a libs/ns-usb.a libs/ns-rpc.a libs/ambiqsuite.a libs/segger_rtt.a libs/codecs.a libs/erpc.a libs/libam_hal.a libs/libam_bsp.a libs/libCMSISDSP-m4-gcc.a libs/libtensorflow-microlite-cm4-gcc-release.a libs/libopus.a
override_libraries += libs/ns-usb-overrides.a
local_app_name := main
