/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
#pragma once

#include "host/kernel_launch_binder.h"
#include "host_build_graph/kernel_graph_template.h"
#include "host_build_graph/kernel_graph_slot_wire.h"

namespace hbg {

// A prepared writable HostArgs buffer paired with the independently sealed slot.
// Storage is created outside launch. Its exclusive lease covers native HostArgs
// submission; the enclosing callable lease covers resolution and submission.
struct GraphKernelBinding {
    const GraphHostArgs &args;
    const GraphSlotRegistration &slot;

    static int validate(
        void *context, const KernelInvocationBinding &invocation, const KernelResourceBinding &resources
    ) noexcept {
        if (context == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
        const auto &self = *static_cast<const GraphKernelBinding *>(context);
        if (invocation.placeholder_count != 1 || !invocation.placeholders ||
            invocation.placeholders[0].address_offset != self.args.address_offset ||
            invocation.placeholders[0].data_offset != self.args.data_offset ||
            !valid_graph_slot_registration(self.slot) || resources.count != 5 || !resources.regions ||
            invocation.device_id != self.slot.device_id || invocation.context_generation != self.slot.slot_generation ||
            invocation.packet.data != reinterpret_cast<const uint8_t *>(self.args.storage.data()) ||
            invocation.packet.size != self.args.bytes || self.args.bytes > self.args.storage.size() * sizeof(uint64_t))
            return PTO_RUNTIME_ERR_INTERNAL;
        for (size_t i = 0; i < 5; ++i) {
            const auto &sealed = i == 4 ? self.slot.registry : self.slot.destinations[i];
            if (resources.regions[i].address != sealed.address || resources.regions[i].capacity != sealed.capacity)
                return PTO_RUNTIME_ERR_INVALID_STATE;
        }
        if (validate_graph_packet(self.args.storage.data(), self.args.bytes, GraphPacketAddress::HostTemplate) !=
            GraphPacketStatus::Ok)
            return PTO_RUNTIME_ERR_INTERNAL;
        GraphPacketHeader header{};
        std::memcpy(&header, invocation.packet.data + sizeof(SimplerKernelInvocationHeader), sizeof(header));
        if (header.device_id != self.slot.device_id || header.slot_generation != self.slot.slot_generation ||
            header.runtime_binary_id != self.slot.runtime_binary_id || self.args.bytes != self.slot.max_packet_bytes ||
            self.args.address_offset !=
                sizeof(SimplerKernelInvocationHeader) + offsetof(GraphPacketHeader, inline_payload_addr) ||
            self.args.data_offset != sizeof(SimplerKernelInvocationHeader) + header.payload_offset)
            return PTO_RUNTIME_ERR_INVALID_STATE;
        for (size_t i = 0; i < 4; ++i)
            if (header.destinations[i].address != self.slot.destinations[i].address ||
                header.destinations[i].capacity != self.slot.destinations[i].capacity)
                return PTO_RUNTIME_ERR_INVALID_STATE;
        return 0;
    }
};
}  // namespace hbg
