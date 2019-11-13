
#include "assembler/hash_state_manager.h"

namespace assembler {

HashStateManager::HashStateManager(int32_t bucket_size)
    : buckets_(bucket_size, -1), extra_(bucket_size) {}
}  // namespace assembler
