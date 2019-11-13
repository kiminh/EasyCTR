#ifndef ASSEMBLER_HASH_STATE_MANAGER_H_
#define ASSEMBLER_HASH_STATE_MANAGER_H_

#include <cstdint>
#include <vector>

namespace assembler {

struct HashExtraInfo {
  uint32_t freq = 0;
  uint32_t ts = 0;
};

class HashStateManager {
 public:
  explicit HashStateManager(int32_t bucket_size);

 private:
  std::vector<int32_t> buckets_;
  std::vector<HashExtraInfo> extra_;
};

}  // namespace assembler

#endif  // ASSEMBLER_HASH_STATE_MANAGER_H_
