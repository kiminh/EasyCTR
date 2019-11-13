
#include "assembler/monitor_feature.h"

#include <string>
#include <unordered_map>

#include "deps/attr/Attr_API.h"

namespace assembler {

///////////////////////////////////////////////////////////////////////////////
// NOTO: 小心改动，不要再执行生成代码的操作量，因为已经申请了 attr id
// Generate code using monitor_code_generator.py
// python monitor_code_generator.py > monitor_code_generator.template
#include "assembler/monitor_code_generator.template"
///////////////////////////////////////////////////////////////////////////////

void training_monitor_feature(int idx, int pos1, int pos2) {
  std::string func = std::string("training_monitor_func_") +
                     std::to_string(idx) + '_' + std::to_string(pos1) + '_' +
                     std::to_string(pos2);
  auto it = training_monitor_func_map.find(func);
  if (it == training_monitor_func_map.end()) {
    Attr_API(34506114, 1);  // training_monitor_func缺失
  } else {
    (*it).second();
  }
}

void serving_monitor_feature(int idx, int pos1, int pos2) {
  std::string func = std::string("serving_monitor_func_") +
                     std::to_string(idx) + '_' + std::to_string(pos1) + '_' +
                     std::to_string(pos2);
  auto it = serving_monitor_func_map.find(func);
  if (it == serving_monitor_func_map.end()) {
    Attr_API(34506115, 1);  // serving_monitor_func缺失
  } else {
    (*it).second();
  }
}

}  // namespace assembler
