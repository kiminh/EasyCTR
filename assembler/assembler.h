#ifndef ASSEMBLER_ASSEMBLER_H_
#define ASSEMBLER_ASSEMBLER_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "deps/jsoncpp/json/json.h"

#include "assembler/column.h"
#include "assembler/conf_parser.h"
#include "assembler/data_transform.h"
#include "assembler/feature.h"

namespace assembler {

class Assembler {
 public:
  explicit Assembler(::tensorflow::OpKernelConstruction* ctx);

  // Train
  void Init(const std::string& conf_path, bool load_dict = true,
            bool debug = false);
  Example GetExample(const std::string& input) const;
  xJson::Value GetFeatureScheme() const;
  size_t feature_size() const { return transform_.feature_size(); }
  const std::vector<std::pair<std::shared_ptr<BasicColumn>, Field>>& columns()
      const {
    return transform_.columns();
  }

  // Serving
  void GetServingInputs(const std::string& user_feature,
                        const std::vector<std::string>& ctx_features,
                        const std::vector<std::string>& item_features,
                        std::vector<Feature>* features) const;
  void Serialize(std::string* output) const;
  bool ParseFromString(const std::string& input);
  void PrintDebugInfo() const;
  void PrintExample(const Example& e) const;

 private:
  Assembler(const Assembler&) = delete;
  void operator=(const Assembler&) = delete;
  ::tensorflow::OpKernelConstruction* ctx_;

  DataTransform transform_;
};

}  // namespace assembler

#endif  // ASSEMBLER_ASSEMBLER_H_
