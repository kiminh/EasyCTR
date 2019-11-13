#ifndef ASSEMBLER_DATA_TRANSFORM_H_
#define ASSEMBLER_DATA_TRANSFORM_H_

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/env.h"

#include "assembler/column.h"
#include "assembler/conf_parser.h"
#include "assembler/feature.h"

namespace assembler {

class MetaManager {
 public:
  explicit MetaManager(::tensorflow::Env* env);

  void Init(const std::string& meta_path);
  const std::map<std::string, Field>& meta() const { return meta_; }

 private:
  MetaManager(const MetaManager&) = delete;
  void operator=(const MetaManager&) = delete;

  ::tensorflow::Env* env_;
  std::map<std::string, Field> meta_;
};

class DataTransform {
 public:
  explicit DataTransform(::tensorflow::OpKernelConstruction* ctx);

  void Init(const ConfParser& cp);
  size_t feature_size() const { return feature_size_; }
  const std::vector<std::pair<std::shared_ptr<BasicColumn>, Field>>& columns()
      const {
    return columns_;
  }
  Example Transform(const std::string& input) const;
  void ServingTransform(const std::string& user_feature,
                        const std::vector<std::string>& ctx_features,
                        const std::vector<std::string>& item_features,
                        std::vector<Feature>* features) const;
  void Serialize(xJson::Value *s) const;
  void ReConstruct(const xJson::Value &root);

 private:
  DataTransform(const DataTransform&) = delete;
  void operator=(const DataTransform&) = delete;

  ::tensorflow::OpKernelConstruction* ctx_;
  std::vector<std::pair<std::shared_ptr<BasicColumn>, Field>> columns_;
  size_t feature_size_ = 0;
};

}  // namespace assembler

#endif  // ASSEMBLER_DATA_TRANSFORM_H_
