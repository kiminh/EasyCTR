#include "assembler/assembler.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>

#include "assembler/jsoncpp_helper.hpp"
#include "assembler/utils.h"

namespace assembler {

Assembler::Assembler(::tensorflow::OpKernelConstruction* ctx)
    : ctx_(ctx), transform_(ctx) {}

void Assembler::Init(const std::string& conf_path, bool load_dict, bool debug) {
  ConfParser cp(ctx_);
  if (!cp.Parse(conf_path, load_dict)) {
    std::cerr << "Parse conf error." << std::endl;
    throw std::logic_error("Parse conf error");
  }
  transform_.Init(cp);
  if (debug) {
    cp.PrintDebugInfo();
  }
}

Example Assembler::GetExample(const std::string& input) const {
  return transform_.Transform(input);
}

// user, item and context feature scheme
xJson::Value Assembler::GetFeatureScheme() const {
  xJson::Value obj;
  xJson::Value& feature_columns = obj["feature_columns"];
  feature_columns = xJson::arrayValue;
  for (const auto& c : transform_.columns()) {
    feature_columns.append(c.first->ToJson());
  }
  return obj;
}

void Assembler::GetServingInputs(const std::string& user_feature,
                                 const std::vector<std::string>& ctx_features,
                                 const std::vector<std::string>& item_features,
                                 std::vector<Feature>* features) const {
  transform_.ServingTransform(user_feature, ctx_features, item_features,
                              features);
}

void Assembler::Serialize(std::string* serialized) const {
  xJson::Value s;
  transform_.Serialize(&s);
  *serialized = s.toStyledString();
}

bool Assembler::ParseFromString(const std::string& input) {
  std::istringstream iss(input);
  xJson::Value root;
  xJson::CharReaderBuilder rbuilder;
  std::string err;
  bool parse_ok = xJson::parseFromStream(rbuilder, iss, &root, &err);
  if (!parse_ok) {
    std::cerr << err << std::endl;
    return false;
  }
  try {
    transform_.ReConstruct(root);
  } catch (const std::exception& e) {
    std::cout << "parse from string catch except: " << e.what() << std::endl;
    return false;
  }
  return true;
}

void Assembler::PrintDebugInfo() const {
  std::cout << " ---- Debug, Print Assembler info ----" << std::endl;
  std::cout << "feature_columns size = " << transform_.columns().size()
            << std::endl;
  for (auto c : transform_.columns()) {
    std::cout << "###### column info" << std::endl;
    c.first->PrintDebugInfo();
    c.second.PrintDebugInfo();
    std::cout << "######" << std::endl;
  }
}

void Assembler::PrintExample(const Example& e) const {
  std::cout << "------- Example -------" << std::endl;
  std::cout << "label = " << e.label << std::endl;
  std::cout << "weight = " << e.weight << std::endl;
  size_t idx = 0;
  for (auto c : transform_.columns()) {
    std::cout << "iname = " << c.first->iname() << ", ";
    std::cout << "type = " << c.first->type() << ", ";
    std::cout << "width = " << c.first->width() << ", feature = ";
    for (int i = 0; i < c.first->width(); ++i) {
      if (idx >= e.feature.size()) {
        std::cout << "Feature transform error!!!" << std::endl;
        break;
      }
      std::cout << e.feature[idx] << " ";
      ++idx;
    }
    std::cout << std::endl;
  }
  std::cout << "----------------------" << std::endl;
  std::cout << std::endl;
}

}  // namespace assembler
