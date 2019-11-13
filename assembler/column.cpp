#include "assembler/column.h"

#include <algorithm>
#include <iostream>
#include <string>

#include "assembler/jsoncpp_helper.hpp"
#include "assembler/utils.h"
#include "deps/attr/Attr_API.h"
#include "deps/jsoncpp/json/json.h"

namespace assembler {

using ::utils::jsoncpp_helper::checkJsonArgs;
using ::utils::jsoncpp_helper::string_tag;
using ::utils::jsoncpp_helper::int32_tag;
using ::utils::jsoncpp_helper::double_tag;
using ::utils::jsoncpp_helper::array_tag;

void BasicColumn::Serialize(xJson::Value* bc) const {
  (*bc)["iname"] = iname();
  (*bc)["type"] = type();
  (*bc)["def"] = def();
  (*bc)["width"] = width();
}

std::shared_ptr<BasicColumn> BasicColumn::Parse(
    ::tensorflow::OpKernelConstruction* ctx, const xJson::Value& bc) {
  if (!checkJsonArgs(bc, "iname", string_tag, "type", int32_tag, "def",
                     double_tag, "width", int32_tag)) {
    throw std::logic_error("parse basic column error");
  }

  std::shared_ptr<BasicColumn> c;
  std::string iname = bc["iname"].asString();
  int type = bc["type"].asInt();
  double def = bc["def"].asDouble();
  int width = bc["width"].asInt();
  if (type == Field::Type::INT) {
    c.reset(new NumericColumn(ctx, iname, Field::Type::INT, def));
  } else if (type == Field::Type::FLOAT) {
    c.reset(new NumericColumn(ctx, iname, Field::Type::FLOAT, def));
  } else if (type == Field::Type::STRING) {
    StringColumn* sc = new StringColumn(ctx, iname, def, 0, -1, -1);
    sc->ReConstruct(bc);
    c.reset(sc);
  } else if (type == Field::Type::STRING_LIST) {
    StringListColumn* sc = new StringListColumn(ctx, iname, def, 0, -1, width);
    sc->ReConstruct(bc);
    c.reset(sc);
  } else if (type == Field::Type::WEIGHTED_STRING_LIST) {
    WeightedStringListColumn* sc =
        new WeightedStringListColumn(ctx, iname, def, 0, -1, width / 2, 0);
    sc->ReConstruct(bc);
    c.reset(sc);
  } else {
    std::cerr << "Parse serialized error, unknow type." << std::endl;
    throw std::logic_error("Parse serialized error, unknow type.");
  }
  return c;
}

void BasicColumn::PrintDebugInfo() const {
  std::cout << "iname = '" << iname() << "'" << std::endl;
  std::cout << "type = " << type() << std::endl;
  std::cout << "default = " << def() << std::endl;
  std::cout << "width = " << width() << std::endl;
}

double NumericColumn::ToValue(const std::string& key, bool* use_def) const {
  Attr_API(34457365, 1);  // numeric特征总数
  *use_def = false;
  double v = def();
  if (key == "") {
    Attr_API(34457366, 1);  // numeric特征为空
    *use_def = true;
    return v;
  }
  try {
    v = std::stof(key);
    if (std::isnan(v) || std::isinf(v)) {
      Attr_API(34457367, 1);  // numeric特征数值不合法
      v = def();
      *use_def = true;
    }
  } catch (const std::exception& e) {
    Attr_API(34457368, 1);  // numeric特征异常
    *use_def = true;
  }
  return v;
}

xJson::Value NumericColumn::ToJson() const {
  xJson::Value obj;

  obj["name"] = iname();
  obj["width"] = width();
  Field::Type t = type();
  if (t == Field::Type::INT) {
    obj["type"] = "int";
  } else if (t == Field::Type::FLOAT) {
    obj["type"] = "float";
  }
  return obj;
}

void NumericColumn::Serialize(xJson::Value* bc) const {
  BasicColumn::Serialize(bc);
}

void NumericColumn::PrintDebugInfo() const { BasicColumn::PrintDebugInfo(); }

bool StringColumn::LoadFromDictFile(const std::string& dict_file) {
  return utils::LoadFromDictFile(ctx()->env(), dict_file, min_count_, top_k_,
                                 &indexer_, &keys_);
}

double StringColumn::ToValue(const std::string& key, bool* use_def) const {
  Attr_API(34457369, 1);  // string特征总数
  *use_def = false;
  double v = def();
  auto it = indexer_.find(key);
  if (it != indexer_.end()) {
    v = static_cast<double>(it->second);
  } else {
    Attr_API(34457370, 1);  // string特征不在词典中
    if (oov_buckets_ > 0) {
      Attr_API(34488045, 1);  // hash oov特征
      v = utils::MurMurHash3(key) % oov_buckets_;
    } else {
      *use_def = true;
    }
  }
  return v;
}

xJson::Value StringColumn::ToJson() const {
  xJson::Value obj;

  obj["name"] = iname();
  obj["width"] = width();
  obj["type"] = "string";
  obj["vocab"] = xJson::arrayValue;
  obj["size"] = keys_.size() + oov_buckets_;

  for (const auto& key : keys_) {
    obj["vocab"].append(key);
  }

  return obj;
}

void StringColumn::Serialize(xJson::Value* bc) const {
  BasicColumn::Serialize(bc);
  (*bc)["min_count"] = min_count_;
  (*bc)["top_k"] = top_k_;
  (*bc)["oov_buckets"] = oov_buckets_;
  xJson::Value& keys = (*bc)["keys"];
  keys = xJson::arrayValue;
  for (size_t i = 0; i < keys_.size(); ++i) {
    keys.append(keys_[i]);
  }
}

void StringColumn::ReConstruct(const xJson::Value& bc) {
  if (!checkJsonArgs(bc, "min_count", int32_tag, "top_k", int32_tag,
                     "oov_buckets", int32_tag, "keys", array_tag)) {
    throw std::logic_error("ReConstruct error");
  }
  keys_.clear();
  indexer_.clear();
  min_count_ = bc["min_count"].asInt();
  top_k_ = bc["top_k"].asInt();
  oov_buckets_ = bc["oov_buckets"].asInt();
  const xJson::Value& keys = bc["keys"];
  for (xJson::ArrayIndex i = 0; i < keys.size(); ++i) {
    if (!keys[i].isString()) {
      throw std::logic_error("ReConstruct error");
    }
    std::string key = keys[i].asString();
    keys_.push_back(key);
    indexer_[key] = i;
  }
}

void StringColumn::PrintDebugInfo() const {
  BasicColumn::PrintDebugInfo();
  std::cout << "string column min_count = " << min_count_ << std::endl;
  std::cout << "string column top_k = " << top_k_ << std::endl;
  std::cout << "string column oov size = " << oov_buckets_ << std::endl;
  std::cout << "string column dict size = " << keys_.size() << std::endl;
  for (size_t i = 0; i < std::min<size_t>(10, keys_.size()); ++i) {
    std::cout << "#" << i << ": " << keys_[i] << std::endl;
  }
}

bool StringListColumn::LoadFromDictFile(const std::string& dict_file) {
  return utils::LoadFromDictFile(ctx()->env(), dict_file, min_count_, top_k_,
                                 &indexer_, &keys_);
}

// 优先取在词典中的值，不在词典的值放在最后
std::vector<double> StringListColumn::ToListValue(const std::string& key,
                                                  bool* use_def) const {
  *use_def = true;  // 当list全部为默认值时才为true
  std::vector<double> values;
  auto tokens = utils::Split(key, ",");
  int w = std::min(static_cast<int>(tokens.size()), width());
  Attr_API(34457371, w);  // string_list特征总数
  for (int i = 0; i < w; ++i) {
    auto it = indexer_.find(tokens[i].ToString());
    if (it != indexer_.end()) {
      values.push_back(it->second);
      *use_def = false;
    } else {
      Attr_API(34457372, 1);  // string_list特征不在词典中
    }
  }
  for (size_t i = values.size(); i < (size_t)width(); ++i) {
    values.push_back(def());
  }

  return values;
}

xJson::Value StringListColumn::ToJson() const {
  xJson::Value obj;

  obj["name"] = iname();
  obj["width"] = width();
  obj["type"] = "string_list";
  obj["vocab"] = xJson::arrayValue;
  obj["size"] = keys_.size();

  for (const auto& key : keys_) {
    obj["vocab"].append(key);
  }

  return obj;
}

void StringListColumn::Serialize(xJson::Value* bc) const {
  BasicColumn::Serialize(bc);
  (*bc)["min_count"] = min_count_;
  (*bc)["top_k"] = top_k_;
  xJson::Value& keys = (*bc)["keys"];
  keys = xJson::arrayValue;
  for (size_t i = 0; i < keys_.size(); ++i) {
    keys.append(keys_[i]);
  }
}

void StringListColumn::ReConstruct(const xJson::Value& bc) {
  if (!checkJsonArgs(bc, "min_count", int32_tag, "top_k", int32_tag, "keys",
                     array_tag)) {
    throw std::logic_error("ReConstruct error");
  }
  keys_.clear();
  indexer_.clear();
  min_count_ = bc["min_count"].asInt();
  top_k_ = bc["top_k"].asInt();
  const xJson::Value& keys = bc["keys"];
  for (xJson::ArrayIndex i = 0; i < keys.size(); ++i) {
    if (!keys[i].isString()) {
      throw std::logic_error("ReConstruct error");
    }
    std::string key = keys[i].asString();
    keys_.push_back(key);
    indexer_[key] = i;
  }
}

void StringListColumn::PrintDebugInfo() const {
  BasicColumn::PrintDebugInfo();
  std::cout << "string_list column min_count = " << min_count_ << std::endl;
  std::cout << "string_list column top_k = " << top_k_ << std::endl;
  std::cout << "string_list column dict size = " << keys_.size() << std::endl;
  for (size_t i = 0; i < std::min<size_t>(10, keys_.size()); ++i) {
    std::cout << "#" << i << ": " << keys_[i] << std::endl;
  }
}

std::vector<double> FloatListColumn::ToListValue(const std::string& key,
                                                 bool* use_def) const {
  *use_def = true;  // 当list全部为默认值时才为true
  std::vector<double> values(width(), def());
  auto tokens = utils::Split(key, ",");
  int total = std::min<int>(width(), static_cast<int>(tokens.size()));
  Attr_API(34549322, total);  // float_list特征总数
  for (int i = 0; i < total; ++i) {
    if (tokens[i] == "") {
      Attr_API(34549323, 1);  // float_list特征为空
      continue;
    }
    try {
      values[i] = std::stof(tokens[i].ToString());
      if (std::isnan(values[i]) || std::isinf(values[i])) {
        Attr_API(34549324, 1);  // float_list特征数值不合法[数值异常]
        values[i] = def();
        continue;
      }
      *use_def = false;
    } catch (const std::exception& e) {
      Attr_API(34549325, 1);  // float_list特征数值不合法[转换异常]
    }
  }
  return values;
}

xJson::Value FloatListColumn::ToJson() const {
  xJson::Value obj;

  obj["name"] = iname();
  obj["width"] = width();
  obj["type"] = "float_list";

  return obj;
}

void FloatListColumn::Serialize(xJson::Value* bc) const {
  BasicColumn::Serialize(bc);
}

void FloatListColumn::PrintDebugInfo() const { BasicColumn::PrintDebugInfo(); }

bool WeightedStringListColumn::LoadFromDictFile(const std::string& dict_file) {
  return utils::LoadFromDictFile(ctx()->env(), dict_file, min_count_, top_k_,
                                 &indexer_, &keys_);
}

// 优先取在词典中的值，不在词典的值放在最后
std::vector<double> WeightedStringListColumn::ToListValue(
    const std::string& key, bool* use_def) const {
  *use_def = true;  // 当list全部为默认值时才为true
  std::vector<double> values(width(), def());
  // 权重必须默认为 0，当 -1 时，weight 必须为 0
  for (int i = width() / 2; i < width(); i++) {
    values[i] = 0;
  }

  auto tokens = utils::Split(key, ",");
  int w = std::min(static_cast<int>(tokens.size()), width() / 2);
  Attr_API(34593865, w);  // weighted_string_list特征总数

  int idx = 0;
  for (int i = 0; i < w; ++i) {
    auto subtokens = utils::Split(tokens[i], ":");
    if (subtokens.size() != 2) {
      Attr_API(34593866,
               1);  // weighted_string_list特征格式错误[subtokens大小问题]
      continue;
    }
    double weight = 0.0;
    try {
      weight = std::stof(subtokens[1].ToString());
    } catch (const std::exception& e) {
      Attr_API(34593867,
               1);  // weighted_string_list特征格式错误[浮点数转换问题]
      continue;
    }
    if (weight < min_weight_) {
      Attr_API(34593868, 1);  // weighted_string_list权重过滤总数
      continue;
    }
    auto it = indexer_.find(subtokens[0].ToString());
    if (it != indexer_.end()) {
      values[idx] = it->second;
      values[idx + width() / 2] = weight;
      ++idx;
      *use_def = false;
    } else {
      Attr_API(34593869, 1);  // weighted_string_list特征不在词典中
    }
  }
  return values;
}

xJson::Value WeightedStringListColumn::ToJson() const {
  xJson::Value obj;

  obj["name"] = iname();
  obj["width"] = width();
  obj["type"] = "weighted_string_list";
  obj["vocab"] = xJson::arrayValue;
  obj["size"] = keys_.size();

  for (const auto& key : keys_) {
    obj["vocab"].append(key);
  }

  return obj;
}

void WeightedStringListColumn::Serialize(xJson::Value* bc) const {
  BasicColumn::Serialize(bc);
  (*bc)["min_count"] = min_count_;
  (*bc)["top_k"] = top_k_;
  (*bc)["min_weight"] = min_weight_;
  xJson::Value& keys = (*bc)["keys"];
  keys = xJson::arrayValue;
  for (size_t i = 0; i < keys_.size(); ++i) {
    keys.append(keys_[i]);
  }
}

void WeightedStringListColumn::ReConstruct(const xJson::Value& bc) {
  if (!checkJsonArgs(bc, "min_count", int32_tag, "top_k", int32_tag,
                     "min_weight", double_tag, "keys", array_tag)) {
    throw std::logic_error("ReConstruct error");
  }

  keys_.clear();
  indexer_.clear();
  min_count_ = bc["min_count"].asInt();
  top_k_ = bc["top_k"].asInt();
  min_weight_ = bc["min_weight"].asDouble();
  const xJson::Value& keys = bc["keys"];
  for (xJson::ArrayIndex i = 0; i < keys.size(); ++i) {
    if (!keys[i].isString()) {
      throw std::logic_error("ReConstruct error");
    }
    std::string key = keys[i].asString();
    keys_.push_back(key);
    indexer_[key] = i;
  }
}

void WeightedStringListColumn::PrintDebugInfo() const {
  BasicColumn::PrintDebugInfo();
  std::cout << "weighted_string_list column min_count = " << min_count_
            << std::endl;
  std::cout << "weighted_string_list column top_k = " << top_k_ << std::endl;
  std::cout << "weighted_string_list column min_weight = " << min_weight_
            << std::endl;
  std::cout << "weighted_string_list column dict size = " << keys_.size()
            << std::endl;
  for (size_t i = 0; i < std::min<size_t>(10, keys_.size()); ++i) {
    std::cout << "#" << i << ": " << keys_[i] << std::endl;
  }
}

}  // namespace assembler
