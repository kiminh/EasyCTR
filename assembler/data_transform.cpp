#include "assembler/data_transform.h"

#include <exception>
#include <fstream>
#include <iostream>
#include <string>

#include "tensorflow/core/platform/file_system.h"

#include "assembler/jsoncpp_helper.hpp"
#include "assembler/monitor_feature.h"
#include "assembler/utils.h"
#include "assembler/file_reader.h"
#include "deps/attr/Attr_API.h"

namespace assembler {

using ::utils::jsoncpp_helper::checkJsonArgs;
using ::utils::jsoncpp_helper::int32_tag;
using ::utils::jsoncpp_helper::array_tag;
using ::utils::jsoncpp_helper::object_tag;

MetaManager::MetaManager(::tensorflow::Env* env) : env_(env) {}

void MetaManager::Init(const std::string& meta_path) {
  FileReader file_reader(env_, meta_path);
  if (!file_reader.Init()) {
    std::cerr << "Open file " << meta_path << " failed." << std::endl;
    throw std::logic_error("open meta file failed");
  }

  std::string line;
  int lineindex = 0;
  std::string token;
  while (file_reader.ReadLine(&line)) {
    ++lineindex;
    if (line.empty()) {
      continue;
    }
    if (line[0] == '#') {
      continue;
    }
    auto tokens = utils::Split(line, " \t");
    if (tokens.size() != 5) {
      std::cerr << "meta file format error[1] in line " << lineindex
                << std::endl;
      throw std::logic_error("meta file format error");
    }
    token = tokens[2].ToString();
    Field::Type type;
    if (token == "int") {
      type = Field::Type::INT;
    } else if (token == "string") {
      type = Field::Type::STRING;
    } else if (token == "float") {
      type = Field::Type::FLOAT;
    } else if (token == "string_list") {
      type = Field::Type::STRING_LIST;
    } else if (token == "float_list") {
      type = Field::Type::FLOAT_LIST;
    } else if (token == "weighted_string_list") {
      type = Field::Type::WEIGHTED_STRING_LIST;
    } else {
      std::cerr << "meta file format error[2] in line " << lineindex
                << std::endl;
      throw std::logic_error("meta file format error");
    }
    size_t pos1 = -1, pos2 = -1;
    try {
      pos1 = std::stoi(tokens[3].ToString());
      pos2 = std::stoi(tokens[4].ToString());
    } catch (const std::exception& e) {
      std::cerr << "meta file format error[3] in line " << lineindex
                << std::endl;
      throw std::logic_error("meta file format error");
    }
    token = tokens[1].ToString();
    if (meta_.count(token) > 0) {
      std::cerr << "meta file format error[4] in line " << lineindex
                << std::endl;
      throw std::logic_error("meta file format error");
    }

    token = tokens[0].ToString();
    Field::CID cid;
    if (token.substr(0, 4) == "user") {
      cid = Field::CID::USER;
    } else if (token.substr(0, 4) == "item") {
      cid = Field::CID::ITEM;
    } else if (token.substr(0, 3) == "ctx") {
      cid = Field::CID::CTX;
    } else if (token.substr(0, 5) == "extra") {
      cid = Field::CID::EXTRA;
    } else {
      std::cerr << "meta error: unknow cid = " << token << std::endl;
      throw std::logic_error("meta error: unknow cid");
    }
    token = tokens[1].ToString();
    meta_[token] = {type, cid, pos1, pos2};
  }
}

DataTransform::DataTransform(::tensorflow::OpKernelConstruction* ctx)
    : ctx_(ctx) {}

void DataTransform::Init(const ConfParser& cp) {
  MetaManager meta(ctx_->env());
  meta.Init(cp.data_paths().meta_file);
  for (auto& c : cp.columns()) {
    feature_size_ += c->width();
    if (meta.meta().count(c->iname()) == 0) {
      std::cerr << "column " << c->iname() << "not in meta." << std::endl;
      throw std::logic_error("column not in meta");
    }
    Field field = meta.meta().at(c->iname());
    if (field.type != c->type()) {
      std::string types[] = {"int",        "float",
                             "string",     "string_list",
                             "float_list", "weighted_string_list"};
      std::cerr << "column '" << c->iname() << "' type not matched with meta, "
                << "meta type = " << types[field.type]
                << ", conf type = " << types[c->type()] << std::endl;
      throw std::logic_error("column type not match with meta");
    }
    columns_.push_back({c, field});
  }
}

// training, throw exception when error
Example DataTransform::Transform(const std::string& input) const {
  Attr_API(34506112, 1);  // training特征转换请求
  Example example;
  std::vector<StringPiece> tokens = utils::Split(input, "|");
  if (tokens.size() < 5) {
    std::cerr << "error: input tokens size less than 5. input = " << input
              << std::endl;
    throw std::logic_error("input tokens size less than 5");
  } else if (tokens.size() > 6) {
    std::cerr << "error: input tokens size larger than 6. input = " << input
              << std::endl;
    throw std::logic_error("input tokens size larger than 6");
  }
  example.label = std::stof(tokens[0].ToString());
  example.weight = std::stof(tokens[1].ToString());

  std::vector<std::vector<StringPiece>> user_pieces;
  std::vector<std::vector<StringPiece>> item_pieces;
  std::vector<std::vector<StringPiece>> ctx_pieces;
  utils::TotalSplit(tokens[2], &user_pieces);
  utils::TotalSplit(tokens[3], &item_pieces);
  utils::TotalSplit(tokens[4], &ctx_pieces);
  std::vector<std::vector<StringPiece>>* p[3] = {&user_pieces, &item_pieces,
                                                 &ctx_pieces};
  std::string token;
  bool use_def = false;
  for (auto& c : columns_) {
    int idx = c.second.cid;
    size_t pos1 = c.second.pos1;
    size_t pos2 = c.second.pos2;
    if (pos1 >= p[idx]->size() || pos2 >= (*p[idx])[pos1].size()) {
      std::cerr << "tokens size error, input = " << input << std::endl;
      throw std::logic_error("tokens size error");
    }
    token = (*p[idx])[pos1][pos2].ToString();
    if (c.second.type == Field::Type::STRING_LIST ||
        c.second.type == Field::Type::FLOAT_LIST ||
        c.second.type == Field::Type::WEIGHTED_STRING_LIST) {
      std::vector<double> v = c.first->ToListValue(token, &use_def);
      example.feature.insert(example.feature.end(), v.begin(), v.end());
    } else {
      double v = c.first->ToValue(token, &use_def);
      example.feature.push_back(v);
    }
    if (use_def) {
      training_monitor_feature(idx, pos1, pos2);
    }
  }
  return example;
}

// serving, use default value when error
void DataTransform::ServingTransform(
    const std::string& user_feature,
    const std::vector<std::string>& ctx_features,
    const std::vector<std::string>& item_features,
    std::vector<Feature>* features) const {
  Attr_API(34506113, 1);  // serving特征转换请求
  if (user_feature == "") {
    Attr_API(34467197, 1);  // user特征为空
  }
  std::vector<std::vector<StringPiece>> user_pieces;
  std::vector<std::vector<StringPiece>> item_pieces;
  std::vector<std::vector<StringPiece>> ctx_pieces;
  utils::TotalSplit(user_feature, &user_pieces);
  std::vector<std::vector<StringPiece>>* p[3] = {&user_pieces, &item_pieces,
                                                 &ctx_pieces};
  std::string token;
  for (size_t i = 0; i < item_features.size(); ++i) {
    Attr_API(34459407, columns_.size());  // serving column转换特征总数
    item_pieces.clear();
    ctx_pieces.clear();
    if (item_features[i] == "") {
      Attr_API(34467198, 1);  // item特征为空
    } else {
      utils::TotalSplit(item_features[i], &item_pieces);
    }
    if (item_features.size() == ctx_features.size()) {
      if (ctx_features[i] == "") {
        Attr_API(34467199, 1);  // ctx特征为空
      } else {
        utils::TotalSplit(ctx_features[i], &ctx_pieces);
      }
    } else {
      Attr_API(34466077, 1);  // ctx特征与item特征个数不一致
    }
    Feature feature;
    bool use_def = false;
    for (auto& c : columns_) {
      int idx = c.second.cid;
      size_t pos1 = c.second.pos1;
      size_t pos2 = c.second.pos2;
      if (pos1 >= p[idx]->size() || pos2 >= (*p[idx])[pos1].size()) {
        // 上报排除空特征
        if (p[idx]->size() > 0) {
          Attr_API(34457373, 1);  // serving特征格式错误
        }
        token = "";
      } else {
        token = (*p[idx])[pos1][pos2].ToString();
      }

      if (c.second.type == Field::Type::STRING_LIST ||
          c.second.type == Field::Type::FLOAT_LIST ||
          c.second.type == Field::Type::WEIGHTED_STRING_LIST) {
        std::vector<double> v = c.first->ToListValue(token, &use_def);
        feature.insert(feature.end(), v.begin(), v.end());
      } else {
        double v = c.first->ToValue(token, &use_def);
        feature.push_back(v);
      }
      if (use_def) {
        serving_monitor_feature(idx, pos1, pos2);
      }
    }
    features->push_back(feature);
    Attr_API(34459489, feature.size());  // serving column特征总数(list特征展开)
  }
}

void DataTransform::Serialize(xJson::Value* s) const {
  (*s)["feature_size"] = feature_size_;
  xJson::Value& cols = (*s)["columns"];
  cols = xJson::arrayValue;
  for (const auto& c : columns_) {
    xJson::Value bc;
    xJson::Value f;
    c.first->Serialize(&bc);  // basic column
    c.second.Serialize(&f);   // field
    xJson::Value col;
    col["basic_column"] = bc;
    col["field"] = f;
    cols.append(col);
  }
}
void DataTransform::ReConstruct(const xJson::Value& root) {
  if (!checkJsonArgs(root, "columns", array_tag, "feature_size", int32_tag)) {
    throw std::logic_error(
        "DataTransform::ReConstruct: member 'columns' or 'feature_size' error");
  }
  feature_size_ = root["feature_size"].asInt();

  const xJson::Value& cols = root["columns"];
  for (xJson::ArrayIndex i = 0; i < cols.size(); ++i) {
    const xJson::Value& col = cols[i];
    if (!checkJsonArgs(col, "basic_column", object_tag, "field", object_tag)) {
      throw std::logic_error(
          "DataTransform::ReConstruct error, member 'basic_column' or 'field' "
          "not exist or not object");
    }
    auto column = BasicColumn::Parse(ctx_, col["basic_column"]);
    Field field;
    field.Parse(col["field"]);
    columns_.push_back({column, field});
  }
}

}  // namespace assembler
