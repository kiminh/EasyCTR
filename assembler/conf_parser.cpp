#include "assembler/conf_parser.h"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "deps/jsoncpp/json/json.h"

#include "assembler/column.h"
#include "assembler/file_reader.h"

namespace assembler {

bool ConfParser::Parse(const std::string& conf_path, bool load_dict) {
  xJson::Value root;
  xJson::CharReaderBuilder rbuilder;
  rbuilder["collectComments"] = false;
  FileReader file_reader(ctx_->env(), conf_path);
  if (!file_reader.Init()) {
    std::cerr << "Open file " << conf_path << " failed." << std::endl;
    return false;
  }

  std::string conf;
  if (!file_reader.ReadAll(&conf)) {
    std::cerr << "Read file " << conf_path << " failed." << std::endl;
    return false;
  }

  std::istringstream iss(conf);
  std::string err;
  bool parse_ok = xJson::parseFromStream(rbuilder, iss, &root, &err);
  if (!parse_ok) {
    std::cerr << err << std::endl;
    return false;
  }

  if (!root.isMember("assembler")) {
    std::cerr << "No assembler field." << std::endl;
    return false;
  }

  if (!root["assembler"].isArray()) {
    std::cerr << "field assembler is not array." << std::endl;
    return false;
  }

  for (xJson::ArrayIndex idx = 0; idx < root["assembler"].size(); ++idx) {
    if (!root["assembler"][idx].isObject()) {
      std::cerr << "column is not object." << std::endl;
      return false;
    }
    const xJson::Value& json_column = root["assembler"][idx];
    if (!json_column.isMember("iname") || !json_column.isMember("type")) {
      std::cerr << "column lack of field 'iname' or or 'type', idx = " << idx
                << std::endl;
      return false;
    }

    if (!json_column["iname"].isString() || !json_column["type"].isString()) {
      std::cerr << "column fields 'iname' and 'type' should be string"
                << std::endl;
      return false;
    }

    std::shared_ptr<BasicColumn> column;
    bool create_column_ok = true;
    const std::string& ct = json_column["type"].asString();
    if (ct == "int") {
      create_column_ok =
          CreateNumericColumn(json_column, Field::Type::INT, &column);
    } else if (ct == "float") {
      create_column_ok =
          CreateNumericColumn(json_column, Field::Type::FLOAT, &column);
    } else if (ct == "string") {
      create_column_ok = CreateStringColumn(json_column, &column, load_dict);
    } else if (ct == "string_list") {
      create_column_ok =
          CreateStringListColumn(json_column, &column, load_dict);
    } else if (ct == "float_list") {
      create_column_ok = CreateFloatListColumn(json_column, &column);
    } else if (ct == "weighted_string_list") {
      create_column_ok =
          CreateWeightedStringListColumn(json_column, &column, load_dict);
    } else {
      std::cerr << "Unknown column::type '" << ct << "'" << std::endl;
      return false;
    }
    if (!create_column_ok) {
      return false;
    }
    columns_.push_back(column);
  }

  if (!root.isMember("input_data") || !root["input_data"].isObject()) {
    std::cerr << "field 'data' is not object." << std::endl;
    return false;
  }

  const xJson::Value& data = root["input_data"];
  if (data.isMember("meta_file")) {
    if (!data["meta_file"].isString()) {
      std::cerr << "field meta_file is not string." << std::endl;
      return false;
    }
    data_paths_.meta_file = data["meta_file"].asString();
  }

  return true;
}

bool ConfParser::CreateNumericColumn(const xJson::Value& obj, Field::Type type,
                                     std::shared_ptr<BasicColumn>* column) {
  const std::string& iname = obj["iname"].asString();
  if (!obj.isMember("default")) {
    std::cerr << "NumericColumn no field default." << std::endl;
    return false;
  }
  if (!obj["default"].isDouble()) {
    std::cerr << "NumericColumn field default is not double." << std::endl;
    return false;
  }
  double def = static_cast<double>(obj["default"].asDouble());
  (*column).reset(new NumericColumn(ctx_, iname, type, def));
  return true;
}

bool ConfParser::CreateStringColumn(const xJson::Value& obj,
                                    std::shared_ptr<BasicColumn>* column,
                                    bool load_dict) {
  const std::string& iname = obj["iname"].asString();
  if (!obj.isMember("min_count")) {
    std::cerr << "StringColumn has no field min_count." << std::endl;
    return false;
  }
  if (!obj["min_count"].isInt()) {
    std::cerr << "StringColumn field min_count is not int." << std::endl;
    return false;
  }
  if (!obj.isMember("top_k")) {
    std::cerr << "StringColumn has no field top_k." << std::endl;
    return false;
  }
  if (!obj["top_k"].isInt()) {
    std::cerr << "StringColumn field top_k is not int." << std::endl;
    return false;
  }
  if (!obj.isMember("oov_buckets")) {
    std::cerr << "StringColumn has no field oov_buckets." << std::endl;
    return false;
  }
  if (!obj["oov_buckets"].isInt()) {
    std::cerr << "StringColumn field oov_buckets is not int." << std::endl;
    return false;
  }

  int min_count = obj["min_count"].asInt();
  int top_k = obj["top_k"].asInt();
  int oov_buckets = obj["oov_buckets"].asInt();
  double def = -1;
  StringColumn* sc =
      new StringColumn(ctx_, iname, def, min_count, top_k, oov_buckets);
  if (!obj.isMember("dict_file")) {
    std::cerr << "StringColumn no field dict_file." << std::endl;
    return false;
  }
  if (!obj["dict_file"].isString()) {
    std::cerr << "StringColumn field dict_file is not string." << std::endl;
    return false;
  }
  const std::string& dict_file = obj["dict_file"].asString();
  if (load_dict && !sc->LoadFromDictFile(dict_file)) {
    std::cerr << "Load dict file '" << dict_file << "' error." << std::endl;
    return false;
  }
  (*column).reset(sc);
  return true;
}

bool ConfParser::CreateStringListColumn(const xJson::Value& obj,
                                        std::shared_ptr<BasicColumn>* column,
                                        bool load_dict) {
  const std::string& iname = obj["iname"].asString();
  if (!obj.isMember("min_count")) {
    std::cerr << "StringListColumn has no field min_count." << std::endl;
    return false;
  }
  if (!obj["min_count"].isInt()) {
    std::cerr << "StringListColumn field min_count is not int." << std::endl;
    return false;
  }
  int min_count = obj["min_count"].asInt();

  if (!obj.isMember("top_k")) {
    std::cerr << "StringListColumn has no field top_k." << std::endl;
    return false;
  }
  if (!obj["top_k"].isInt()) {
    std::cerr << "StringListColumn field top_k is not int." << std::endl;
    return false;
  }
  int top_k = obj["top_k"].asInt();

  if (!obj.isMember("width")) {
    std::cerr << "StringListColumn has no field width." << std::endl;
    return false;
  }
  if (!obj["width"].isInt()) {
    std::cerr << "StringListColumn field width is not int." << std::endl;
    return false;
  }
  int width = obj["width"].asInt();
  double def = -1;
  StringListColumn* sc =
      new StringListColumn(ctx_, iname, def, min_count, top_k, width);
  if (!obj.isMember("dict_file")) {
    std::cerr << "StringListColumn no field dict_file." << std::endl;
    return false;
  }
  if (!obj["dict_file"].isString()) {
    std::cerr << "StringListColumn field dict_file is not string." << std::endl;
    return false;
  }
  const std::string& dict_file = obj["dict_file"].asString();
  if (load_dict && !sc->LoadFromDictFile(dict_file)) {
    std::cerr << "Load dict file '" << dict_file << "' error." << std::endl;
    return false;
  }
  (*column).reset(sc);

  return true;
}

bool ConfParser::CreateFloatListColumn(const xJson::Value& obj,
                                       std::shared_ptr<BasicColumn>* column) {
  const std::string& iname = obj["iname"].asString();
  if (!obj.isMember("width")) {
    std::cerr << "FloatListColumn has no field width." << std::endl;
    return false;
  }
  if (!obj["width"].isInt()) {
    std::cerr << "FloatListColumn field width is not int." << std::endl;
    return false;
  }
  int width = obj["width"].asInt();
  double def = 0.0;
  FloatListColumn* sc = new FloatListColumn(ctx_, iname, def, width);
  (*column).reset(sc);

  return true;
}

void ConfParser::PrintDebugInfo() {
  std::cout << "--------------- ConfParser Info ---------------" << std::endl;
  std::cout << "#columns = " << columns_.size() << std::endl;
  std::vector<std::string> types = {"int",        "float",
                                    "string",     "string_list",
                                    "float_list", "weighted_string_list"};
  for (auto& c : columns_) {
    std::cout << "iname = " << c->iname() << ", type = " << types[c->type()]
              << ", def = " << c->def() << ", width = " << c->width()
              << std::endl;
  }
  std::cout << "meta_file = " << data_paths_.meta_file << std::endl;
  std::cout << "------------- ConfParser Info End -------------" << std::endl;
}

bool ConfParser::CreateWeightedStringListColumn(
    const xJson::Value& obj, std::shared_ptr<BasicColumn>* column,
    bool load_dict) {
  const std::string& iname = obj["iname"].asString();
  if (!obj.isMember("min_count")) {
    std::cerr << "WeightedStringListColumn has no field min_count."
              << std::endl;
    return false;
  }
  if (!obj["min_count"].isInt()) {
    std::cerr << "WeightedStringListColumn field min_count is not int."
              << std::endl;
    return false;
  }
  int min_count = obj["min_count"].asInt();

  if (!obj.isMember("top_k")) {
    std::cerr << "WeightedStringListColumn has no field top_k." << std::endl;
    return false;
  }
  if (!obj["top_k"].isInt()) {
    std::cerr << "WeightedStringListColumn field top_k is not int."
              << std::endl;
    return false;
  }
  int top_k = obj["top_k"].asInt();

  if (!obj.isMember("width")) {
    std::cerr << "WeightedStringListColumn has no field width." << std::endl;
    return false;
  }
  if (!obj["width"].isInt()) {
    std::cerr << "WeightedStringListColumn field width is not int."
              << std::endl;
    return false;
  }
  int width = obj["width"].asInt();

  if (!obj.isMember("min_weight")) {
    std::cerr << "WeightedStringListColumn has no field min_weight."
              << std::endl;
    return false;
  }
  if (!obj["min_weight"].isDouble()) {
    std::cerr << "WeightedStringListColumn field min_weight is not int."
              << std::endl;
    return false;
  }
  double min_weight = obj["min_weight"].asDouble();

  double def = -1;
  WeightedStringListColumn* sc = new WeightedStringListColumn(
      ctx_, iname, def, min_count, top_k, width, min_weight);
  if (!obj.isMember("dict_file")) {
    std::cerr << "WeightedStringListColumn no field dict_file." << std::endl;
    return false;
  }
  if (!obj["dict_file"].isString()) {
    std::cerr << "WeightedStringListColumn field dict_file is not string."
              << std::endl;
    return false;
  }
  const std::string& dict_file = obj["dict_file"].asString();
  if (load_dict && !sc->LoadFromDictFile(dict_file)) {
    std::cerr << "Load dict file '" << dict_file << "' error." << std::endl;
    return false;
  }
  (*column).reset(sc);

  return true;
}

}  // namespace assembler
