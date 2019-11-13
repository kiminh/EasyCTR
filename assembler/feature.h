#ifndef ASSEMBLER_FEATURE_H_
#define ASSEMBLER_FEATURE_H_

#include <string>
#include <vector>
#include <iostream>

#include "deps/jsoncpp/json/json.h"
#include "assembler/jsoncpp_helper.hpp"
#include "assembler/utils.h"

namespace assembler {

using ::utils::jsoncpp_helper::checkJsonArgs;
using ::utils::jsoncpp_helper::int32_tag;
using ::utils::jsoncpp_helper::int64_tag;

struct Field {
  enum Type {
    INT = 0,
    FLOAT,
    STRING,
    STRING_LIST,
    FLOAT_LIST,
    WEIGHTED_STRING_LIST,
    TYPE_END,
  } type;
  enum CID { USER = 0, ITEM, CTX, EXTRA, CID_END } cid;
  size_t pos1;
  size_t pos2;

  void Serialize(xJson::Value *f) const {
    (*f)["type"] = type;
    (*f)["cid"] = cid;
    (*f)["pos1"] = pos1;
    (*f)["pos2"] = pos2;
  }
  void Parse(const xJson::Value &f) {
    if (!checkJsonArgs(f, "type", int32_tag, "cid", int32_tag, "pos1",
                       int64_tag, "pos2", int64_tag)) {
      throw std::logic_error("parse  field error");
    }
    int t = f["type"].asInt();
    int c = f["cid"].asInt();
    type = static_cast<Field::Type>(t);
    cid = static_cast<Field::CID>(c);
    pos1 = f["pos1"].asInt64();
    pos2 = f["pos2"].asInt64();
  }

  void PrintDebugInfo() {
    std::cout << "type = " << type << std::endl;
    std::cout << "cid = " << cid << std::endl;
    std::cout << "pos1 = " << pos1 << std::endl;
    std::cout << "pos2 = " << pos2 << std::endl;
  }
};

typedef std::vector<double> Feature;
struct Example {
  Feature feature;
  double label;
  double weight;
};

}  // namespace assembler

#endif  // ASSEMBLER_FEATURE_H_
