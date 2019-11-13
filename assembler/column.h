#ifndef ASSEMBLER_COLUMN_H_
#define ASSEMBLER_COLUMN_H_

#include <cmath>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "assembler/feature.h"
#include "deps/jsoncpp/json/json.h"

namespace assembler {

class BasicColumn {
 public:
  BasicColumn(::tensorflow::OpKernelConstruction* ctx, const std::string& iname,
              Field::Type type, double def, int width)
      : ctx_(ctx), iname_(iname), type_(type), def_(def), width_(width) {}
  virtual ~BasicColumn() {}

  virtual double ToValue(const std::string&, bool*) const = 0;
  virtual std::vector<double> ToListValue(const std::string&, bool*) const = 0;
  virtual xJson::Value ToJson() const = 0;
  virtual void Serialize(xJson::Value* bc) const;

  static std::shared_ptr<BasicColumn> Parse(
      ::tensorflow::OpKernelConstruction* ctx, const xJson::Value& bc);

  ::tensorflow::OpKernelConstruction* ctx() { return ctx_; }
  const std::string& iname() const { return iname_; }
  Field::Type type() const { return type_; }
  double def() const { return def_; }
  int width() const { return width_; }

  virtual void PrintDebugInfo() const;

 private:
  BasicColumn(const BasicColumn&) = delete;
  void operator=(const BasicColumn&) = delete;

  ::tensorflow::OpKernelConstruction* ctx_;
  std::string iname_;
  Field::Type type_;
  double def_;
  int width_;
};

class NumericColumn : public BasicColumn {
 public:
  NumericColumn(::tensorflow::OpKernelConstruction* ctx,
                const std::string& iname, Field::Type type, double def)
      : BasicColumn(ctx, iname, type, def, 1) {}
  NumericColumn(const NumericColumn&) = default;
  NumericColumn& operator=(const NumericColumn&) = default;
  double ToValue(const std::string& key, bool*) const override;
  std::vector<double> ToListValue(const std::string&, bool*) const override {
    throw std::logic_error("NumericColumn is not list type.");
  }

  xJson::Value ToJson() const override;
  void Serialize(xJson::Value* bc) const override;

  void PrintDebugInfo() const override;
};

class StringColumn : public BasicColumn {
 public:
  StringColumn(::tensorflow::OpKernelConstruction* ctx,
               const std::string& iname, double def, int min_count, int top_k,
               int oov_buckets)
      : BasicColumn(ctx, iname, Field::Type::STRING, def, 1),
        min_count_(min_count),
        top_k_(top_k),
        oov_buckets_(oov_buckets) {}
  StringColumn(const StringColumn&) = default;
  StringColumn& operator=(const StringColumn&) = default;

  double ToValue(const std::string&, bool*) const override;
  std::vector<double> ToListValue(const std::string&, bool*) const override {
    throw std::logic_error("StringColumn is not list type.");
  }
  xJson::Value ToJson() const override;
  void Serialize(xJson::Value* bc) const override;
  void PrintDebugInfo() const override;

  int min_count() const { return min_count_; }
  int top_k() const { return top_k_; }
  int oov_buckets() const { return oov_buckets_; }
  const std::unordered_map<std::string, int>& indexer() const {
    return indexer_;
  }
  bool LoadFromDictFile(const std::string&);
  void ReConstruct(const xJson::Value& bc);

 private:
  std::unordered_map<std::string, int> indexer_;
  std::vector<std::string> keys_;
  int min_count_;
  int top_k_;
  int oov_buckets_;
};

class StringListColumn : public BasicColumn {
 public:
  StringListColumn(::tensorflow::OpKernelConstruction* ctx,
                   const std::string& iname, double def, int min_count,
                   int top_k, int width)
      : BasicColumn(ctx, iname, Field::Type::STRING_LIST, def, width),
        min_count_(min_count),
        top_k_(top_k) {}
  StringListColumn(const StringListColumn&) = default;
  StringListColumn& operator=(const StringListColumn&) = default;

  double ToValue(const std::string&, bool*) const override {
    throw std::logic_error("StringListColumn is a list type.");
  }
  std::vector<double> ToListValue(const std::string&, bool*) const override;
  xJson::Value ToJson() const override;
  void Serialize(xJson::Value* bc) const override;
  void PrintDebugInfo() const override;

  int min_count() const { return min_count_; }
  int top_k() const { return top_k_; }
  const std::unordered_map<std::string, int>& indexer() const {
    return indexer_;
  }
  bool LoadFromDictFile(const std::string&);
  void ReConstruct(const xJson::Value& bc);

 private:
  std::unordered_map<std::string, int> indexer_;
  std::vector<std::string> keys_;
  int min_count_;
  int top_k_;
};

class FloatListColumn : public BasicColumn {
 public:
  FloatListColumn(::tensorflow::OpKernelConstruction* ctx,
                  const std::string& iname, double def, int width)
      : BasicColumn(ctx, iname, Field::Type::FLOAT_LIST, def, width) {}
  FloatListColumn(const FloatListColumn&) = default;
  FloatListColumn& operator=(const FloatListColumn&) = default;

  double ToValue(const std::string&, bool*) const override {
    throw std::logic_error("FloatListColumn is a list type.");
  }
  std::vector<double> ToListValue(const std::string&, bool*) const override;

  xJson::Value ToJson() const override;
  void Serialize(xJson::Value* bc) const override;
  void PrintDebugInfo() const override;

 private:
};

class WeightedStringListColumn : public BasicColumn {
 public:
  WeightedStringListColumn(::tensorflow::OpKernelConstruction* ctx,
                           const std::string& iname, double def, int min_count,
                           int top_k, int w, double min_weight)
      : BasicColumn(ctx, iname, Field::Type::WEIGHTED_STRING_LIST, def, w * 2),
        min_count_(min_count),
        top_k_(top_k),
        min_weight_(min_weight) {}
  WeightedStringListColumn(const WeightedStringListColumn&) = default;
  WeightedStringListColumn& operator=(const WeightedStringListColumn&) =
      default;

  double ToValue(const std::string&, bool*) const override {
    throw std::logic_error("WeightedStringListColumn is a list type.");
  }
  // 前 width 个元素为 id 值，后 width 个元素为 weight 值
  std::vector<double> ToListValue(const std::string&, bool*) const override;

  xJson::Value ToJson() const override;
  void Serialize(xJson::Value* bc) const override;
  void PrintDebugInfo() const override;

  int min_count() const { return min_count_; }
  int top_k() const { return top_k_; }
  double min_weight() const { return min_weight_; }
  const std::unordered_map<std::string, int>& indexer() const {
    return indexer_;
  }
  bool LoadFromDictFile(const std::string&);
  void ReConstruct(const xJson::Value& bc);

 private:
  std::unordered_map<std::string, int> indexer_;
  std::vector<std::string> keys_;
  int min_count_;
  int top_k_;
  double min_weight_;
};

}  // namespace assembler

#endif  // ASSEMBLER_COLUMN_H_
