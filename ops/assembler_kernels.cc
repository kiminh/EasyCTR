#include <time.h>
#include <memory>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "assembler/assembler.h"
#include "assembler/utils.h"
#include "deps/attr/Attr_API.h"
#include "deps/jsoncpp/json/json.h"

namespace tensorflow {
class AssemblerOp : public OpKernel {
 public:
  explicit AssemblerOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), cnt_(0) {
    Attr_API(34459387, 1);  // AssemblerOp初始化次数
    LOG(INFO) << "\n\n";
    LOG(INFO) << " ------------------------ ";
    LOG(INFO) << "Init AssemblerOp ...";
    std::string conf_path;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("conf_path", &conf_path));
    assembler_.reset(new assembler::Assembler(ctx));
    assembler_->Init(conf_path, true, true);
    LOG(INFO) << "Init AssemblerOp OK.";
    LOG(INFO) << " ------------------------ ";
    LOG(INFO) << "\n\n";
  }

  void Compute(OpKernelContext* ctx) override {
    Attr_API(34459388, 1);  // AssemberOp总请求量
    auto flat_input = ctx->input(0).flat<std::string>();
    OP_REQUIRES(ctx, flat_input.size() == 1,
                errors::InvalidArgument("input size is not 1"));
    const std::string& input = flat_input(0);
    auto example = assembler_->GetExample(input);
    int dim = assembler_->feature_size();
    const int every_step = 10000000;
    if (cnt_ % every_step == 0) {
      mutex_lock l(mu_);
      if (cnt_ % every_step == 0) {
        assembler_->PrintExample(example);
      }
      ++cnt_;
    } else {
      ++cnt_;  // do not use mutex for performence
    }
    // Create output tensors
    Tensor* feature_tensor = nullptr;
    Tensor* label_tensor = nullptr;
    Tensor* weight_tensor = nullptr;
    TensorShape feature_shape;
    feature_shape.AddDim(dim);
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, feature_shape, &feature_tensor));

    TensorShape label_shape;
    label_shape.AddDim(1);
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, label_shape, &label_tensor));

    TensorShape weight_shape;
    weight_shape.AddDim(1);
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, weight_shape, &weight_tensor));

    // Fill output tensors
    auto flat_feature = feature_tensor->flat<double>();
    auto flat_label = label_tensor->flat<double>();
    auto flat_weight = weight_tensor->flat<double>();
    flat_label(0) = example.label;
    flat_weight(0) = example.weight;
    for (size_t fi = 0; fi < example.feature.size(); ++fi) {
      flat_feature(fi) = example.feature[fi];
    }
  }

 private:
  std::shared_ptr<assembler::Assembler> assembler_;
  size_t cnt_;

  mutex mu_;
};

REGISTER_KERNEL_BUILDER(Name("Assembler").Device(DEVICE_CPU), AssemblerOp);

class AssemblerSchemeOp : public OpKernel {
 public:
  explicit AssemblerSchemeOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    LOG(INFO) << "Init AssemblerSchemeOp ...";
    std::string conf_path;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("conf_path", &conf_path));
    assembler_.reset(new assembler::Assembler(ctx));
    assembler_->Init(conf_path);
    scheme_ = assembler_->GetFeatureScheme().toStyledString();
    LOG(INFO) << "Init AssemblerSchemeOp done";
  }

  void Compute(OpKernelContext* ctx) override {
    // Create output tensors
    Tensor* tensor = nullptr;
    TensorShape shape;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &tensor));
    auto scalar = tensor->scalar<std::string>();
    scalar(0) = scheme_;
  }

 private:
  std::shared_ptr<assembler::Assembler> assembler_;
  std::string scheme_;
};

REGISTER_KERNEL_BUILDER(Name("AssemblerScheme").Device(DEVICE_CPU),
                        AssemblerSchemeOp);

class AssemblerSerializeOp : public OpKernel {
 public:
  explicit AssemblerSerializeOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    LOG(INFO) << "\n\n";
    LOG(INFO) << "Init AssemblerSerializeOp ...";
    std::string conf_path;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("conf_path", &conf_path));
    assembler_.reset(new assembler::Assembler(ctx));
    assembler_->Init(conf_path);
    assembler_->Serialize(&serialized_);
    LOG(INFO) << "Init AssemblerSerializeOp done";
    LOG(INFO) << "\n\n";
  }

  void Compute(OpKernelContext* ctx) override {
    // Create output tensors
    Tensor* tensor = nullptr;
    TensorShape shape;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &tensor));
    auto scalar = tensor->scalar<std::string>();
    scalar(0) = serialized_;
  }

 private:
  std::shared_ptr<assembler::Assembler> assembler_;
  std::string serialized_;
};

REGISTER_KERNEL_BUILDER(Name("AssemblerSerialize").Device(DEVICE_CPU),
                        AssemblerSerializeOp);

class AssemblerServingOp : public OpKernel {
 public:
  explicit AssemblerServingOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    Attr_API(34459389, 1);  // AssemblerServingOp初始化次数
    model_uptime_ = time(NULL);
    LOG(INFO) << "\n\n";
    LOG(INFO) << " ------------------------ ";
    LOG(INFO) << "Init AssemblerServingOp ...";
    std::string serialized;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("serialized", &serialized));
    assembler_.reset(new assembler::Assembler(ctx));
    OP_REQUIRES(ctx, assembler_->ParseFromString(serialized),
                errors::Internal("assembler::ParseFromString error"));
    LOG(INFO) << "Assembler columns size = " << assembler_->columns().size();
    assembler_->PrintDebugInfo();
    LOG(INFO) << "Init AssemblerServingOp done";
    LOG(INFO) << " ------------------------ ";
    LOG(INFO) << "\n\n";
  }

  void Compute(OpKernelContext* ctx) override {
    Attr_API(34459390, 1);  // AssemblerServingOp总请求量
    time_t now = time(NULL);
    if (now - model_uptime_ > 120 * 60 * 60) {
      Attr_API(34459391, 1);  // 模型大于120小时没有更新
    } else if (now - model_uptime_ > 96 * 60 * 60) {
      Attr_API(34459392, 1);  // 模型大于96小时没有更新
    } else if (now - model_uptime_ > 72 * 60 * 60) {
      Attr_API(34459393, 1);  // 模型大于72小时没有更新
    } else if (now - model_uptime_ > 48 * 60 * 60) {
      Attr_API(34459394, 1);  // 模型大于48小时没有更新
    } else if (now - model_uptime_ > 24 * 60 * 60) {
      Attr_API(34459395, 1);  // 模型大于24小时没有更新
    } else if (now - model_uptime_ > 12 * 60 * 60) {
      Attr_API(34459396, 1);  // 模型大于12小时没有更新
    } else if (now - model_uptime_ > 6 * 60 * 60) {
      Attr_API(34459397, 1);  // 模型大于6小时没有更新
    } else {
      Attr_API(34459398, 1);  // 模型6小时内正常更新
    }

    const Tensor& user_feature_tensor = ctx->input(0);
    const Tensor& ctx_feature_tensor = ctx->input(1);
    const Tensor& item_feature_tensor = ctx->input(2);
    std::string user_feature = user_feature_tensor.flat<string>()(0);
    std::vector<std::string> items;
    std::vector<std::string> ctxs;
    auto flat_items = item_feature_tensor.flat<std::string>();
    auto flat_ctxs = ctx_feature_tensor.flat<std::string>();
    Attr_API(34459399, flat_items.size());  // 排序item总数
    for (int i = 0; i < flat_items.size(); ++i) {
      items.push_back(flat_items(i));
    }
    for (int i = 0; i < flat_ctxs.size(); ++i) {
      ctxs.push_back(flat_ctxs(i));
    }

    std::vector<std::vector<double>> features;
    assembler_->GetServingInputs(user_feature, ctxs, items, &features);
    // Create output tensors
    Tensor* output = NULL;
    size_t sz = assembler_->feature_size();
    TensorShape shape;
    shape.AddDim(features.size());
    shape.AddDim(sz);
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &output));
    auto matrix_output = output->matrix<double>();
    for (size_t i = 0; i < features.size(); ++i) {
      for (size_t j = 0; j < features[i].size(); ++j) {
        OP_REQUIRES(
            ctx, sz == features[i].size(),
            errors::Internal("Internal error, features.size not matched"));
        matrix_output(i, j) = features[i][j];
      }
    }
  }

 private:
  std::shared_ptr<assembler::Assembler> assembler_;
  time_t model_uptime_;
};

REGISTER_KERNEL_BUILDER(Name("AssemblerServing").Device(DEVICE_CPU),
                        AssemblerServingOp);

}  // namespace tensorflow
