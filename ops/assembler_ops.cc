
#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("Assembler")
    .Input("input: string")
    .Output("feature: double")
    .Output("label: double")
    .Output("weight: double")
    .SetIsStateful()
    .Attr("conf_path: string = ''")
    .Doc(R"doc(
)doc");

REGISTER_OP("AssemblerScheme")
    .Output("scheme: string")
    .SetIsStateful()
    .Attr("conf_path: string = ''")
    .Doc(R"doc(
)doc");

REGISTER_OP("AssemblerSerialize")
    .Output("output: string")
    .SetIsStateful()
    .Attr("conf_path: string = ''")
    .Doc(R"doc(
)doc");

REGISTER_OP("AssemblerServing")
    .Input("user_feature: string")
    .Input("ctx_features: string")
    .Input("item_feature: string")
    .Output("features: double")
    .SetIsStateful()
    .Attr("serialized: string")
    .Doc(R"doc(
)doc");

}  // namespace tensorflow
