# ranker ini file
# TG
# Expression=(if (== TrafficGroupId 816) 1 0)
TGs = "T0:816, T1:865, T2:1079, T3:1031, T4:912, T5:1034, T6:1028, T7:726, T8:910, T9:809, T10:601, T11:746, T12:758, T13:566, T14:1035, T15:1027, T16:1064, T17:1032, T18:1029, T19:673, T20:834, T21:914, T22:911, T23:1081, T24:1068, T25:1033, T26:690, T27:542, T28:908, T29:1010, T30:1058"
TGItems = [x.strip().split(":") for x in TGs.split(", ")]
TGItems = [(int(v2), int(v1[1:])) for v1, v2 in TGItems]

def generate_expression(items, name, default):
    if len(items) == 0:
        return default
    key, value = items[0]
    return f"(if (== {name} {key}) {value} {generate_expression(items[1:], name, default)})"

TGExpression = generate_expression(TGItems, "TrafficGroupId", 31)

# Demand Type
Ds = "D0:TA, D1:PA, D2:AIM-PA, D3:AIM-IA"
DemandTypeId = 0

ranker = f"""
[DNN]
Sink=4

[Input:1]
Intercept=0
Slope=1
Transform=FreeForm
Expression={TGExpression}

[Input:2]
Type=FloatVector
Name=DenseFloatStoreVector:&MsanUserDenseVector4
Size=32

[Input:3]
Type=FloatVector
Name=DenseFloatStoreVector:&MsanAdDenseVector4
Size=32

[Input:4]
Expression={DemandTypeId}
Transform=FreeForm
Slope=1.0
Intercept=0

[Evaluator:1]
Operation=Cast
To=int64
NumArgs=1
Arg1=I:1

[Evaluator:2]
Operation=Cast
To=int64
NumArgs=1
Arg1=I:4

[Evaluator:3]
Operation=RunOnnxModel
ModelPath=classifier_click.onnx
TGId=E:1
DemandId=E:2
UserEmb=I:2
AdsEmb=I:3
ModelOutputs=Score

[Evaluator:4]
Operation=RunOnnxModel
ModelPath=classifier_conv.onnx
TGId=E:1
DemandId=E:2
UserEmb=I:2
AdsEmb=I:3
ModelOutputs=Score

[Evaluator:5]
Operation=FreeForm
Expression=(+ (* E:3 0.5) (* E:4 0.5))

"""

with open("./ranker.ini", "w") as f:
    f.write(ranker)


# export onnx model
import torch
import torch.nn as nn
from unitorch.models import GenericModel
from unitorch.cli import cached_path

class ClickModel(GenericModel):
    def __init__(self, projection_dim: int = 32, num_tgs=50, num_demands=10):
        super().__init__()
        self.projection_dim = projection_dim
        self.click_tg_embedding = nn.Embedding(num_tgs, self.projection_dim)
        self.click_tg_layer_norm = nn.LayerNorm(self.projection_dim)
        self.click_demand_embedding = nn.Embedding(num_demands, self.projection_dim)
        self.click_demand_layer_norm = nn.LayerNorm(self.projection_dim)
        self.user_click_final_projection = nn.Linear(
            self.projection_dim * 3,
            self.projection_dim,
        )
        self.click_classifier = nn.Linear(1, 1)
    
    def forward(self, UserEmb, AdEmb, TGId, DemandId):
        tg_ids = TGId.reshape(1).to(torch.int64)
        demand_ids = DemandId.reshape(1).to(torch.int64)
        tg_emb = self.click_tg_layer_norm(self.click_tg_embedding(tg_ids))
        demand_emb = self.click_demand_layer_norm(
            self.click_demand_embedding(demand_ids)
        )
        user_emb = UserEmb.reshape(1, 32)
        user_emb = torch.cat([user_emb, tg_emb, demand_emb], dim=-1)
        user_emb = self.user_click_final_projection(user_emb)
        user_emb = user_emb / user_emb.norm(
                dim=-1, keepdim=True
            )        
        ads_emb = AdEmb.reshape(1, 32)
        ads_emb = ads_emb / ads_emb.norm(
                dim=-1, keepdim=True
            )
        scores = torch.sum(user_emb * ads_emb, dim=-1)
        scores = self.click_classifier(scores)
        scores = torch.sigmoid(scores)
        return scores.reshape(1, 1)

class ConvModel(GenericModel):
    def __init__(self, projection_dim: int = 32, num_tgs=50, num_demands=10):
        super().__init__()
        self.projection_dim = projection_dim
        self.conv_tg_embedding = nn.Embedding(num_tgs, self.projection_dim)
        self.conv_tg_layer_norm = nn.LayerNorm(self.projection_dim)
        self.conv_demand_embedding = nn.Embedding(num_demands, self.projection_dim)
        self.conv_demand_layer_norm = nn.LayerNorm(self.projection_dim)
        self.user_conv_final_projection = nn.Linear(
            self.projection_dim * 3,
            self.projection_dim,
        )
        self.conv_classifier = nn.Linear(1, 1)

    def forward(self, UserEmb, AdsEmb, TGId, DemandId):
        tg_ids = TGId.reshape(1).to(torch.int64)
        demand_ids = DemandId.reshape(1).to(torch.int64)
        tg_emb = self.conv_tg_layer_norm(self.conv_tg_embedding(tg_ids))
        demand_emb = self.conv_demand_layer_norm(
            self.conv_demand_embedding(demand_ids)
        )
        user_emb = UserEmb.reshape(1, 32)
        user_emb = torch.cat([user_emb, tg_emb, demand_emb], dim=-1)
        user_emb = self.user_conv_final_projection(user_emb)
        user_emb = user_emb / user_emb.norm(
                dim=-1, keepdim=True
            )        
        ads_emb = AdsEmb.reshape(1, 32)
        ads_emb = ads_emb / ads_emb.norm(
                dim=-1, keepdim=True
            )
        scores = torch.sum(user_emb * ads_emb, dim=-1)
        scores = self.conv_classifier(scores)
        scores = torch.sigmoid(scores)
        return scores.reshape(1, 1)
    
dummy_user = torch.randn(1, 32)
dummy_ad = torch.randn(1, 32)
dummy_tg = torch.tensor([0])
dummy_demand = torch.tensor([0])

weight_path = cached_path("https://unitorchblobfuse.blob.core.windows.net/shares/models/msan/pytorch_model.msan.l1.bletchley.v1.0.796.bin")

click_model = ClickModel()
click_model.from_pretrained(weight_path)
click_model.eval()
torch.onnx.export(click_model, (dummy_user, dummy_ad, dummy_tg, dummy_demand), "classifier_click.onnx", input_names=["UserEmb", "AdsEmb", "TGId", "DemandId"], output_names=["Score"], export_params = True, do_constant_folding = True, opset_version=11,)

conv_model = ConvModel()
conv_model.from_pretrained(weight_path)
conv_model.eval()
torch.onnx.export(conv_model, (dummy_user, dummy_ad, dummy_tg, dummy_demand), "classifier_conv.onnx", input_names=["UserEmb", "AdsEmb", "TGId", "DemandId"], output_names=["Score"], export_params = True, do_constant_folding = True, opset_version=11)

import onnxruntime as ort

check_user = torch.randn(1, 32)
check_ad = torch.randn(1, 32)
check_tg = torch.tensor([1])
check_demand = torch.tensor([1])

click_truth = click_model(check_user, check_ad, check_tg, check_demand).detach().numpy()

click_session = ort.InferenceSession("classifier_click.onnx")
click_result = click_session.run(None, {"UserEmb": check_user.numpy(), "AdsEmb": check_ad.numpy(), "TGId": check_tg.numpy(), "DemandId": check_demand.numpy()})
click_truth = click_model(check_user, check_ad, check_tg, check_demand).detach().numpy()

print("click model:")
for input in click_session.get_inputs():
    print(input)

for output in click_session.get_outputs():
    print(output)

print(click_result, click_truth)

conv_session = ort.InferenceSession("classifier_conv.onnx")
conv_result = conv_session.run(None, {"UserEmb": check_user.numpy(), "AdsEmb": check_ad.numpy(), "TGId": check_tg.numpy(), "DemandId": check_demand.numpy()})
conv_truth = conv_model(check_user, check_ad, check_tg, check_demand).detach().numpy()

print("conv model:")
for input in conv_session.get_inputs():
    print(input)

for output in conv_session.get_outputs():
    print(output)

print(conv_result, conv_truth)
