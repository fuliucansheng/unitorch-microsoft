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
Sink=15

[Input:1]
Expression=(/ 1.0 127.5)
Transform=FreeForm
Slope=1.0
Intercept=0

[Input:2]
Intercept=0
Slope=1
Transform=FreeForm
Expression=(if (== TrafficGroupId 816) 0 (if (== TrafficGroupId 865) 1 (if (== TrafficGroupId 1079) 2 (if (== TrafficGroupId 1031) 3 (if (== TrafficGroupId 912) 4 (if (== TrafficGroupId 1034) 5 (if (== TrafficGroupId 1028) 6 (if (== TrafficGroupId 726) 7 (if (== TrafficGroupId 910) 8 (if (== TrafficGroupId 809) 9 (if (== TrafficGroupId 601) 10 (if (== TrafficGroupId 746) 11 (if (== TrafficGroupId 758) 12 (if (== TrafficGroupId 566) 13 (if (== TrafficGroupId 1035) 14 (if (== TrafficGroupId 1027) 15 (if (== TrafficGroupId 1064) 16 (if (== TrafficGroupId 1032) 17 (if (== TrafficGroupId 1029) 18 (if (== TrafficGroupId 673) 19 (if (== TrafficGroupId 834) 20 (if (== TrafficGroupId 914) 21 (if (== TrafficGroupId 911) 22 (if (== TrafficGroupId 1081) 23 (if (== TrafficGroupId 1068) 24 (if (== TrafficGroupId 1033) 25 (if (== TrafficGroupId 690) 26 (if (== TrafficGroupId 542) 27 (if (== TrafficGroupId 908) 28 (if (== TrafficGroupId 1010) 29 (if (== TrafficGroupId 1058) 30 31)))))))))))))))))))))))))))))))

[Input:3]
;click user embedding
Type=FloatVector
Name=DenseFloatQueryVector:&Queene3
Size=32

[Input:4]
;click ad embedding
Type=FloatVector
Name=DenseFloatStoreVector:&MsanAdDenseVector6
Size=32

[Input:5]
;conv user embedding
Type=FloatVector
Name=DenseFloatQueryVector:&Queene4
Size=32

[Input:6]
;conv ad embedding
Type=FloatVector
Name=DenseFloatStoreVector:&MsanAdDenseVector7
Size=32

;DemandType: TA：0
[Input:7]
Expression=0
Transform=FreeForm
Slope=1.0
Intercept=0


[Data:1]
Type=Dense
Rows=1
Columns=32
Values=-127.5 -127.5 -127.5 -127.5 -127.5 -127.5 -127.5 -127.5 -127.5 -127.5 -127.5 -127.5 -127.5 -127.5 -127.5 -127.5 -127.5 -127.5 -127.5 -127.5 -127.5 -127.5 -127.5 -127.5 -127.5 -127.5 -127.5 -127.5 -127.5 -127.5 -127.5 -127.5

[Data:2]
Type=Dense
Rows=1
Columns=32
Values=4.0 4.0 4.0 4.0 4.0 4.0 4.0 4.0 4.0 4.0 4.0 4.0 4.0 4.0 4.0 4.0 4.0 4.0 4.0 4.0 4.0 4.0 4.0 4.0 4.0 4.0 4.0 4.0 4.0 4.0 4.0 4.0


;click user embedding dequantization
[Evaluator:1]
Operation=Sum
NumArgs=2
Arg1=I:3
Arg2=D:1

[Evaluator:2]
Operation=PerElementProduct
NumArgs=2
Arg1=E:1
Arg2=I:1

;final click user embedding = click user embedding * 4
[Evaluator:3]
Operation=PerElementProduct
NumArgs=2
Arg1=E:2
Arg2=D:2

;conv user embedding dequantization
[Evaluator:4]
Operation=Sum
NumArgs=2
Arg1=I:5
Arg2=D:1

[Evaluator:5]
Operation=PerElementProduct
NumArgs=2
Arg1=E:4
Arg2=I:1

;final conv user embedding = conv user embedding * 4
[Evaluator:6]
Operation=PerElementProduct
NumArgs=2
Arg1=E:5
Arg2=D:2

;click ad embedding * 4
[Evaluator:7]
Operation=PerElementProduct
NumArgs=2
Arg1=I:4
Arg2=D:2


;conv ad embedding * 4
[Evaluator:8]
Operation=PerElementProduct
NumArgs=2
Arg1=I:6
Arg2=D:2

; TG
[Evaluator:9]
Operation=Cast
To=int64
NumArgs=1
Arg1=I:2

;DemandType
[Evaluator:10]
Operation=Cast
To=int64
NumArgs=1
Arg1=I:7

; pClick
[Evaluator:11]
Operation=RunOnnxModel
ModelPath=classifier_click.onnx
TGId=E:9
DemandId=E:10
UserEmb=E:3
AdsEmb=E:7
ModelOutputs=Score

[Evaluator:12]
; Check if user embedding is available
Operation=DotProduct
NumArgs=2
Arg1=I:3
Arg2=I:3

[Evaluator:13]
; Check if ad embedding is available
Operation=DotProduct
NumArgs=2
Arg1=I:4
Arg2=I:4

;pConvert
[Evaluator:14]
Operation=RunOnnxModel
ModelPath=classifier_conv.onnx
TGId=E:9
DemandId=E:10
UserEmb=E:6
AdsEmb=E:8
ModelOutputs=Score


[Evaluator:15]
Operation=FreeForm
; Expression=(if (> E:user 0) (if (> E:ad 0) E:output 0.6789) (if (> E:ad 0) 0.7890 0.89))
Expression=(if (> E:12 0) (if (> E:13 0) (+ (* E:11 0.6) (* E:14 0.4)) 0.6789) (if (> E:13 0) 0.7890 0.89))
ExportedId=1
ExportedIntercept=0
ExportedSlope=100000000

[Config:QueryVector:&Queene3]
RequestSource=Queene3
[Config:QueryVector:&Queene4]
RequestSource=Queene4
[Config:StoreVector:&MsanAdDenseVector6]
ModelName=MsanAdDenseVector6
[Config:StoreVector:&MsanAdDenseVector7]
ModelName=MsanAdDenseVector7

"""

with open("./classifier_click.ini", "w") as f:
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

        eps = 1e-8  # Small constant to avoid division by zero

        user_emb = UserEmb.reshape(1, 32)
        user_emb = torch.cat([user_emb, tg_emb, demand_emb], dim=-1)
        user_emb = self.user_click_final_projection(user_emb)
        user_emb = user_emb / (user_emb.norm(dim=-1, keepdim=True) + eps)

        ads_emb = AdEmb.reshape(1, 32)
        ads_emb = ads_emb / (ads_emb.norm(dim=-1, keepdim=True) + eps)

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
        demand_emb = self.conv_demand_layer_norm(self.conv_demand_embedding(demand_ids))

        eps = 1e-8  # Small constant to avoid division by zero

        user_emb = UserEmb.reshape(1, 32)
        user_emb = torch.cat([user_emb, tg_emb, demand_emb], dim=-1)
        user_emb = self.user_conv_final_projection(user_emb)
        user_emb = user_emb / (user_emb.norm(dim=-1, keepdim=True) + eps)

        ads_emb = AdsEmb.reshape(1, 32)
        ads_emb = ads_emb / (ads_emb.norm(dim=-1, keepdim=True) + eps)

        scores = torch.sum(user_emb * ads_emb, dim=-1)
        scores = self.conv_classifier(scores)
        scores = torch.sigmoid(scores)
        return scores.reshape(1, 1)


dummy_user = torch.randn(1, 32)
dummy_ad = torch.randn(1, 32)
dummy_tg = torch.tensor([0])
dummy_demand = torch.tensor([0])

weight_path = cached_path(
    "https://unitorchblobfuse.blob.core.windows.net/shares/models/msan/pytorch_model.msan.l1.bletchley.v1.0.796.bin"
)

click_model = ClickModel()
click_model.from_pretrained(weight_path)
click_model.eval()
torch.onnx.export(
    click_model,
    (dummy_user, dummy_ad, dummy_tg, dummy_demand),
    "classifier_click.onnx",
    input_names=["UserEmb", "AdsEmb", "TGId", "DemandId"],
    output_names=["Score"],
    export_params=True,
    do_constant_folding=True,
    opset_version=11,
)

conv_model = ConvModel()
conv_model.from_pretrained(weight_path)
conv_model.eval()
torch.onnx.export(
    conv_model,
    (dummy_user, dummy_ad, dummy_tg, dummy_demand),
    "classifier_conv.onnx",
    input_names=["UserEmb", "AdsEmb", "TGId", "DemandId"],
    output_names=["Score"],
    export_params=True,
    do_constant_folding=True,
    opset_version=11,
)

import onnxruntime as ort

# Click test case
# input_string = "0.1411764705882353 3.4352941176470586 -0.1411764705882353 -0.39215686274509803 -1.2078431372549019 0.23529411764705882 -0.611764705882353 -0.2980392156862745 1.803921568627451 -0.611764705882353 -0.17254901960784313 1.0196078431372548 -0.5490196078431373 0.26666666666666666 -1.3019607843137255 0.1411764705882353 -0.6431372549019608 -0.8 -0.8 0.32941176470588235 -0.3607843137254902 -0.9254901960784314 2.023529411764706 -0.8 0.6745098039215687 -1.1764705882352942 0.3607843137254902 -0.5803921568627451 0.611764705882353 -0.4549019607843137 -0.17254901960784313 0.8313725490196079"
input_string = "0.145261 3.436224 -0.154717 -0.380139 -1.218375 0.241882 -0.605219 -0.304838 1.802028 -0.619002 -0.161716 1.017094 -0.559977 0.255828 -1.289941 0.146162 -0.627858 -0.805891 -0.802558 0.316891 -0.371566 -0.921741 2.035887 -0.787782 0.687254 -1.188827 0.345612 -0.566132 0.602964 -0.462149 -0.179822 0.845118"

float_values = [float(value) for value in input_string.split()]
check_user = torch.tensor(float_values, dtype=torch.float32)
check_user = check_user.view(1, -1)

# input_string = "-0.172548 -0.20392 0.04706 0.235296 0.141176 0.109804 0.20392 -0.266668 0.172548 0.235296 0.04706 -0.04706 0.015688 -0.04706 0.015688 0.109804 -0.109804 0.078432 -0.29804 0.29804 0.109804 0.172548 0.392156 -0.04706 0.015688 0.04706 -0.235296 0.392156 -0.015688 -0.109804 -0.172548 0.04706"
input_string = "-0.161775 -0.212988 0.042703 0.233323 0.144678 0.099189 0.190394 -0.271432 0.162829 0.232118 0.054603 -0.033854 0.014112 -0.045791 0.031274 0.120954 -0.099189 0.082846 -0.303817 0.306227 0.101072 0.181356 0.383198 -0.052645 0.014291 0.056373 -0.222026 0.376570 -0.002238 -0.096251 -0.165088 0.050762"

float_values = [float(value) for value in input_string.split()]
check_ad = torch.tensor(float_values, dtype=torch.float32)
check_ad = check_ad.view(1, -1)

check_tg = torch.tensor([0])
check_demand = torch.tensor([0])

click_truth = click_model(check_user, check_ad, check_tg, check_demand).detach().numpy()

click_session = ort.InferenceSession("classifier_click.onnx")
click_result = click_session.run(
    None,
    {
        "UserEmb": check_user.numpy(),
        "AdsEmb": check_ad.numpy(),
        "TGId": check_tg.numpy(),
        "DemandId": check_demand.numpy(),
    },
)
click_truth = click_model(check_user, check_ad, check_tg, check_demand).detach().numpy()

print("click model:")
for input in click_session.get_inputs():
    print(input)

for output in click_session.get_outputs():
    print(output)

print(click_result, click_truth)


# Conv test case
# input_string = "0.7372549019607844 -0.9254901960784314 -0.611764705882353 -1.0196078431372548 0.17254901960784313 1.5215686274509803 -0.5803921568627451 0.6431372549019608 -0.1411764705882353 -1.2392156862745098 -0.8941176470588236 1.2392156862745098 -0.6745098039215687 0.5803921568627451 -0.2980392156862745 2.9647058823529413 -0.5176470588235295 -0.3607843137254902 -0.1411764705882353 -1.0196078431372548 -1.4588235294117646 -0.8313725490196079 1.772549019607843 1.5529411764705883 -0.32941176470588235 -0.4549019607843137 -0.39215686274509803 0.7686274509803922 0.4235294117647059 0.8 -0.7686274509803922 -0.5490196078431373"
input_string = "0.740840 -0.922068 -0.600169 -1.033987 0.171723 1.519307 -0.589485 0.657826 -0.141143 -1.233740 -0.897266 1.251632 -0.672141 0.585924 -0.295949 2.958390 -0.513785 -0.366502 -0.142920 -1.025336 -1.455941 -0.817830 1.760587 1.549752 -0.339848 -0.462646 -0.386421 0.765737 0.419093 0.785951 -0.774664 -0.544622"

float_values = [float(value) for value in input_string.split()]
check_user = torch.tensor(float_values, dtype=torch.float32)
check_user = check_user.view(1, -1)

# input_string = "0.29804 -0.04706 0.04706 -0.015688 -0.20392 0.235296 0.04706 0.109804 0.172548 0.015688 -0.015688 -0.235296 -0.141176 0.329412 -0.04706 0.109804 -0.109804 0.172548 0.078432 0.20392 -0.266668 -0.172548 -0.20392 0.172548 0.29804 -0.141176 -0.04706 -0.141176 0.423528 0.04706 -0.015688 -0.109804"
input_string = "0.301814 -0.042355 0.038926 -0.009265 -0.188334 0.231355 0.057572 0.107686 0.178878 0.006789 -0.001288 -0.249869 -0.139852 0.338309 -0.039258 0.117942 -0.094300 0.164093 0.087241 0.191398 -0.251734 -0.163161 -0.204850 0.165825 0.290360 -0.150774 -0.055308 -0.140385 0.418491 0.036628 -0.004712 -0.108818"

float_values = [float(value) for value in input_string.split()]
check_ad = torch.tensor(float_values, dtype=torch.float32)
check_ad = check_ad.view(1, -1)

check_tg = torch.tensor([0])
check_demand = torch.tensor([0])


conv_session = ort.InferenceSession("classifier_conv.onnx")
conv_result = conv_session.run(
    None,
    {
        "UserEmb": check_user.numpy(),
        "AdsEmb": check_ad.numpy(),
        "TGId": check_tg.numpy(),
        "DemandId": check_demand.numpy(),
    },
)
conv_truth = conv_model(check_user, check_ad, check_tg, check_demand).detach().numpy()

print("conv model:")
for input in conv_session.get_inputs():
    print(input)

for output in conv_session.get_outputs():
    print(output)

print(conv_result, conv_truth)

# online:0.812 offline:0.07958
print("final score:", 0.6 * click_result[0][0] + 0.4 * conv_result[0][0])
