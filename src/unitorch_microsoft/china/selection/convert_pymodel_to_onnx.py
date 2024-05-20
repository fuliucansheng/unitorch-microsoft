import os
import torch
import torch.nn as nn
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import onnx
import onnxruntime as ort
import onnxruntime.transformers.optimizer as optimizer
from onnxruntime.transformers.fusion_options import FusionOptions
import onnxruntime.quantization as quantization
import onnxruntime.quantization.preprocess as preprocess
from unitorch_microsoft.models.bletchley.modeling_v1 import (
    get_bletchley_text_config,
    BletchleyTextEncoder,
)
from unitorch_microsoft.models.bletchley.processing_v1 import BletchleyProcessor
fusion_options = FusionOptions("bert")
fusion_options.enable_embed_layer_norm = False
import numpy as np

class OnnxConverter:
    def __init__(self, 
                model,
                OutputFolder
                ):
        self.model = model
        self.OutputFolder = OutputFolder
        if not os.path.exists(self.OutputFolder):
            os.system('mkdir -p '+self.OutputFolder)

    def export(self,
               query_input_ids = None,
               query_attention_mask=None,
               do_optimize = True,
               do_quantize = True):
        torch.set_printoptions(profile="full")
        onnx_path = os.path.join(self.OutputFolder, 'model.onnx')
        torch.onnx.export(
            self.model,
            (query_input_ids, query_attention_mask),
            f=onnx_path,
            input_names=["query_input_ids", "query_attention_mask"],
            output_names=["qvec", "dummy_score"],
            export_params=True,
            dynamic_axes={"query_input_ids": {0: "querylen"}, "query_attention_mask": {0: "attentionlen"}},
            do_constant_folding=True,
            verbose=False,
            opset_version=13,
        )
        if do_optimize:
            opt_onnx_path = os.path.join(self.OutputFolder, 'model.optimized.onnx')
            optimized_model = optimizer.optimize_model(onnx_path,optimization_options=fusion_options)
            optimized_model.save_model_to_file(opt_onnx_path)
            onnx_path = opt_onnx_path
        if do_quantize:
            quant_onnx_path = os.path.join(self.OutputFolder, 'model.optimized.quantized.onnx')
            pre_path = onnx_path
            extra_options = {'DefaultTensorType':onnx.TensorProto.FLOAT}
            quantization.quantize_dynamic(
                pre_path, quant_onnx_path, extra_options=extra_options
            )

    def ParityCheck(self,
                    onnx_path,
                    query_input_ids, query_attention_mask):
        #output from model
        embedding, dummy_score = self.model(query_input_ids, query_attention_mask)
        print("model output")
        print(embedding)
        print(embedding.shape)
        #onnx output
        ort_session = ort.InferenceSession(onnx_path)
        outputs = ort_session.run(
            None,
            {"query_input_ids": query_input_ids.cpu().numpy(), "query_attention_mask": query_attention_mask.cpu().numpy()},
        )
        print("onnx output")
        for x in ort_session.get_outputs():
            print(x)
        print(outputs[0])
        print(outputs[0].shape)
        #compare result
        np.testing.assert_allclose(embedding.detach().cpu().numpy(), outputs[0], rtol=1e-03, atol=1e-05)

    def ParityCheckPair(self,
                    onnx_path,
                    query_input_ids_a, query_attention_mask_a,
		    query_input_ids_b, query_attention_mask_b):
        #output from model
        embedding1, dummy_score = self.model(query_input_ids_a, query_attention_mask_a)
        embedding2, dummy_score = self.model(query_input_ids_b, query_attention_mask_b)
        #onnx output
        ort_session = ort.InferenceSession(onnx_path)
        outputs1 = ort_session.run(
            None,
            {"query_input_ids": query_input_ids_a.cpu().numpy(), "query_attention_mask": query_attention_mask_a.cpu().numpy()},
        )
        outputs2 = ort_session.run(
            None,
            {"query_input_ids": query_input_ids_b.cpu().numpy(), "query_attention_mask": query_attention_mask_b.cpu().numpy()},
        )
	#cosines similarity
        embedding1 = embedding1.detach().cpu().numpy()
        embedding2 = embedding2.detach().cpu().numpy()
        cosine1 = np.dot(embedding1, embedding2.T) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        cosine1_2 = np.dot(embedding1, embedding2.T)
        cosine2 = np.dot(outputs1[0], outputs2[0].T) / (np.linalg.norm(outputs1[0]) * np.linalg.norm(outputs2[0]))
        cosine2_2 = np.dot(outputs1[0], outputs2[0].T)
        cosine3 = np.dot(embedding1, outputs1[0].T) / (np.linalg.norm(embedding1) * np.linalg.norm(outputs1[0]))
        cosine3_2 = np.dot(embedding1, outputs1[0].T)

        print(cosine1, cosine1_2)
        print(cosine2, cosine2_2)
        print(cosine3, cosine3_2)


class BletchleyForTextPretrainQueryEncoder(nn.Module):
    def __init__ (
        self,
        query_config_type: str,
        weight_path: str,
        projection_dim: Optional[int] = 1024,
    ):
        super().__init__()
        query_config = get_bletchley_text_config(
            query_config_type
        )

        self.query_embed_dim = query_config.hidden_size
        self.query_encoder = BletchleyTextEncoder(
            query_config, add_projection_layer=False
        )
        self.query_projection = nn.Linear(
            self.query_embed_dim,
            projection_dim,
        )
        state_dict = torch.load(weight_path, map_location="cpu")
        self.check_load(self.state_dict(), state_dict)
        self.load_state_dict(state_dict, strict=False)
   
    def check_load(self, model_state_dict, ckpt_state_dict):
        non_load_keys = []
        check_node = []
        for key,value in list(model_state_dict.items()):
            if key not in ckpt_state_dict or value.shape != ckpt_state_dict[key].shape:
                if key not in non_load_keys:
                    non_load_keys.append(key)
        for key,value in list(ckpt_state_dict.items()):    
            if "layer.0" in key and "MatMul_" in key:
                check_node.append((key, ckpt_state_dict[key]))
        print("%f params in model can't find weight"%(float(len(non_load_keys))/len(model_state_dict)))
        print(non_load_keys)
        print(check_node)
        #print(ckpt_state_dict.keys())


    def forward(
        self,
        query_input_ids=None,
        query_attention_mask=None,
    ):
        query_input_ids = query_input_ids.unsqueeze(0)
        query_attention_mask = query_attention_mask.unsqueeze(0)
        query_outputs = self.query_encoder(
            input_ids=query_input_ids,
            attention_mask=query_attention_mask,
        )
        query_embeds = query_outputs[:, 0]
        query_embeds = self.query_projection(query_embeds)
        query_embeds = query_embeds / query_embeds.norm(dim=-1, keepdim=True)

        return query_embeds.view(1, -1), query_embeds[:, 0].view(1).squeeze()

if __name__ == '__main__':

    #prepare input:
    query = "这是一个测试"
    processor = BletchleyProcessor(max_seq_length = 16)
    outputs = processor._tokenize(query, max_seq_length = 16)
    input_ids = outputs.input_ids
    attention_mask = outputs.attention_mask
    #another query
    query2 = "苹果"
    outputs2 = processor._tokenize(query2, max_seq_length = 16)
    input_ids2 = outputs2.input_ids
    attention_mask2 = outputs2.attention_mask
    
    print(input_ids.shape)
    print(attention_mask.shape)
    print(input_ids)
    print(attention_mask)
    #prepare model
    model_file = '/home/xucha/unitorch-microsoft/src/unitorch_microsoft/ckpt/pytorch_model_noquant.bin'
    model = BletchleyForTextPretrainQueryEncoder(query_config_type='0.15B', weight_path=model_file, projection_dim=64)
    model.eval()

    #export to onnx
    output_folder = './onnx_noquant'
    onnx_converter = OnnxConverter(model, output_folder)
    #onnx_converter.export(input_ids, attention_mask)

    #Parity check
    onnx_path='/home/xucha/unitorch-microsoft/src/unitorch_microsoft/china/selection/onnx_noquant/model.optimized.quantized.onnx'
    onnx_converter.ParityCheckPair(onnx_path, input_ids, attention_mask, input_ids2, attention_mask2)
    onnx_converter.ParityCheck(onnx_path, input_ids, attention_mask)

    

       



