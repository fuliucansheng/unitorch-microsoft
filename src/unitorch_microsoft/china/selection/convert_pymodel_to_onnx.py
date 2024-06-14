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
               do_optimize = False,
               do_quantize = True):
        torch.set_printoptions(profile="full")
        onnx_path = os.path.join(self.OutputFolder, 'model.onnx')
        torch.onnx.export(
            self.model,
            query_input_ids,
            f=onnx_path,
            input_names=["input_ids"],
            output_names=["dummy_score", "embeddings"],
            export_params=True,
            #dynamic_axes={"input_ids": {0:"max_len"}},
            do_constant_folding=True,
            verbose=False,
            opset_version=11,
        )
        if do_optimize:
            opt_onnx_path = os.path.join(self.OutputFolder, 'model.optimized.onnx')
            optimized_model = optimizer.optimize_model(onnx_path,optimization_options=fusion_options)
            optimized_model.save_model_to_file(opt_onnx_path)
            onnx_path = opt_onnx_path
        if do_quantize:
            quant_onnx_path = os.path.join(self.OutputFolder, 'model.optimized.quantized.onnx')
            quantization.quantize_dynamic(
                onnx_path, quant_onnx_path)
	    #pre_path = onnx_path
            #extra_options = {'DefaultTensorType':onnx.TensorProto.FLOAT}
            #quantization.quantize_dynamic(
            #    pre_path, quant_onnx_path, extra_options=extra_options
            #)

    def Quant(self, data):
        import math
        sumval = 0.0
        for tmp in data[0]:
            sumval += tmp*tmp
        sumval = math.sqrt(sumval)
        res = []
        for tmp in data[0]:
            tmp = float(tmp)/sumval
            if tmp < -1:
                tmp = 0
            elif tmp > 255:
                tmp = 255
            else:
                tmp = 127.5*tmp + 127.5
            res.append(tmp)
        res = np.array([res], dtype=np.uint8)
        return res
    
    def L2(self, data0, data1):
        val = 0
        for k,q in zip(data0,data1):
            val += (k-q)*(k-q)
        return val




    def ParityCheck(self,
                    onnx_path,
                    query_input_ids):
        #output from model
        dummy_score, embedding = self.model(query_input_ids)
        embedding = embedding.detach().cpu().numpy() 
        print("=========== model float embedding ===============")
        print(embedding)
        #onnx output
        ort_session = ort.InferenceSession(onnx_path)
        outputs = ort_session.run(
            None,
            {"input_ids": query_input_ids.cpu().numpy()},
        )
        print("=========== onnx float embedding ==============")
        print(outputs[1])
        
        #check after quant to uint8
        model_emb = self.Quant(embedding)
        print("========== model uint8 embedding ==============")
        print(model_emb)

        onnx_emb = self.Quant(outputs[1])
        print("========== onnx uint8 embedding ===============")
        print(onnx_emb)

        #calculate cosine distance && L2 distance
        cosine = np.dot(embedding[0], outputs[1][0].T)/ (np.linalg.norm(embedding[0]) * np.linalg.norm(outputs[1][0]))
        dist = self.L2(model_emb[0], onnx_emb[0])
        print("cosine: "+str(cosine)+" L2: "+str(dist))

        #np.testing.assert_allclose(embedding.detach().cpu().numpy(), outputs[1], rtol=1e-03, atol=1e-05)





    def ParityCheckPair(self,
                    onnx_path,
                    query_input_ids_a,
		    query_input_ids_b,):
        #output from model
        dummy_score, embedding1 = self.model(query_input_ids_a)
        dummy_score, embedding2 = self.model(query_input_ids_b)
        #onnx output
        ort_session = ort.InferenceSession(onnx_path)
        outputs1 = ort_session.run(
            None,
            {"input_ids": query_input_ids_a.cpu().numpy()},
        )
        outputs2 = ort_session.run(
            None,
            {"input_ids": query_input_ids_b.cpu().numpy()},
        )
	#cosines similarity
        embedding1 = embedding1.detach().cpu().numpy()
        embedding2 = embedding2.detach().cpu().numpy()
        cosine1 = np.dot(embedding1, embedding2.T) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        cosine1_2 = np.dot(embedding1, embedding2.T)
        cosine2 = np.dot(outputs1[1], outputs2[1].T) / (np.linalg.norm(outputs1[1]) * np.linalg.norm(outputs2[1]))
        cosine2_2 = np.dot(outputs1[1], outputs2[1].T)
        cosine3 = np.dot(embedding1, outputs1[1].T) / (np.linalg.norm(embedding1) * np.linalg.norm(outputs1[1]))
        cosine3_2 = np.dot(embedding1, outputs1[1].T)
        
        print(embedding1)
        print(outputs1[1])

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
        #self.check_load(self.state_dict(), state_dict)
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
        #filter pad tokens
        query_input_ids = query_input_ids[query_input_ids != 1].unsqueeze(-1)
        query_input_ids = torch.cat(
            [torch.tensor([0], dtype=torch.int64).reshape(1, -1), query_input_ids],
            dim=0,
        )
        query_input_ids = torch.cat(
            [query_input_ids, torch.tensor([2], dtype=torch.int64).reshape(1, -1)],
            dim=0,
        )
        query_input_ids = query_input_ids.reshape(1, -1)
        #gen attention mask
        query_attention_mask = torch.ones([1, query_input_ids.shape[1]]).reshape(1, -1).to(query_input_ids)
        query_input_ids[query_input_ids > 250001] = 250001
        query_input_ids[query_input_ids < 0] = 0
        
        query_outputs = self.query_encoder(
            input_ids=query_input_ids,
            attention_mask=query_attention_mask,
        )
        query_embeds = query_outputs[:, 0]
        query_embeds = self.query_projection(query_embeds)
        query_embeds = query_embeds / query_embeds.norm(dim=-1, keepdim=True)

        return query_embeds[:, 0].view(1).squeeze(), query_embeds.view(1,64)

if __name__ == '__main__':

    #prepare input from real query:
    '''
    query = "女 夏 运动 两 件 套"
    processor = BletchleyProcessor(max_seq_length = 16)
    outputs = processor._tokenize(query, max_seq_length = 16)
    input_ids = outputs.input_ids.unsqueeze(-1)
    print(input_ids)
    '''
    #prepare input from fixed input
    input_ids = torch.tensor([6,4870,6,13525,6,28188,6,6442,6,7006,6,14046,1,1,1,1]).unsqueeze(0)
    print(input_ids.shape)
    
    #prepare model
    model_file = '/home/xucha/unitorch-microsoft/src/unitorch_microsoft/ckpt/pytorch_model_quant.bin'
    model = BletchleyForTextPretrainQueryEncoder(query_config_type='0.15B', weight_path=model_file, projection_dim=64)
    model.eval()

    #export to onnx
    output_folder = './onnx_quant'
    onnx_converter = OnnxConverter(model, output_folder)
    #onnx_converter.export(input_ids)

    #Parity check
    onnx_path='/home/xucha/unitorch-microsoft/src/unitorch_microsoft/china/selection/onnx_quant/model.quant.onnx'
    #onnx_converter.ParityCheckPair(onnx_path, input_ids, input_ids2)
    onnx_converter.ParityCheck(onnx_path, input_ids)

