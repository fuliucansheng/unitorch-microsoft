import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { UploadCloud, Image as ImageIcon, Loader2, Focus, Settings2, ArrowLeft } from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5000';

type ModelType = 'basnet' | 'detr';

export default function ROIDetection() {
  const navigate = useNavigate();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<ModelType>('basnet');
  
  // 独立保存两个模型的阈值
  const [thresholds, setThresholds] = useState({ basnet: 0.1, detr: 0.5 });
  // 独立保存两个模型的结果图
  const [results, setResults] = useState<{ basnet: string | null; detr: string | null }>({ basnet: null, detr: null });
  // 独立保存两个模型的加载状态
  const [loading, setLoading] = useState({ basnet: false, detr: false });

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setSelectedFile(file);
    setPreview(URL.createObjectURL(file));
    setResults({ basnet: null, detr: null });
  };

  const detectROI = async (model: ModelType) => {
    if (!selectedFile) return;
    
    setLoading(prev => ({ ...prev, [model]: true }));
    try {
      const formData = new FormData();
      formData.append('image', selectedFile);

      const url = `${API_BASE}/microsoft/apps/spaces/picasso/${model}/generate1?threshold=${thresholds[model]}`;
      const response = await fetch(url, { 
        method: 'POST', 
        body: formData,
        headers: {
          'accept': 'application/json'
        }
      });

      if (!response.ok) throw new Error('API response was not ok');

      const blob = await response.blob();
      const imageUrl = URL.createObjectURL(blob);
      
      setResults(prev => ({ ...prev, [model]: imageUrl }));
    } catch (error) {
      console.error(`API Error (${model}):`, error);
      alert(`Failed to analyze image with ${model.toUpperCase()}. Ensure API server is running.`);
    } finally {
      setLoading(prev => ({ ...prev, [model]: false }));
    }
  };

  const handleThresholdChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setThresholds(prev => ({
      ...prev,
      [activeTab]: parseFloat(e.target.value)
    }));
  };

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <div className="flex items-start sm:items-center gap-4">
        <button 
          onClick={() => navigate('/picasso/examples')}
          className="p-2.5 rounded-full bg-white dark:bg-zinc-900/60 border border-black/[0.08] dark:border-white/[0.05] shadow-sm hover:bg-zinc-50 dark:hover:bg-zinc-800 transition-colors text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-100 shrink-0"
        >
          <ArrowLeft size={18} />
        </button>
        <div>
          <h2 className="text-3xl font-bold text-zinc-900 dark:text-zinc-100 tracking-tight">ROI Detection</h2>
          <p className="text-zinc-500 dark:text-zinc-400 mt-1.5 font-light">Detect regions of interest using BASNet (V1) or DETR (V2) models.</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* 左侧上传区 */}
        <div className="bg-white dark:bg-zinc-900/40 border border-black/[0.04] dark:border-white/[0.05] rounded-3xl p-6 shadow-sm flex flex-col items-center justify-center min-h-[400px]">
          {preview ? (
            <div className="relative w-full h-full flex flex-col items-center justify-center">
              <img src={preview} alt="Preview" className="max-h-[300px] object-contain rounded-xl shadow-md" />
              <label className="mt-6 px-4 py-2 bg-white dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded-lg text-sm font-medium cursor-pointer hover:bg-zinc-50 dark:hover:bg-zinc-700/50 transition-colors">
                Upload New Image
                <input type="file" accept="image/*" className="hidden" onChange={handleFileChange} />
              </label>
            </div>
          ) : (
            <label className="w-full h-full min-h-[300px] border-2 border-dashed border-zinc-300 dark:border-zinc-700/60 rounded-2xl flex flex-col items-center justify-center gap-4 cursor-pointer hover:bg-black/[0.02] dark:hover:bg-white/[0.02] hover:border-zinc-400 transition-all">
              <div className="w-16 h-16 rounded-full bg-white dark:bg-zinc-800 flex items-center justify-center shadow-sm">
                <UploadCloud size={28} className="text-zinc-400" />
              </div>
              <div className="text-center">
                <p className="text-zinc-700 dark:text-zinc-300 font-medium">Click to upload image</p>
                <p className="text-xs text-zinc-500 mt-1">Supports JPG, PNG</p>
              </div>
              <input type="file" accept="image/*" className="hidden" onChange={handleFileChange} />
            </label>
          )}
        </div>

        {/* 右侧控制与结果区 */}
        <div className="bg-white dark:bg-zinc-900/40 border border-black/[0.04] dark:border-white/[0.05] rounded-3xl p-6 shadow-sm flex flex-col h-full">
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-6">
            <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">Model Controls</h3>
            
            {/* 选项卡切换 */}
            <div className="flex items-center w-full sm:w-auto p-1 rounded-lg bg-black/5 dark:bg-white/5 border border-black/[0.04] dark:border-white/[0.05]">
              <button
                onClick={() => setActiveTab('basnet')}
                className={`flex-1 sm:flex-none text-center px-4 py-1.5 rounded-md text-xs font-medium transition-all ${
                  activeTab === 'basnet'
                    ? 'bg-white dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100 shadow-sm'
                    : 'text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-300'
                }`}
              >
                BASNet (V1)
              </button>
              <button
                onClick={() => setActiveTab('detr')}
                className={`flex-1 sm:flex-none text-center px-4 py-1.5 rounded-md text-xs font-medium transition-all ${
                  activeTab === 'detr'
                    ? 'bg-white dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100 shadow-sm'
                    : 'text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-300'
                }`}
              >
                DETR (V2)
              </button>
            </div>
          </div>

          {/* 控制面板 */}
          <div className="space-y-5 mb-6 pb-6 border-b border-black/[0.04] dark:border-white/[0.05]">
            <div className="space-y-3">
              <div className="flex justify-between text-sm">
                <span className="text-zinc-600 dark:text-zinc-400 flex items-center gap-1">
                  <Settings2 size={14} /> Threshold
                </span>
                <span className="text-zinc-900 dark:text-zinc-200 font-mono">{thresholds[activeTab]}</span>
              </div>
              <input 
                type="range" 
                min="0" 
                max="1" 
                step="0.01" 
                value={thresholds[activeTab]} 
                onChange={handleThresholdChange}
                className="w-full accent-indigo-500 bg-zinc-200 dark:bg-zinc-800 h-1.5 rounded-full appearance-none cursor-pointer" 
              />
            </div>
            
            <button 
              onClick={() => detectROI(activeTab)}
              disabled={!selectedFile || loading[activeTab]}
              className="w-full py-2.5 px-4 bg-indigo-600 hover:bg-indigo-700 disabled:bg-zinc-300 dark:disabled:bg-zinc-800 text-white disabled:text-zinc-500 rounded-xl font-medium text-sm flex items-center justify-center gap-2 transition-all"
            >
              {loading[activeTab] ? <Loader2 size={16} className="animate-spin" /> : <Focus size={16} />}
              {loading[activeTab] ? 'Processing...' : `Run ${activeTab.toUpperCase()}`}
            </button>
          </div>

          {/* 结果显示 */}
          <div className="flex-1 flex flex-col items-center justify-center bg-black/[0.02] dark:bg-white/[0.02] rounded-2xl p-4 border border-black/[0.02] dark:border-white/[0.02] min-h-[200px]">
            {loading[activeTab] ? (
              <div className="flex flex-col items-center justify-center text-zinc-500 gap-3">
                <Loader2 className="animate-spin" size={24} />
                <p className="text-sm">Detecting regions of interest...</p>
              </div>
            ) : results[activeTab] ? (
              <img src={results[activeTab]!} alt="ROI Result" className="max-h-[300px] object-contain rounded-xl shadow-sm" />
            ) : (
              <div className="flex flex-col items-center justify-center text-zinc-400">
                <ImageIcon size={32} className="mb-2 opacity-30" />
                <p className="text-sm text-center">Output will appear here<br/>after running the model</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}