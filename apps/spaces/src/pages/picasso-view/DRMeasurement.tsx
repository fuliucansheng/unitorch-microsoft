import { useState } from 'react';
import { UploadCloud, Image as ImageIcon, Loader2 } from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5000';

type DisplayMode = 'DR' | 'Others' | 'All';

interface AnalysisResults {
  dr?: {
    badCrop: number;
    badPad: number;
  };
  others?: {
    imageTypes: { poster: number; real: number; logo: number };
    blurryScore: number;
    backgroundTypes: { Complex: number; Simple: number; White: number };
    watermarkScore: number;
    aestheticScore: number;
    googleCategory: any[];
  };
}

export default function DRMeasurement() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  // 默认使用 DR 模式，减少初始请求量
  const [displayMode, setDisplayMode] = useState<DisplayMode>('DR');
  const [results, setResults] = useState<AnalysisResults | null>(null);

  const fetchAnalysis = async (file: File, mode: DisplayMode, currentResults: AnalysisResults | null) => {
    setIsLoading(true);
    const formData = new FormData();
    formData.append('image', file);

    const newResults: AnalysisResults = currentResults ? { ...currentResults } : {};
    const fetchPromises = [];

    try {
      // 如果需要显示 DR，并且 DR 数据尚未获取
      if ((mode === 'DR' || mode === 'All') && !newResults.dr) {
        fetchPromises.push(
          Promise.all([
            fetch(`${API_BASE}/microsoft/apps/fastapi/siglip2/generate1`, { method: 'POST', body: formData }).then(r => r.json()),
            fetch(`${API_BASE}/microsoft/apps/fastapi/siglip2/generate2`, { method: 'POST', body: formData }).then(r => r.json()),
          ]).then(([resSiglip1, resSiglip2]) => {
            newResults.dr = {
              badCrop: resSiglip1['Bad Cropped'],
              badPad: resSiglip2['Bad Padding']
            };
          })
        );
      }

      // 如果需要显示 Others，并且 Others 数据尚未获取
      if ((mode === 'Others' || mode === 'All') && !newResults.others) {
        fetchPromises.push(
          Promise.all([
            fetch(`${API_BASE}/microsoft/apps/fastapi/bletchley/v1/generate1`, { method: 'POST', body: formData }).then(r => r.json()),
            fetch(`${API_BASE}/microsoft/apps/fastapi/bletchley/v1/generate2`, { method: 'POST', body: formData }).then(r => r.json()),
            fetch(`${API_BASE}/microsoft/apps/fastapi/bletchley/v1/generate3`, { method: 'POST', body: formData }).then(r => r.json()),
            fetch(`${API_BASE}/microsoft/apps/fastapi/bletchley/v3/generate1`, { method: 'POST', body: formData }).then(r => r.json()),
            fetch(`${API_BASE}/microsoft/apps/fastapi/bletchley/v3/generate2`, { method: 'POST', body: formData }).then(r => r.json()),
            fetch(`${API_BASE}/microsoft/apps/fastapi/swin/googlecate/generate?topk=5`, { method: 'POST', body: formData }).then(r => r.json()),
          ]).then(([resV1Gen1, resV1Gen2, resV1Gen3, resV3Gen1, resV3Gen2, resGoogleCate]) => {
            newResults.others = {
              imageTypes: { poster: resV1Gen1.Poster, real: resV1Gen1.Real, logo: resV1Gen1.Logo },
              blurryScore: resV1Gen2.Blurry,
              backgroundTypes: resV1Gen3,
              watermarkScore: resV3Gen1.Watermark,
              aestheticScore: resV3Gen2['Bad Aesthetics'],
              googleCategory: resGoogleCate
            };
          })
        );
      }

      await Promise.all(fetchPromises);
      setResults(newResults);
    } catch (error) {
      console.error("API Error:", error);
      alert("Failed to analyze image. Ensure API server is running at " + API_BASE);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setSelectedFile(file);
    setPreview(URL.createObjectURL(file));
    setResults(null); // 清空上一次结果
    
    // 按当前模式获取对应数据
    fetchAnalysis(file, displayMode, null);
  };

  const handleModeChange = (mode: DisplayMode) => {
    setDisplayMode(mode);
    if (selectedFile) {
      // 切换模式时按需请求缺失的数据
      fetchAnalysis(selectedFile, mode, results);
    }
  };

  const renderProgress = (label: string, value: number = 0) => (
    <div key={label} className="mb-3">
      <div className="flex justify-between text-xs mb-1">
        <span className="text-zinc-600 dark:text-zinc-400 font-medium">{label}</span>
        <span className="text-zinc-900 dark:text-zinc-200 font-mono">{(value * 100).toFixed(1)}%</span>
      </div>
      <div className="w-full bg-black/5 dark:bg-white/10 rounded-full h-1.5 overflow-hidden">
        <div 
          className="bg-indigo-500 h-1.5 rounded-full transition-all duration-500" 
          style={{ width: `${Math.min(100, Math.max(0, value * 100))}%` }}
        />
      </div>
    </div>
  );

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <div>
        <h2 className="text-3xl font-bold text-zinc-900 dark:text-zinc-100 tracking-tight">DR Measurement</h2>
        <p className="text-zinc-500 dark:text-zinc-400 mt-2 font-light">Upload an image to analyze its quality, type, and categories.</p>
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

        {/* 右侧结果区 */}
        <div className="bg-white dark:bg-zinc-900/40 border border-black/[0.04] dark:border-white/[0.05] rounded-3xl p-6 shadow-sm flex flex-col">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">Analysis Results</h3>
            
            {/* 选项卡切换 */}
            <div className="flex items-center p-1 rounded-lg bg-black/5 dark:bg-white/5 border border-black/[0.04] dark:border-white/[0.05]">
              {(['DR', 'Others', 'All'] as DisplayMode[]).map((mode) => (
                <button
                  key={mode}
                  onClick={() => handleModeChange(mode)}
                  className={`px-4 py-1.5 rounded-md text-xs font-medium transition-all ${
                    displayMode === mode
                      ? 'bg-white dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100 shadow-sm'
                      : 'text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-300'
                  }`}
                >
                  {mode}
                </button>
              ))}
            </div>
          </div>

          <div className="flex-1 overflow-y-auto pr-2 custom-scrollbar">
            {isLoading ? (
              <div className="h-full flex flex-col items-center justify-center text-zinc-500 gap-3">
                <Loader2 className="animate-spin" size={24} />
                <p className="text-sm">Analyzing image...</p>
              </div>
            ) : !results ? (
              <div className="h-full flex flex-col items-center justify-center text-zinc-400">
                <ImageIcon size={32} className="mb-2 opacity-50" />
                <p className="text-sm">Upload an image to see results</p>
              </div>
            ) : (
              <div className="space-y-8">
                {/* DR 相关信息 */}
                {(displayMode === 'DR' || displayMode === 'All') && results.dr && (
                  <div className="space-y-4">
                    <h4 className="text-sm font-semibold text-indigo-600 dark:text-indigo-400 uppercase tracking-wider border-b border-black/5 dark:border-white/5 pb-2">DR Measurements</h4>
                    <div>
                      {renderProgress('Bad Crop Score', results.dr.badCrop)}
                      {renderProgress('Bad Pad Score', results.dr.badPad)}
                    </div>
                  </div>
                )}

                {/* Others 相关信息 */}
                {(displayMode === 'Others' || displayMode === 'All') && results.others && (
                  <div className="space-y-6">
                    <div className="space-y-4">
                      <h4 className="text-sm font-semibold text-indigo-600 dark:text-indigo-400 uppercase tracking-wider border-b border-black/5 dark:border-white/5 pb-2">Image Quality & Properties</h4>
                      <div className="grid grid-cols-2 gap-x-6 gap-y-2">
                        <div className="col-span-2">{renderProgress('Blurry Score', results.others.blurryScore)}</div>
                        <div className="col-span-2">{renderProgress('Watermark Score', results.others.watermarkScore)}</div>
                        <div className="col-span-2">{renderProgress('Bad Aesthetic Score', results.others.aestheticScore)}</div>
                      </div>
                    </div>

                    <div className="space-y-4">
                      <h4 className="text-xs font-medium text-zinc-500 uppercase tracking-wider">Image Types</h4>
                      <div className="grid grid-cols-2 gap-x-6">
                        {renderProgress('Poster', results.others.imageTypes.poster)}
                        {renderProgress('Real', results.others.imageTypes.real)}
                        {renderProgress('Logo', results.others.imageTypes.logo)}
                      </div>
                    </div>

                    <div className="space-y-4">
                      <h4 className="text-xs font-medium text-zinc-500 uppercase tracking-wider">Background Types</h4>
                      <div className="grid grid-cols-2 gap-x-6">
                        {renderProgress('Complex', results.others.backgroundTypes.Complex)}
                        {renderProgress('Simple', results.others.backgroundTypes.Simple)}
                        {renderProgress('White', results.others.backgroundTypes.White)}
                      </div>
                    </div>

                    {results.others.googleCategory && results.others.googleCategory.length > 0 && (
                      <div className="space-y-3">
                        <h4 className="text-xs font-medium text-zinc-500 uppercase tracking-wider border-t border-black/5 dark:border-white/5 pt-4">Google Categories</h4>
                        <ul className="space-y-2">
                          {results.others.googleCategory.map((cat: any, i: number) => (
                            <li key={i} className="text-xs flex flex-col bg-black/[0.02] dark:bg-white/[0.02] p-2.5 rounded-lg">
                              <span className="text-zinc-700 dark:text-zinc-300 mb-1 leading-relaxed">{cat.category}</span>
                              <span className="text-zinc-500 font-mono text-[10px]">Score: {(cat.score * 100).toFixed(2)}%</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}