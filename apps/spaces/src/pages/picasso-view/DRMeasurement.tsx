import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { UploadCloud, Image as ImageIcon, Loader2, Info, ArrowLeft } from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5000';

type DisplayMode = 'DR' | 'Category' | 'Others' | 'All';

interface AnalysisResults {
  dr?: {
    badCrop: number;
    badPad: number;
  };
  others?: {
    blurryScore: number;
    backgroundTypes: { Complex: number; Simple: number; White: number };
    watermarkScore: number;
  };
  category?: {
    googleCategory: any[];
  };
}

export default function DRMeasurement() {
  const navigate = useNavigate();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [displayMode, setDisplayMode] = useState<DisplayMode>('DR');
  const [results, setResults] = useState<AnalysisResults | null>(null);

  const fetchAnalysis = async (file: File, mode: DisplayMode, currentResults: AnalysisResults | null) => {
    setIsLoading(true);
    const formData = new FormData();
    formData.append('image', file);

    const newResults: AnalysisResults = currentResults ? { ...currentResults } : {};
    const fetchPromises = [];

    try {
      // DR endpoints
      if ((mode === 'DR' || mode === 'All') && !newResults.dr) {
        fetchPromises.push(
          Promise.all([
            fetch(`${API_BASE}/microsoft/apps/spaces/picasso/siglip2/generate1`, { method: 'POST', body: formData }).then(r => r.json()),
            fetch(`${API_BASE}/microsoft/apps/spaces/picasso/siglip2/generate2`, { method: 'POST', body: formData }).then(r => r.json()),
          ]).then(([resSiglip1, resSiglip2]) => {
            newResults.dr = {
              badCrop: resSiglip1['Bad Cropped'],
              badPad: resSiglip2['Bad Padding']
            };
          })
        );
      }

      // Others endpoints
      if ((mode === 'Others' || mode === 'All') && !newResults.others) {
        fetchPromises.push(
          Promise.all([
            fetch(`${API_BASE}/microsoft/apps/spaces/picasso/bletchley/v1/generate1`, { method: 'POST', body: formData }).then(r => r.json()),
            fetch(`${API_BASE}/microsoft/apps/spaces/picasso/bletchley/v1/generate2`, { method: 'POST', body: formData }).then(r => r.json()),
            fetch(`${API_BASE}/microsoft/apps/spaces/picasso/bletchley/v3/generate1`, { method: 'POST', body: formData }).then(r => r.json()),
          ]).then(([resBlurry, resBg, resWatermark]) => {
            newResults.others = {
              blurryScore: resBlurry.Blurry,
              backgroundTypes: resBg,
              watermarkScore: resWatermark.Watermark
            };
          })
        );
      }

      // Category endpoint
      if ((mode === 'Category' || mode === 'All') && !newResults.category) {
        fetchPromises.push(
          fetch(`${API_BASE}/microsoft/apps/spaces/picasso/swin/googlecate/generate?topk=5`, { method: 'POST', body: formData })
            .then(r => r.json())
            .then(resGoogleCate => {
              newResults.category = {
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
    setResults(null);
    
    fetchAnalysis(file, displayMode, null);
  };

  const handleModeChange = (mode: DisplayMode) => {
    setDisplayMode(mode);
    if (selectedFile) {
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
      <div className="flex items-start sm:items-center gap-4">
        <button 
          onClick={() => navigate('/picasso/examples')}
          className="p-2.5 rounded-full bg-white dark:bg-zinc-900/60 border border-black/[0.08] dark:border-white/[0.05] shadow-sm hover:bg-zinc-50 dark:hover:bg-zinc-800 transition-colors text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-100 shrink-0"
        >
          <ArrowLeft size={18} />
        </button>
        <div>
          <h2 className="text-3xl font-bold text-zinc-900 dark:text-zinc-100 tracking-tight">DR Measurement</h2>
          <p className="text-zinc-500 dark:text-zinc-400 mt-1.5 font-light">Upload an image to analyze its quality, type, and categories.</p>
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

        {/* 右侧整体容器 */}
        <div className="flex flex-col gap-4 h-full">
          {/* 结果区 */}
          <div className="bg-white dark:bg-zinc-900/40 border border-black/[0.04] dark:border-white/[0.05] rounded-3xl p-6 shadow-sm flex flex-col flex-1">
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-6">
              <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">Analysis Results</h3>
              
              {/* 选项卡切换 */}
              <div className="flex items-center w-full sm:w-auto p-1 rounded-lg bg-black/5 dark:bg-white/5 border border-black/[0.04] dark:border-white/[0.05]">
                {(['DR', 'Category', 'Others', 'All'] as DisplayMode[]).map((mode) => (
                  <button
                    key={mode}
                    onClick={() => handleModeChange(mode)}
                    className={`flex-1 sm:flex-none text-center px-2 sm:px-4 py-1.5 rounded-md text-xs font-medium transition-all ${
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
                <div className="space-y-6">
                  {/* DR 相关信息 */}
                  {(displayMode === 'DR' || displayMode === 'All') && results.dr && (
                    <div className="space-y-5">
                      <div>
                        {renderProgress('Bad Crop Score', results.dr.badCrop)}
                        {renderProgress('Bad Pad Score', results.dr.badPad)}
                      </div>
                    </div>
                  )}

                  {/* Category 相关信息 */}
                  {(displayMode === 'Category' || displayMode === 'All') && results.category && results.category.googleCategory && results.category.googleCategory.length > 0 && (
                    <div className="space-y-4">
                      {displayMode === 'All' && <h4 className="text-sm font-medium text-zinc-900 dark:text-zinc-100 border-b border-zinc-100 dark:border-zinc-800 pb-2">Google Categories</h4>}
                      <ul className="space-y-2">
                        {results.category.googleCategory.map((cat: any, i: number) => (
                          <li key={i} className="text-xs flex flex-col bg-black/[0.02] dark:bg-white/[0.02] p-2.5 rounded-lg">
                            <span className="text-zinc-700 dark:text-zinc-300 mb-1 leading-relaxed">{cat.category}</span>
                            <span className="text-zinc-500 font-mono text-[10px]">Score: {(cat.score * 100).toFixed(2)}%</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Others 相关信息 */}
                  {(displayMode === 'Others' || displayMode === 'All') && results.others && (
                    <div className="space-y-5">
                      {displayMode === 'All' && <h4 className="text-sm font-medium text-zinc-900 dark:text-zinc-100 border-b border-zinc-100 dark:border-zinc-800 pb-2">Other Metrics</h4>}
                      
                      <div>
                        {renderProgress('Blurry Score', results.others.blurryScore)}
                        {renderProgress('Watermark Score', results.others.watermarkScore)}
                      </div>

                      <div className="pt-2 border-t border-black/[0.04] dark:border-white/[0.05]">
                        <h5 className="text-xs font-semibold text-zinc-500 dark:text-zinc-400 uppercase tracking-wider mb-4 mt-2">Background Scores</h5>
                        <div>
                          {renderProgress('Complex', results.others.backgroundTypes.Complex)}
                          {renderProgress('Simple', results.others.backgroundTypes.Simple)}
                          {renderProgress('White', results.others.backgroundTypes.White)}
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* 独立在外的 Evaluation Criteria，在 DR 或 All tab 下展示 */}
          {(displayMode === 'DR' || displayMode === 'All') && (
            <div className="overflow-hidden rounded-3xl border border-indigo-100/60 dark:border-indigo-500/10 bg-gradient-to-br from-indigo-50/50 to-white dark:from-indigo-500/5 dark:to-zinc-900/40 p-5 shadow-sm shrink-0">
              <div className="flex items-center gap-2 mb-3 pb-3 border-b border-indigo-100/60 dark:border-indigo-500/10">
                <Info size={16} className="text-indigo-500 dark:text-indigo-400" />
                <span className="text-sm font-semibold text-indigo-900 dark:text-indigo-300 uppercase tracking-wider">Evaluation Criteria</span>
              </div>
              
              <div className="space-y-3 w-full text-[13px]">
                <div className="flex flex-col sm:flex-row sm:items-start gap-1 sm:gap-3">
                  <span className="font-semibold text-zinc-900 dark:text-zinc-100 whitespace-nowrap min-w-[70px]">Bad Crop</span>
                  <span className="text-zinc-500 dark:text-zinc-400 leading-relaxed">Indicates poorly cropped images (scores ≥ 45% are considered bad).</span>
                </div>
                
                <div className="flex flex-col sm:flex-row sm:items-start gap-1 sm:gap-3">
                  <span className="font-semibold text-zinc-900 dark:text-zinc-100 whitespace-nowrap min-w-[70px]">Bad Pad</span>
                  <span className="text-zinc-500 dark:text-zinc-400 leading-relaxed">Indicates excessive padding (scores ≥ 50% are considered bad).</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}