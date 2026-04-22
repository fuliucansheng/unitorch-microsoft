import { useState } from 'react';
import { UploadCloud, Wand2, Image as ImageIcon, Loader2, Sparkles, Edit3, X, Plus, ArrowLeft, Frame, Menu, PanelLeftClose, PanelLeftOpen, Layers } from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5000';

type ViewMode = 'list' | 'generation' | 'editing';
type ModelType = 'gpt' | 'gemini';

const SIZE_OPTIONS = [
  { value: '1024x1024', label: '1024 × 1024 (1:1)' },
  { value: '1536x1024', label: '1536 × 1024 (3:2)' },
  { value: '1024x1536', label: '1024 × 1536 (2:3)' }
];

const Demos = () => {
  const [view, setView] = useState<ViewMode>('list');
  const [activeModel, setActiveModel] = useState<ModelType>('gpt');
  
  // Sidebar states
  const [isDesktopCollapsed, setIsDesktopCollapsed] = useState(false);
  const [isMobileOpen, setIsMobileOpen] = useState(false);
  
  // Inputs
  const [genPrompt, setGenPrompt] = useState('');
  const [editPrompt, setEditPrompt] = useState('');
  
  // Sizes
  const [genSize, setGenSize] = useState('1024x1024');
  const [editSize, setEditSize] = useState('1024x1024');
  
  // Files
  const [editFiles, setEditFiles] = useState<File[]>([]);
  const [editPreviews, setEditPreviews] = useState<string[]>([]);

  // States
  const [loading, setLoading] = useState({
    gpt: { generation: false, editing: false },
    gemini: { generation: false, editing: false }
  });
  
  const [results, setResults] = useState<{
    gpt: { generation: string | null; editing: string | null };
    gemini: { generation: string | null; editing: string | null };
  }>({
    gpt: { generation: null, editing: null },
    gemini: { generation: null, editing: null }
  });

  const handleViewChange = (newView: ViewMode) => {
    setView(newView);
    setIsMobileOpen(false);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const newFiles = Array.from(e.target.files);
      const totalFiles = [...editFiles, ...newFiles].slice(0, 5);
      
      setEditFiles(totalFiles);
      editPreviews.forEach(p => URL.revokeObjectURL(p));
      setEditPreviews(totalFiles.map(f => URL.createObjectURL(f)));
      
      setResults(prev => ({
        ...prev,
        [activeModel]: { ...prev[activeModel], editing: null }
      }));
    }
  };

  const removeFile = (index: number) => {
    const newFiles = [...editFiles];
    newFiles.splice(index, 1);
    setEditFiles(newFiles);

    const newPreviews = [...editPreviews];
    URL.revokeObjectURL(newPreviews[index]);
    newPreviews.splice(index, 1);
    setEditPreviews(newPreviews);
  };

  const handleRun = async (mode: 'generation' | 'editing') => {
    const prompt = mode === 'generation' ? genPrompt : editPrompt;
    const size = mode === 'generation' ? genSize : editSize;
    
    if (!prompt.trim()) {
      alert("Please enter a prompt first.");
      return;
    }
    if (mode === 'editing' && editFiles.length === 0) {
      alert("Please upload at least one image for editing.");
      return;
    }

    setLoading(prev => ({
      ...prev,
      [activeModel]: { ...prev[activeModel], [mode]: true }
    }));

    try {
      const queryParams = new URLSearchParams({
        prompt: prompt,
        size: size,
        background: 'transparent'
      }).toString();

      let endpoint = '';
      const options: RequestInit = {
        method: 'POST',
        headers: { 'accept': 'application/json' }
      };

      if (mode === 'generation') {
        endpoint = activeModel === 'gpt' 
          ? '/microsoft/apps/spaces/gpt/image-15/generate'
          : '/microsoft/apps/spaces/gemini/image/generate';
        options.body = '';
      } else {
        endpoint = activeModel === 'gpt'
          ? '/microsoft/apps/spaces/gpt/image-15/edit'
          : '/microsoft/apps/spaces/gemini/image/edit';

        const formData = new FormData();
        editFiles.forEach(file => formData.append('images', file));
        if (activeModel === 'gpt') formData.append('mask', '');
        
        options.body = formData;
      }

      const response = await fetch(`${API_BASE}${endpoint}?${queryParams}`, options);
      if (!response.ok) throw new Error(`API returned ${response.status}`);

      const blob = await response.blob();
      const imageUrl = URL.createObjectURL(blob);
      
      setResults(prev => ({
        ...prev,
        [activeModel]: { ...prev[activeModel], [mode]: imageUrl }
      }));
    } catch (error) {
      console.error(`API Error (${activeModel}):`, error);
      alert(`Failed to process with ${activeModel.toUpperCase()}.`);
    } finally {
      setLoading(prev => ({
        ...prev,
        [activeModel]: { ...prev[activeModel], [mode]: false }
      }));
    }
  };

  // ================= Components =================

  const renderWorkspaceHeader = (title: string, Icon: any, colorClass: string) => (
    <div className="flex flex-wrap items-center justify-between gap-4 mb-6 lg:mb-8">
      <div className="flex items-center gap-3 sm:gap-4 shrink-0 max-w-full">
        <button 
          onClick={() => handleViewChange('list')}
          className="p-2 sm:p-2.5 rounded-full bg-white dark:bg-zinc-900/60 border border-black/[0.08] dark:border-white/[0.05] shadow-sm hover:bg-zinc-50 dark:hover:bg-zinc-800 transition-colors text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-100 shrink-0"
        >
          <ArrowLeft size={18} />
        </button>
        <div className="flex items-center gap-2 min-w-0">
          <Icon size={24} className={`shrink-0 ${colorClass}`} />
          <h2 className="text-xl sm:text-2xl font-bold text-zinc-900 dark:text-zinc-100 tracking-tight truncate">{title}</h2>
        </div>
      </div>

      <div className="flex p-1 rounded-xl bg-black/5 dark:bg-white/5 border border-black/[0.04] dark:border-white/[0.05] w-full md:w-auto overflow-x-auto custom-scrollbar">
        <button
          onClick={() => setActiveModel('gpt')}
          className={`flex-1 md:flex-none flex items-center justify-center gap-2 px-3 sm:px-4 py-1.5 rounded-lg text-sm font-medium transition-all whitespace-nowrap ${
            activeModel === 'gpt'
              ? 'bg-white dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100 shadow-sm'
              : 'text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-300'
          }`}
        >
          <div className={`w-4 h-4 rounded-full flex items-center justify-center text-[10px] shrink-0 ${activeModel === 'gpt' ? 'bg-black/5 dark:bg-white/10' : 'bg-black/5 dark:bg-white/5'}`}>G</div>
          GPT-Image-1.5
        </button>
        <button
          onClick={() => setActiveModel('gemini')}
          className={`flex-1 md:flex-none flex items-center justify-center gap-2 px-3 sm:px-4 py-1.5 rounded-lg text-sm font-medium transition-all whitespace-nowrap ${
            activeModel === 'gemini'
              ? 'bg-white dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100 shadow-sm'
              : 'text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-300'
          }`}
        >
          <div className={`w-4 h-4 rounded-full flex items-center justify-center text-[10px] shrink-0 ${activeModel === 'gemini' ? 'bg-black/5 dark:bg-white/10' : 'bg-black/5 dark:bg-white/5'}`}>✦</div>
          Gemini-3-Pro
        </button>
      </div>
    </div>
  );

  const renderList = () => (
    <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div>
        <h2 className="text-3xl font-bold text-zinc-900 dark:text-zinc-100 tracking-tight">Workspaces</h2>
        <p className="text-zinc-500 dark:text-zinc-400 mt-2 font-light">Explore generation and editing capabilities across different models.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div 
          onClick={() => handleViewChange('generation')}
          className="group bg-white dark:bg-zinc-900/40 border border-black/[0.04] dark:border-white/[0.05] rounded-3xl p-7 shadow-[0_2px_20px_rgba(0,0,0,0.02)] hover:shadow-[0_8px_30px_rgba(0,0,0,0.06)] hover:-translate-y-1 transition-all duration-300 cursor-pointer flex flex-col h-full"
        >
          <div className="flex items-center gap-2 mb-3">
            <Sparkles size={20} className="text-indigo-500" />
            <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100 group-hover:text-indigo-600 dark:group-hover:text-indigo-400 transition-colors tracking-tight">
              Image Generation
            </h3>
          </div>
          <p className="text-sm text-zinc-500 dark:text-zinc-400 leading-relaxed flex-1 font-light">
            Create stunning images from text prompts using GPT-Image-1.5 or Gemini-3-Pro.
          </p>
          <div className="mt-8 pt-5 border-t border-black/[0.03] dark:border-white/[0.05] flex items-center justify-between">
            <span className="text-xs font-semibold text-indigo-600 dark:text-indigo-400 uppercase tracking-wider">Enter Workspace</span>
            <span className="text-zinc-300 dark:text-zinc-600 group-hover:text-indigo-600 dark:group-hover:text-indigo-400 group-hover:translate-x-1 transition-all">→</span>
          </div>
        </div>

        <div 
          onClick={() => handleViewChange('editing')}
          className="group bg-white dark:bg-zinc-900/40 border border-black/[0.04] dark:border-white/[0.05] rounded-3xl p-7 shadow-[0_2px_20px_rgba(0,0,0,0.02)] hover:shadow-[0_8px_30px_rgba(0,0,0,0.06)] hover:-translate-y-1 transition-all duration-300 cursor-pointer flex flex-col h-full"
        >
          <div className="flex items-center gap-2 mb-3">
            <Edit3 size={20} className="text-emerald-500" />
            <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100 group-hover:text-emerald-600 dark:group-hover:text-emerald-400 transition-colors tracking-tight">
              Image Editing
            </h3>
          </div>
          <p className="text-sm text-zinc-500 dark:text-zinc-400 leading-relaxed flex-1 font-light">
            Upload reference images and modify them with text instructions using our advanced models.
          </p>
          <div className="mt-8 pt-5 border-t border-black/[0.03] dark:border-white/[0.05] flex items-center justify-between">
            <span className="text-xs font-semibold text-emerald-600 dark:text-emerald-400 uppercase tracking-wider">Enter Workspace</span>
            <span className="text-zinc-300 dark:text-zinc-600 group-hover:text-emerald-600 dark:group-hover:text-emerald-400 group-hover:translate-x-1 transition-all">→</span>
          </div>
        </div>
      </div>
    </div>
  );

  const renderGeneration = () => (
    <div className="animate-in fade-in zoom-in-95 duration-300 h-full flex flex-col">
      {renderWorkspaceHeader('Image Generation', Sparkles, 'text-indigo-500')}
      
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 flex-1">
        <div className="lg:col-span-4 bg-white dark:bg-zinc-900/40 border border-black/[0.04] dark:border-white/[0.05] rounded-3xl p-6 shadow-sm flex flex-col gap-6">
          <div className="space-y-3">
            <span className="text-sm font-medium text-zinc-700 dark:text-zinc-300 flex items-center gap-2">
              <Frame size={16} className="text-zinc-400" /> Size
            </span>
            <select
              value={genSize}
              onChange={(e) => setGenSize(e.target.value)}
              className="w-full bg-black/[0.02] dark:bg-white/[0.02] border border-black/[0.04] dark:border-white/[0.05] rounded-xl p-3 text-sm text-zinc-900 dark:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-indigo-500/30 transition-all cursor-pointer"
            >
              {SIZE_OPTIONS.map(opt => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>
          </div>

          <div className="space-y-3">
            <span className="text-sm font-medium text-zinc-700 dark:text-zinc-300 flex items-center gap-2">
              <Wand2 size={16} className="text-zinc-400" /> Prompt
            </span>
            <textarea 
              value={genPrompt}
              onChange={(e) => setGenPrompt(e.target.value)}
              placeholder="A futuristic city skyline at sunset..."
              className="w-full h-32 bg-black/[0.02] dark:bg-white/[0.02] border border-black/[0.04] dark:border-white/[0.05] rounded-2xl p-4 text-sm text-zinc-900 dark:text-zinc-100 placeholder:text-zinc-400 focus:outline-none focus:ring-2 focus:ring-indigo-500/30 transition-all resize-none"
            />
          </div>
          <button 
            onClick={() => handleRun('generation')}
            disabled={loading[activeModel].generation || !genPrompt.trim()}
            className="w-full py-3 bg-indigo-600 hover:bg-indigo-700 text-white disabled:bg-zinc-300 dark:disabled:bg-zinc-800 disabled:text-zinc-500 rounded-xl font-medium text-sm flex items-center justify-center gap-2 transition-all shadow-sm mt-auto"
          >
            {loading[activeModel].generation ? <Loader2 size={16} className="animate-spin" /> : <Sparkles size={16} />}
            {loading[activeModel].generation ? 'Generating...' : `Run on ${activeModel === 'gpt' ? 'GPT-1.5' : 'Gemini-3'}`}
          </button>
        </div>

        <div className="lg:col-span-8 bg-white dark:bg-zinc-900/40 border border-black/[0.04] dark:border-white/[0.05] rounded-3xl p-6 shadow-sm flex flex-col min-h-[400px]">
          <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100 mb-4 flex items-center gap-2 flex-wrap">
            Result Output <span className="text-xs px-2 py-1 bg-black/5 dark:bg-white/10 rounded-md font-normal text-zinc-500 whitespace-nowrap">{activeModel === 'gpt' ? 'GPT-Image-1.5' : 'Gemini-3-Pro'}</span>
          </h3>
          <div className="flex-1 flex flex-col items-center justify-center bg-black/[0.02] dark:bg-white/[0.02] rounded-2xl p-4 border border-black/[0.02] dark:border-white/[0.02]">
            {loading[activeModel].generation ? (
              <div className="flex flex-col items-center text-zinc-400 gap-3">
                <Loader2 size={32} className="animate-spin text-indigo-500" />
                <span className="text-sm">Creating image...</span>
              </div>
            ) : results[activeModel].generation ? (
              <img src={results[activeModel].generation!} alt="Generated" className="max-h-[500px] w-full object-contain rounded-xl" />
            ) : (
              <div className="flex flex-col items-center text-zinc-400 opacity-50">
                <ImageIcon size={40} className="mb-3" />
                <span className="text-sm">Image will appear here</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );

  const renderEditing = () => (
    <div className="animate-in fade-in zoom-in-95 duration-300 h-full flex flex-col">
      {renderWorkspaceHeader('Image Editing', Edit3, 'text-emerald-500')}
      
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 flex-1">
        <div className="lg:col-span-4 bg-white dark:bg-zinc-900/40 border border-black/[0.04] dark:border-white/[0.05] rounded-3xl p-6 shadow-sm flex flex-col gap-6">
          
          <div>
            <label className="text-sm font-medium text-zinc-700 dark:text-zinc-300 flex items-center justify-between mb-2">
              <span className="flex items-center gap-2"><ImageIcon size={16} className="text-zinc-400" /> Reference Images</span>
              <span className="text-xs text-zinc-400">{editFiles.length}/5</span>
            </label>
            
            {editPreviews.length > 0 ? (
              <div className="grid grid-cols-3 gap-3">
                {editPreviews.map((src, idx) => (
                  <div key={idx} className="relative aspect-square rounded-xl overflow-hidden border border-zinc-200 dark:border-zinc-800 group bg-black/5 dark:bg-white/5 flex items-center justify-center">
                    <img src={src} alt={`Preview ${idx + 1}`} className="w-full h-full object-contain p-1" />
                    <button 
                      onClick={() => removeFile(idx)}
                      className="absolute top-1.5 right-1.5 w-6 h-6 bg-black/50 hover:bg-red-500 text-white flex items-center justify-center rounded-full opacity-0 group-hover:opacity-100 transition-all shadow-sm backdrop-blur-sm"
                    >
                      <X size={12} />
                    </button>
                  </div>
                ))}
                {editPreviews.length < 5 && (
                  <label className="aspect-square border-2 border-dashed border-zinc-300 dark:border-zinc-700/60 rounded-xl flex flex-col items-center justify-center gap-2 cursor-pointer hover:bg-black/[0.02] dark:hover:bg-white/[0.02] hover:border-zinc-400 transition-all">
                    <Plus size={20} className="text-zinc-400" />
                    <span className="text-[10px] text-zinc-500 font-medium">Add More</span>
                    <input type="file" accept="image/*" multiple className="hidden" onChange={handleFileChange} />
                  </label>
                )}
              </div>
            ) : (
              <label className="w-full aspect-video border-2 border-dashed border-zinc-300 dark:border-zinc-700/60 rounded-2xl flex flex-col items-center justify-center gap-4 cursor-pointer hover:bg-black/[0.02] dark:hover:bg-white/[0.02] hover:border-zinc-400 transition-all">
                <div className="w-12 h-12 rounded-full bg-white dark:bg-zinc-800 flex items-center justify-center shadow-sm">
                  <UploadCloud size={24} className="text-zinc-400" />
                </div>
                <div className="text-center">
                  <p className="text-sm text-zinc-700 dark:text-zinc-300 font-medium">Click to upload images</p>
                  <p className="text-xs text-zinc-500 mt-1">Up to 5 images (JPG, PNG)</p>
                </div>
                <input type="file" accept="image/*" multiple className="hidden" onChange={handleFileChange} />
              </label>
            )}
          </div>

          <div className="space-y-3">
            <span className="text-sm font-medium text-zinc-700 dark:text-zinc-300 flex items-center gap-2">
              <Frame size={16} className="text-zinc-400" /> Size
            </span>
            <select
              value={editSize}
              onChange={(e) => setEditSize(e.target.value)}
              className="w-full bg-black/[0.02] dark:bg-white/[0.02] border border-black/[0.04] dark:border-white/[0.05] rounded-xl p-3 text-sm text-zinc-900 dark:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-emerald-500/30 transition-all cursor-pointer"
            >
              {SIZE_OPTIONS.map(opt => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>
          </div>

          <div className="space-y-3">
            <span className="text-sm font-medium text-zinc-700 dark:text-zinc-300 flex items-center gap-2">
              <Wand2 size={16} className="text-zinc-400" /> Prompt
            </span>
            <textarea 
              value={editPrompt}
              onChange={(e) => setEditPrompt(e.target.value)}
              placeholder="Put the first logo on the top right corner..."
              className="w-full h-24 bg-black/[0.02] dark:bg-white/[0.02] border border-black/[0.04] dark:border-white/[0.05] rounded-2xl p-4 text-sm text-zinc-900 dark:text-zinc-100 placeholder:text-zinc-400 focus:outline-none focus:ring-2 focus:ring-emerald-500/30 transition-all resize-none"
            />
          </div>
          
          <button 
            onClick={() => handleRun('editing')}
            disabled={loading[activeModel].editing || !editPrompt.trim() || editFiles.length === 0}
            className="w-full py-3 bg-emerald-600 hover:bg-emerald-700 text-white disabled:bg-zinc-300 dark:disabled:bg-zinc-800 disabled:text-zinc-500 rounded-xl font-medium text-sm flex items-center justify-center gap-2 transition-all shadow-sm mt-auto"
          >
            {loading[activeModel].editing ? <Loader2 size={16} className="animate-spin" /> : <Edit3 size={16} />}
            {loading[activeModel].editing ? 'Editing...' : `Run on ${activeModel === 'gpt' ? 'GPT-1.5' : 'Gemini-3'}`}
          </button>
        </div>

        <div className="lg:col-span-8 bg-white dark:bg-zinc-900/40 border border-black/[0.04] dark:border-white/[0.05] rounded-3xl p-6 shadow-sm flex flex-col min-h-[400px]">
          <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100 mb-4 flex items-center gap-2 flex-wrap">
            Result Output <span className="text-xs px-2 py-1 bg-black/5 dark:bg-white/10 rounded-md font-normal text-zinc-500 whitespace-nowrap">{activeModel === 'gpt' ? 'GPT-Image-1.5' : 'Gemini-3-Pro'}</span>
          </h3>
          <div className="flex-1 flex flex-col items-center justify-center bg-black/[0.02] dark:bg-white/[0.02] rounded-2xl p-4 border border-black/[0.02] dark:border-white/[0.02]">
            {loading[activeModel].editing ? (
              <div className="flex flex-col items-center text-zinc-400 gap-3">
                <Loader2 size={32} className="animate-spin text-emerald-500" />
                <span className="text-sm">Processing image...</span>
              </div>
            ) : results[activeModel].editing ? (
              <img src={results[activeModel].editing!} alt="Edited" className="max-h-[500px] w-full object-contain rounded-xl" />
            ) : (
              <div className="flex flex-col items-center text-zinc-400 opacity-50">
                <ImageIcon size={40} className="mb-3" />
                <span className="text-sm">Image will appear here</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="flex flex-1 h-full relative overflow-hidden bg-transparent">
      {/* ================= 移动端半透明遮罩 ================= */}
      {isMobileOpen && (
        <div 
          className="md:hidden absolute inset-0 bg-zinc-900/20 dark:bg-black/40 backdrop-blur-sm z-30 transition-opacity"
          onClick={() => setIsMobileOpen(false)}
        />
      )}

      {/* ================= 移动端唤出按钮 ================= */}
      {!isMobileOpen && (
        <button
          onClick={() => setIsMobileOpen(true)}
          className="md:hidden absolute top-4 left-0 z-20 py-2 pl-2 pr-3 bg-white/80 dark:bg-zinc-900/80 backdrop-blur-md border border-l-0 border-black/[0.08] dark:border-white/[0.08] rounded-r-xl shadow-sm text-zinc-600 dark:text-zinc-300 hover:text-zinc-900 dark:hover:text-zinc-100 transition-colors"
        >
          <Menu size={20} />
        </button>
      )}

      {/* ================= 侧边栏 (桌面端 Static / 移动端 Drawer) ================= */}
      <aside 
        className={`
          absolute md:relative z-40 h-full flex flex-col flex-shrink-0 
          border-r border-black/[0.04] dark:border-white/[0.05] 
          bg-white/95 md:bg-white/40 dark:bg-zinc-950/95 md:dark:bg-zinc-950/30 md:backdrop-blur-2xl backdrop-blur-3xl
          transition-all duration-300 ease-in-out p-4 shadow-2xl md:shadow-none
          ${isMobileOpen ? 'translate-x-0 w-64' : '-translate-x-full w-64 md:translate-x-0'}
          ${isDesktopCollapsed ? 'md:w-[76px] md:items-center' : 'md:w-64 md:items-stretch'}
        `}
      >
        {/* 顶部标题与控制按钮 */}
        <div className={`flex items-center mb-4 transition-all duration-300 ${isDesktopCollapsed ? 'md:justify-center px-0' : 'justify-between px-2'}`}>
          <div 
            className={`text-xs font-semibold text-zinc-400 dark:text-zinc-500 uppercase tracking-wider transition-all duration-300 overflow-hidden whitespace-nowrap ${
              isDesktopCollapsed ? 'md:max-w-0 md:opacity-0' : 'max-w-[150px] opacity-100'
            }`}
          >
            Demos Menu
          </div>
          
          <button 
            onClick={() => setIsDesktopCollapsed(!isDesktopCollapsed)}
            className="hidden md:flex items-center justify-center w-8 h-8 rounded-lg text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-100 hover:bg-black/5 dark:hover:bg-white/10 transition-colors shrink-0"
            title={isDesktopCollapsed ? "Expand Menu" : "Collapse Menu"}
          >
            {isDesktopCollapsed ? <PanelLeftOpen size={18} /> : <PanelLeftClose size={18} />}
          </button>

          <button 
            onClick={() => setIsMobileOpen(false)}
            className="md:hidden flex items-center justify-center w-8 h-8 rounded-lg text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-100 hover:bg-black/5 dark:hover:bg-white/10 transition-colors shrink-0"
          >
            <X size={20} />
          </button>
        </div>

        {/* Workspaces 菜单组 */}
        <div className={`flex flex-col gap-1 mt-1 group/spaces ${isDesktopCollapsed ? 'md:items-center' : ''}`}>
          <button 
            onClick={() => handleViewChange('list')}
            title="Workspaces"
            className={`flex items-center whitespace-nowrap px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-300 ${
              view === 'list'
                ? 'bg-white dark:bg-white/10 text-zinc-900 dark:text-zinc-100 shadow-[0_2px_10px_rgba(0,0,0,0.06)] border border-black/[0.04] dark:border-white/[0.05]' 
                : 'text-zinc-500 dark:text-zinc-400 hover:bg-black/[0.03] dark:hover:bg-white/[0.05] hover:text-zinc-900 dark:hover:text-zinc-200 border border-transparent'
            } ${isDesktopCollapsed ? 'md:justify-center md:px-0 md:w-10 md:h-10' : ''}`}
          >
            <span className="flex-shrink-0 flex items-center justify-center w-5 h-5 text-zinc-500 dark:text-zinc-400 group-hover:text-zinc-900 dark:group-hover:text-zinc-100">
              <Layers size={18} />
            </span>
            <span 
              className={`transition-all duration-300 overflow-hidden text-left ${
                isDesktopCollapsed ? 'md:max-w-0 md:opacity-0 md:ml-0' : 'max-w-[180px] opacity-100 ml-2'
              }`}
            >
              Workspaces
            </span>
          </button>

          <button 
            onClick={() => handleViewChange('generation')}
            title="Image Generation"
            className={`flex items-center whitespace-nowrap px-3 py-2 rounded-xl text-[13px] font-medium transition-all duration-300 ${
              view === 'generation'
                ? 'bg-white dark:bg-white/10 text-zinc-900 dark:text-zinc-100 shadow-[0_2px_10px_rgba(0,0,0,0.06)] border border-black/[0.04] dark:border-white/[0.05]' 
                : 'text-zinc-500 dark:text-zinc-400 hover:bg-black/[0.03] dark:hover:bg-white/[0.05] hover:text-zinc-900 dark:hover:text-zinc-200 border border-transparent'
            } ${isDesktopCollapsed ? 'md:justify-center md:px-0 md:w-10 md:h-10 md:ml-0' : 'ml-6'}`}
          >
            <span className="flex-shrink-0 flex items-center justify-center w-5 h-5 text-zinc-500 dark:text-zinc-400 group-hover:text-zinc-900 dark:group-hover:text-zinc-100">
              <Sparkles size={16} />
            </span>
            <span 
              className={`transition-all duration-300 overflow-hidden text-left ${
                isDesktopCollapsed ? 'md:max-w-0 md:opacity-0 md:ml-0' : 'max-w-[180px] opacity-100 ml-2'
              }`}
            >
              Image Generation
            </span>
          </button>

          <button 
            onClick={() => handleViewChange('editing')}
            title="Image Editing"
            className={`flex items-center whitespace-nowrap px-3 py-2 rounded-xl text-[13px] font-medium transition-all duration-300 ${
              view === 'editing'
                ? 'bg-white dark:bg-white/10 text-zinc-900 dark:text-zinc-100 shadow-[0_2px_10px_rgba(0,0,0,0.06)] border border-black/[0.04] dark:border-white/[0.05]' 
                : 'text-zinc-500 dark:text-zinc-400 hover:bg-black/[0.03] dark:hover:bg-white/[0.05] hover:text-zinc-900 dark:hover:text-zinc-200 border border-transparent'
            } ${isDesktopCollapsed ? 'md:justify-center md:px-0 md:w-10 md:h-10 md:ml-0' : 'ml-6'}`}
          >
            <span className="flex-shrink-0 flex items-center justify-center w-5 h-5 text-zinc-500 dark:text-zinc-400 group-hover:text-zinc-900 dark:group-hover:text-zinc-100">
              <Edit3 size={16} />
            </span>
            <span 
              className={`transition-all duration-300 overflow-hidden text-left ${
                isDesktopCollapsed ? 'md:max-w-0 md:opacity-0 md:ml-0' : 'max-w-[180px] opacity-100 ml-2'
              }`}
            >
              Image Editing
            </span>
          </button>
        </div>
      </aside>

      {/* ================= 内容区 ================= */}
      <main className="flex-1 overflow-y-auto p-4 pt-16 md:p-8 relative custom-scrollbar">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-indigo-100/40 via-transparent to-transparent dark:from-indigo-900/10 pointer-events-none" />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_bottom_left,_var(--tw-gradient-stops))] from-purple-100/40 via-transparent to-transparent dark:from-purple-900/10 pointer-events-none" />
        
        <div className="relative z-10 max-w-7xl mx-auto h-full">
          {view === 'list' && renderList()}
          {view === 'generation' && renderGeneration()}
          {view === 'editing' && renderEditing()}
        </div>
      </main>
    </div>
  );
};

export default Demos;