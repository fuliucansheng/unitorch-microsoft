import { useState } from 'react';
import { UploadCloud, Wand2, Settings2, Image as ImageIcon, Download, Maximize2, Sparkles, SlidersHorizontal } from 'lucide-react';

const Demos = () => {
  const [prompt, setPrompt] = useState('');

  return (
    <div className="flex flex-col lg:flex-row h-full w-full overflow-y-auto lg:overflow-hidden">
      {/* 左侧控制面板 / 移动端顶部面板 */}
      <aside className="w-full lg:w-[400px] flex-shrink-0 flex flex-col border-b lg:border-b-0 lg:border-r border-zinc-200/80 dark:border-zinc-800/60 bg-white/80 dark:bg-zinc-950/50 backdrop-blur-xl relative z-10 lg:overflow-y-auto">
        <div className="p-4 lg:p-6 space-y-6 lg:space-y-8 flex-1">
          
          {/* 上传区域 */}
          <section className="space-y-3">
            <h2 className="text-sm font-medium text-zinc-700 dark:text-zinc-400 flex items-center gap-2">
              <ImageIcon size={16} /> 参考图像 (可选)
            </h2>
            <div className="group border border-dashed border-zinc-300 dark:border-zinc-700/60 rounded-2xl p-6 lg:p-8 flex flex-col items-center justify-center gap-3 bg-zinc-50/50 dark:bg-zinc-900/30 hover:bg-zinc-100 dark:hover:bg-zinc-800/40 hover:border-zinc-400 dark:hover:border-zinc-500 transition-all cursor-pointer">
              <div className="w-10 h-10 rounded-full bg-white dark:bg-zinc-800 shadow-sm dark:shadow-none flex items-center justify-center group-hover:scale-110 transition-transform">
                <UploadCloud size={20} className="text-zinc-500 dark:text-zinc-400 group-hover:text-zinc-700 dark:group-hover:text-zinc-200" />
              </div>
              <div className="text-center">
                <p className="text-sm text-zinc-700 dark:text-zinc-300 font-medium">点击上传或拖拽</p>
                <p className="text-xs text-zinc-500 mt-1">支持 JPG, PNG (最大 10MB)</p>
              </div>
            </div>
          </section>

          {/* 输入区域 */}
          <section className="space-y-3">
            <h2 className="text-sm font-medium text-zinc-700 dark:text-zinc-400 flex items-center gap-2">
              <Wand2 size={16} /> 提示词
            </h2>
            <div className="relative">
              <textarea 
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="描述你想要生成的画面内容..."
                className="w-full h-24 lg:h-32 bg-white dark:bg-zinc-900/50 border border-zinc-200 dark:border-zinc-800 rounded-2xl p-4 text-sm text-zinc-900 dark:text-zinc-100 placeholder:text-zinc-400 dark:placeholder:text-zinc-600 focus:outline-none focus:ring-1 focus:ring-orange-500/50 focus:border-orange-500/50 resize-none transition-all shadow-sm dark:shadow-none"
              />
              <div className="absolute bottom-3 right-3 text-xs text-zinc-400 dark:text-zinc-600">
                {prompt.length}/500
              </div>
            </div>
          </section>

          {/* 参数设置区 */}
          <section className="space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-medium text-zinc-700 dark:text-zinc-400 flex items-center gap-2">
                <Settings2 size={16} /> 核心参数
              </h2>
              <button className="text-xs text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300 flex items-center gap-1 transition-colors">
                <SlidersHorizontal size={12} /> 高级设置
              </button>
            </div>
            
            <div className="bg-zinc-50/80 dark:bg-zinc-900/40 border border-zinc-200 dark:border-zinc-800/60 rounded-2xl p-4 lg:p-5 space-y-4 lg:space-y-5">
              <div className="space-y-3">
                <div className="flex justify-between text-xs">
                  <span className="text-zinc-600 dark:text-zinc-400">创意度 (CFG Scale)</span>
                  <span className="text-zinc-900 dark:text-zinc-200 font-mono">7.5</span>
                </div>
                <input type="range" min="1" max="20" defaultValue="7.5" step="0.5" className="w-full accent-orange-500 bg-zinc-200 dark:bg-zinc-800 h-1.5 rounded-full appearance-none [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:bg-white dark:[&::-webkit-slider-thumb]:bg-zinc-200 [&::-webkit-slider-thumb]:shadow-md [&::-webkit-slider-thumb]:rounded-full cursor-pointer" />
              </div>
              
              <div className="space-y-3">
                <div className="flex justify-between text-xs">
                  <span className="text-zinc-600 dark:text-zinc-400">生成步数 (Steps)</span>
                  <span className="text-zinc-900 dark:text-zinc-200 font-mono">30</span>
                </div>
                <input type="range" min="10" max="100" defaultValue="30" step="1" className="w-full accent-orange-500 bg-zinc-200 dark:bg-zinc-800 h-1.5 rounded-full appearance-none [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:bg-white dark:[&::-webkit-slider-thumb]:bg-zinc-200 [&::-webkit-slider-thumb]:shadow-md [&::-webkit-slider-thumb]:rounded-full cursor-pointer" />
              </div>
            </div>
          </section>
        </div>

        {/* 生成按钮 */}
        <div className="p-4 lg:p-6 border-t border-zinc-200/80 dark:border-zinc-800/60 bg-white/80 dark:bg-zinc-950/80 backdrop-blur-md sticky bottom-0 lg:static z-20">
          <button className="w-full py-3 lg:py-3.5 px-4 bg-gradient-to-r from-orange-500 to-red-500 hover:from-orange-600 hover:to-red-600 text-white rounded-xl font-medium text-sm flex items-center justify-center gap-2 shadow-lg shadow-orange-500/20 transition-all hover:shadow-orange-500/40 active:scale-[0.98]">
            <Sparkles size={18} />
            立即生成图像
          </button>
        </div>
      </aside>

      {/* 右侧结果预览区 / 移动端下方区域 */}
      <main className="flex-1 relative flex flex-col min-h-[400px] lg:min-h-0 bg-zinc-100/50 dark:bg-[#0a0a0c]">
        {/* 氛围渐变背景 */}
        <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/5 via-purple-500/5 to-transparent pointer-events-none" />
        
        {/* 顶部工具栏 */}
        <header className="h-14 lg:h-16 border-b border-zinc-200/80 dark:border-zinc-800/40 flex items-center justify-end px-4 lg:px-6 relative z-10">
          <div className="flex items-center gap-2 lg:gap-3">
            <button className="p-2 text-zinc-500 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100 hover:bg-white dark:hover:bg-zinc-800/50 rounded-lg transition-colors shadow-sm dark:shadow-none">
              <Download size={18} />
            </button>
            <button className="p-2 text-zinc-500 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100 hover:bg-white dark:hover:bg-zinc-800/50 rounded-lg transition-colors shadow-sm dark:shadow-none">
              <Maximize2 size={18} />
            </button>
          </div>
        </header>

        {/* 预览画布 */}
        <div className="flex-1 p-4 lg:p-8 flex items-center justify-center overflow-hidden relative z-10">
          <div className="w-full max-w-4xl aspect-square sm:aspect-[4/3] lg:aspect-[16/9] bg-white dark:bg-zinc-900/50 border border-zinc-200 dark:border-zinc-800/60 rounded-2xl lg:rounded-3xl shadow-lg lg:shadow-xl dark:shadow-2xl flex items-center justify-center relative overflow-hidden group">
            
            {/* 占位状态 / 待生成状态 */}
            <div className="text-center space-y-3 lg:space-y-4 flex flex-col items-center p-6">
              <div className="w-14 h-14 lg:w-16 lg:h-16 rounded-2xl bg-zinc-50 dark:bg-zinc-800/50 flex items-center justify-center border border-zinc-100 dark:border-zinc-700/50">
                <ImageIcon size={28} className="text-zinc-400 dark:text-zinc-600 lg:w-8 lg:h-8" />
              </div>
              <div>
                <p className="text-zinc-900 dark:text-zinc-400 font-medium text-sm lg:text-base">画布准备就绪</p>
                <p className="text-zinc-500 dark:text-zinc-600 text-xs lg:text-sm mt-1">输入提示词开始创作</p>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Demos;