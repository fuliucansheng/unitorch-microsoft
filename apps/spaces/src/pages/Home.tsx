const Home = () => {
  return (
    <div className="flex-1 flex flex-col items-center justify-center p-8 text-center relative overflow-hidden bg-transparent">
      {/* 高级感的光晕背景 */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[600px] bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-indigo-200/30 via-purple-200/20 to-transparent dark:from-indigo-500/10 dark:via-purple-500/5 rounded-full blur-3xl pointer-events-none opacity-70" />
      
      <div className="max-w-3xl space-y-8 relative z-10">
        <div className="relative inline-block">
          <div className="absolute inset-0 bg-gradient-to-tr from-indigo-500/20 to-purple-500/20 blur-2xl rounded-full" />
          <img src="/icon.ico" alt="Ads Spaces" className="w-24 h-24 mx-auto relative drop-shadow-xl" />
        </div>
        
        <div className="space-y-4">
          <h1 className="text-4xl sm:text-5xl md:text-6xl font-bold text-zinc-900 dark:text-zinc-100 tracking-tight">
            Welcome to <span className="text-transparent bg-clip-text bg-gradient-to-r from-zinc-900 to-zinc-500 dark:from-zinc-100 dark:to-zinc-500">Ads Spaces</span>
          </h1>
          <p className="text-lg sm:text-xl text-zinc-500 dark:text-zinc-400 leading-relaxed font-light max-w-2xl mx-auto">
            A collaborative site for sharing, experimenting, and discovering models, datasets, and documentation tailored for ads scenarios.
          </p>
        </div>

        <div className="pt-8 flex items-center justify-center gap-4">
          <button className="px-6 py-3 rounded-full bg-zinc-900 text-white dark:bg-white dark:text-zinc-900 font-medium text-sm hover:scale-105 transition-transform shadow-[0_8px_30px_rgb(0,0,0,0.12)] dark:shadow-[0_8px_30px_rgba(255,255,255,0.1)]">
            Explore Models
          </button>
          <button className="px-6 py-3 rounded-full bg-white dark:bg-zinc-900 text-zinc-900 dark:text-white font-medium text-sm border border-black/[0.08] dark:border-white/[0.08] hover:bg-zinc-50 dark:hover:bg-zinc-800 transition-colors shadow-sm">
            View Documentation
          </button>
        </div>
      </div>
    </div>
  );
};

export default Home;