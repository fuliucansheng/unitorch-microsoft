import { NavLink, Outlet } from 'react-router-dom';

const Picasso = () => {
  return (
    <div className="flex flex-1 h-full overflow-hidden bg-zinc-50/50 dark:bg-[#0a0a0c]">
      {/* 侧边栏 */}
      <aside className="w-64 flex-shrink-0 border-r border-zinc-200/80 dark:border-zinc-800/60 bg-white/50 dark:bg-zinc-950/30 backdrop-blur-md flex flex-col p-4 space-y-2">
        <div className="text-xs font-semibold text-zinc-400 dark:text-zinc-500 mb-2 px-2 uppercase tracking-wider">
          Picasso Menu
        </div>
        
        <NavLink 
          to="/picasso/overview" 
          className={({ isActive }) => 
            `flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all ${
              isActive 
                ? 'bg-zinc-100 dark:bg-zinc-800/80 text-zinc-900 dark:text-zinc-100 shadow-sm' 
                : 'text-zinc-600 dark:text-zinc-400 hover:bg-zinc-100/50 dark:hover:bg-zinc-800/50 hover:text-zinc-900 dark:hover:text-zinc-200'
            }`
          }
        >
          <span className="text-lg">🌐</span>
          Overview
        </NavLink>

        <NavLink 
          to="/picasso/examples" 
          className={({ isActive }) => 
            `flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all ${
              isActive 
                ? 'bg-zinc-100 dark:bg-zinc-800/80 text-zinc-900 dark:text-zinc-100 shadow-sm' 
                : 'text-zinc-600 dark:text-zinc-400 hover:bg-zinc-100/50 dark:hover:bg-zinc-800/50 hover:text-zinc-900 dark:hover:text-zinc-200'
            }`
          }
        >
          <span className="text-lg">🎢</span>
          Examples
        </NavLink>
      </aside>

      {/* 内容区 */}
      <main className="flex-1 overflow-y-auto p-8 relative custom-scrollbar">
        <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/5 via-purple-500/5 to-transparent pointer-events-none" />
        <div className="relative z-10 h-full">
          <Outlet />
        </div>
      </main>
    </div>
  );
};

export default Picasso;