const Overview = () => {
  return (
    <div className="max-w-4xl space-y-6">
      <div>
        <h2 className="text-3xl font-bold text-zinc-900 dark:text-zinc-100 tracking-tight">Overview</h2>
        <p className="text-zinc-500 dark:text-zinc-400 mt-2">Welcome to the Picasso generative models hub.</p>
      </div>

      <div className="bg-white dark:bg-zinc-900/60 border border-zinc-200 dark:border-zinc-800/60 rounded-2xl p-8 shadow-sm">
        <h3 className="text-xl font-semibold text-zinc-900 dark:text-zinc-100 mb-4">Dashboard</h3>
        <p className="text-zinc-600 dark:text-zinc-400 leading-relaxed text-base">
          This Dashboard provides a comprehensive view of the Picasso image generative models performance and progress. It serves as a centralized hub for tracking key metrics related to the models powering Picasso, ensuring stakeholders and developers can monitor model health, accuracy, and efficiency.
        </p>
      </div>
    </div>
  );
};

export default Overview;